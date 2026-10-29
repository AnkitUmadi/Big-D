"""
Distributed Big Data ETL & Feature Engineering Pipeline using Apache PySpark
Extracts, transforms, and loads Synthea relational EHR tables into an
analytical Parquet Lakehouse partitioned by demographic subgroups.

Implements leak-free feature engineering and clinically valid 30-day
inpatient readmission target definition.

Author: Ankith Umadi, Haneesh E, Vikas R (PES University BDSRAE Team)
"""

import os
import sys
import time
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from pyspark.sql import functions as F
from pyspark.sql.window import Window
from src.big_data.spark_session import get_spark_session


def run_etl_pipeline(raw_dir="data/raw", processed_dir="data/processed", spark=None):
    """
    Executes distributed PySpark ETL:
    1. Ingests Synthea relational CSVs (patients, encounters, conditions, medications, procedures)
    2. Builds 30-day hospital readmission cohort from inpatient index admissions with censoring handling
    3. Prevents temporal data leakage: features computed strictly on or before index discharge
    4. Writes partitioned Parquet lakehouse storage partitioned by [age_group, race]
    """
    start_time = time.time()
    own_spark = False
    if spark is None:
        spark = get_spark_session()
        own_spark = True

    print("[*] [PySpark ETL] Ingesting raw Synthea EHR tables from:", raw_dir)

    # 1. Read raw CSVs with distributed schema inference
    df_patients = spark.read.option("header", "true").csv(os.path.join(raw_dir, "patients.csv"))
    df_encounters = spark.read.option("header", "true").csv(os.path.join(raw_dir, "encounters.csv"))
    df_conditions = spark.read.option("header", "true").csv(os.path.join(raw_dir, "conditions.csv"))
    df_medications = spark.read.option("header", "true").csv(os.path.join(raw_dir, "medications.csv"))
    df_procedures = spark.read.option("header", "true").csv(os.path.join(raw_dir, "procedures.csv"))

    raw_pat_count = df_patients.count()
    raw_enc_count = df_encounters.count()
    print(f"    - Patients raw count:    {raw_pat_count:,}")
    print(f"    - Encounters raw count:  {raw_enc_count:,}")

    # 2. Clean timestamps and patient demographics (preserving source categories)
    df_patients_clean = (
        df_patients
        .select(
            F.col("Id").alias("patient_id"),
            F.to_date(F.col("BIRTHDATE")).alias("birth_date"),
            F.col("GENDER").alias("raw_gender"),
            F.col("RACE").alias("raw_race"),
            F.col("ETHNICITY").alias("raw_ethnicity"),
            F.col("CITY").alias("city"),
            F.col("ZIP").alias("zip_code"),
            F.col("HEALTHCARE_EXPENSES").cast("double").alias("healthcare_expenses"),
            F.col("HEALTHCARE_COVERAGE").cast("double").alias("healthcare_coverage")
        )
        .withColumn(
            "gender",
            F.when(F.upper(F.col("raw_gender")) == "F", "Female")
            .when(F.upper(F.col("raw_gender")) == "M", "Male")
            .otherwise(F.coalesce(F.col("raw_gender"), F.lit("Unknown")))
        )
        .withColumn("race", F.initcap(F.coalesce(F.col("raw_race"), F.lit("Unknown"))))
        .withColumn("ethnicity", F.initcap(F.coalesce(F.col("raw_ethnicity"), F.lit("Unknown"))))
    )

    # 3. Clean encounters
    df_enc_clean = (
        df_encounters
        .select(
            F.col("Id").alias("encounter_id"),
            F.to_timestamp(F.col("START")).alias("enc_start"),
            F.to_timestamp(F.col("STOP")).alias("enc_stop"),
            F.col("PATIENT").alias("patient_id"),
            F.lower(F.col("ENCOUNTERCLASS")).alias("encounter_class"),
            F.col("DESCRIPTION").alias("encounter_desc"),
            F.col("BASE_ENCOUNTER_COST").cast("double").alias("base_cost"),
            F.col("TOTAL_CLAIM_COST").cast("double").alias("total_claim_cost"),
            F.col("REASONDESCRIPTION").alias("reason_desc")
        )
    )

    # Compute maximum observation date in encounters to detect boundary censoring
    max_study_date = df_enc_clean.select(F.max("enc_stop")).collect()[0][0]
    print(f"    - Dataset observation horizon cutoff: {max_study_date}")

    # 4. Join encounters with patient birthdate to calculate age at admission
    df_enc_with_pat = (
        df_enc_clean
        .join(df_patients_clean, on="patient_id", how="inner")
        .withColumn("age", F.floor(F.datediff(F.to_date("enc_start"), F.col("birth_date")) / 365.25))
        .withColumn("age_group",
            F.when(F.col("age") < 18, "Pediatric (<18)")
            .when(F.col("age") < 45, "Young Adult (18-44)")
            .when(F.col("age") < 65, "Middle Age (45-64)")
            .otherwise("Geriatric (65+)")
        )
        .withColumn("length_of_stay", 
            F.greatest(F.lit(1.0), F.round((F.col("enc_stop").cast("long") - F.col("enc_start").cast("long")) / 86400.0, 1))
        )
    )

    # 5. Cohort Selection: Inpatient Index Admissions
    df_index_admissions = (
        df_enc_with_pat
        .filter(F.col("encounter_class") == "inpatient")
        .select(
            "encounter_id", "patient_id", "enc_start", "enc_stop",
            "age", "age_group", "gender", "race", "ethnicity",
            "length_of_stay", "total_claim_cost"
        )
    )
    total_inpatient_admissions = df_index_admissions.count()

    # 6. Target Construction: Next Qualifying Inpatient Admission within 30 Days
    # Self-join inpatient stays on patient_id where nxt.enc_start > cur.enc_stop
    # This prevents:
    # (a) Intervening ambulatory visits from masking readmissions
    # (b) Overlapping/concurrent episodes (nxt.enc_start <= cur.enc_stop) from being labeled as readmissions
    # (c) Emergency visits from being conflated with inpatient readmissions
    df_next_inpatient = (
        df_index_admissions.alias("cur")
        .join(
            df_index_admissions.alias("nxt"),
            on=(F.col("cur.patient_id") == F.col("nxt.patient_id")) & 
               (F.col("nxt.enc_start") > F.col("cur.enc_stop")),
            how="left"
        )
        .groupBy("cur.encounter_id")
        .agg(F.min("nxt.enc_start").alias("next_inpatient_start"))
    )

    # Join next inpatient admission date and compute gap
    df_cohort_with_gap = (
        df_index_admissions
        .join(df_next_inpatient, on="encounter_id", how="left")
        .withColumn(
            "days_to_next_inpatient",
            (F.col("next_inpatient_start").cast("long") - F.col("enc_stop").cast("long")) / 86400.0
        )
        .withColumn(
            "has_sufficient_followup",
            (F.col("next_inpatient_start").isNotNull()) |
            (F.col("enc_stop") <= (F.lit(max_study_date) - F.expr("INTERVAL 30 DAYS")))
        )
        .withColumn(
            "readmission_30d",
            F.when(
                (F.col("days_to_next_inpatient") > 0.0) &
                (F.col("days_to_next_inpatient") <= 30.0),
                1
            ).otherwise(0)
        )
    )

    # Censor Handling: Exclude index admissions that occur < 30 days before dataset horizon without readmission
    censored_count = df_cohort_with_gap.filter(~F.col("has_sufficient_followup")).count()
    if censored_count > 0:
        print(f"    - Censor Handling: Excluded {censored_count} index admissions occurring <30 days before dataset cutoff without readmission.")
    df_cohort_labeled = df_cohort_with_gap.filter(F.col("has_sufficient_followup"))

    # 7. Leak-Free Feature Engineering (Strictly Prior to / At Discharge Cutoff)
    # (A) Prior Utilization: count encounters with enc_start < index enc_start
    df_prior_util = (
        df_cohort_labeled.select("encounter_id", "patient_id", "enc_start")
        .join(
            df_enc_clean.select(
                F.col("patient_id"),
                F.col("enc_start").alias("hist_enc_start"),
                F.col("encounter_class").alias("hist_enc_class")
            ),
            on="patient_id",
            how="inner"
        )
        .filter(F.col("hist_enc_start") < F.col("enc_start"))
        .groupBy("encounter_id")
        .agg(
            F.sum(F.when(F.col("hist_enc_class") == "inpatient", 1).otherwise(0)).alias("num_prior_inpatient"),
            F.sum(F.when(F.col("hist_enc_class") == "emergency", 1).otherwise(0)).alias("num_prior_emergency"),
            F.sum(F.when(F.col("hist_enc_class") == "ambulatory", 1).otherwise(0)).alias("num_prior_ambulatory")
        )
    )

    # (B) Chronic Comorbidities: conditions diagnosed on or before index discharge
    df_conditions_clean = (
        df_conditions
        .select(
            F.col("PATIENT").alias("patient_id"),
            F.to_date(F.col("START")).alias("cond_start"),
            F.to_date(F.col("STOP")).alias("cond_stop"),
            F.lower(F.col("DESCRIPTION")).alias("condition_desc")
        )
    )

    df_comorbidities = (
        df_cohort_labeled.select("encounter_id", "patient_id", "enc_stop")
        .join(df_conditions_clean, on="patient_id", how="inner")
        .filter(
            (F.col("cond_start") <= F.to_date(F.col("enc_stop"))) &
            (F.col("cond_stop").isNull() | (F.col("cond_stop") >= F.to_date(F.col("enc_stop"))))
        )
        .groupBy("encounter_id")
        .agg(
            F.max(F.when(F.col("condition_desc").contains("hypertension"), 1).otherwise(0)).alias("has_hypertension"),
            F.max(F.when(F.col("condition_desc").contains("diabetes"), 1).otherwise(0)).alias("has_diabetes"),
            F.max(F.when(F.col("condition_desc").contains("heart failure"), 1).otherwise(0)).alias("has_heart_failure"),
            F.max(F.when(F.col("condition_desc").contains("obstructive pulmonary"), 1).otherwise(0)).alias("has_copd"),
            F.max(F.when(F.col("condition_desc").contains("kidney"), 1).otherwise(0)).alias("has_ckd"),
            F.max(F.when(F.col("condition_desc").contains("coronary"), 1).otherwise(0)).alias("has_cad"),
            F.max(F.when(F.col("condition_desc").contains("depress"), 1).otherwise(0)).alias("has_depression"),
            F.countDistinct("condition_desc").alias("comorbidity_count")
        )
    )

    # (C) Polypharmacy / Active Medications on or before discharge
    df_medications_clean = (
        df_medications
        .select(
            F.col("PATIENT").alias("patient_id"),
            F.to_date(F.col("START")).alias("med_start"),
            F.col("CODE").alias("med_code")
        )
    )
    df_med_counts = (
        df_cohort_labeled.select("encounter_id", "patient_id", "enc_stop")
        .join(df_medications_clean, on="patient_id", how="inner")
        .filter(F.col("med_start") <= F.to_date(F.col("enc_stop")))
        .groupBy("encounter_id")
        .agg(F.countDistinct("med_code").alias("total_medications"))
    )

    # (D) Procedures performed on or before discharge
    df_procedures_clean = (
        df_procedures
        .select(
            F.col("PATIENT").alias("patient_id"),
            F.to_date(F.col("DATE")).alias("proc_date"),
            F.col("CODE").alias("proc_code")
        )
    )
    df_proc_counts = (
        df_cohort_labeled.select("encounter_id", "patient_id", "enc_stop")
        .join(df_procedures_clean, on="patient_id", how="inner")
        .filter(F.col("proc_date") <= F.to_date(F.col("enc_stop")))
        .groupBy("encounter_id")
        .agg(F.countDistinct("proc_code").alias("total_procedures"))
    )

    # 8. Assemble Final Analytical Lakehouse Cohort
    df_final_cohort = (
        df_cohort_labeled
        .join(df_prior_util, on="encounter_id", how="left")
        .join(df_comorbidities, on="encounter_id", how="left")
        .join(df_med_counts, on="encounter_id", how="left")
        .join(df_proc_counts, on="encounter_id", how="left")
        .na.fill({
            "has_hypertension": 0, "has_diabetes": 0, "has_heart_failure": 0,
            "has_copd": 0, "has_ckd": 0, "has_cad": 0, "has_depression": 0,
            "comorbidity_count": 0, "total_medications": 0, "total_procedures": 0,
            "num_prior_inpatient": 0, "num_prior_emergency": 0, "num_prior_ambulatory": 0
        })
        .select(
            "encounter_id", "patient_id",
            "age", "age_group", "gender", "race", "ethnicity",
            "length_of_stay", "num_prior_inpatient", "num_prior_emergency", "num_prior_ambulatory",
            "comorbidity_count", "has_diabetes", "has_hypertension", "has_heart_failure",
            "has_copd", "has_ckd", "has_cad", "has_depression",
            "total_medications", "total_procedures", "total_claim_cost",
            "readmission_30d"
        )
    )

    cohort_count = df_final_cohort.count()
    readm_count = df_final_cohort.filter(F.col("readmission_30d") == 1).count()
    readm_rate = (readm_count / cohort_count) * 100.0 if cohort_count > 0 else 0.0

    print(f"[+] Final Inpatient Readmission Cohort assembled:")
    print(f"    - Index Admissions:     {cohort_count:,} (of {total_inpatient_admissions:,} total inpatient stays)")
    print(f"    - 30-Day Readmissions:  {readm_count:,} ({readm_rate:.2f}%)")

    # 9. Partitioned Parquet Lakehouse Storage Design
    print(f"[*] Writing analytical Parquet lakehouse partitioned by [age_group, race] to: {processed_dir}")
    os.makedirs(processed_dir, exist_ok=True)
    
    parquet_path = os.path.join(processed_dir, "readmission_lakehouse.parquet")
    (
        df_final_cohort
        .write
        .mode("overwrite")
        .partitionBy("age_group", "race")
        .parquet(parquet_path)
    )

    elapsed = time.time() - start_time
    print(f"[OK] PySpark ETL completed in {elapsed:.2f} seconds.")

    if own_spark:
        spark.stop()

    return parquet_path


def load_processed_data(processed_dir="data/processed"):
    """
    Reads the partitioned Parquet lakehouse into a Pandas DataFrame for downstream
    scikit-learn and Fairlearn modeling & audits.
    """
    import pandas as pd
    parquet_path = os.path.join(processed_dir, "readmission_lakehouse.parquet")
    if not os.path.exists(parquet_path):
        raise FileNotFoundError(f"Parquet dataset not found at {parquet_path}. Run ETL pipeline first.")
    df = pd.read_parquet(parquet_path)
    return df


if __name__ == "__main__":
    run_etl_pipeline()
