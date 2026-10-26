"""
Distributed Big Data ETL & Feature Engineering Pipeline using Apache PySpark
Extracts, transforms, and loads Synthea relational EHR tables into an
analytical Parquet Lakehouse partitioned by demographic subgroups.

Author: Ankith Umadi, Haneesh E, Vikas R (PES University BDSRAE Team)
"""

import os
import time
from pyspark.sql import functions as F
from pyspark.sql.window import Window
from src.big_data.spark_session import get_spark_session


def run_etl_pipeline(raw_dir="data/raw", processed_dir="data/processed", spark=None):
    """
    Executes distributed PySpark ETL:
    1. Ingests Synthea relational CSVs (patients, encounters, conditions, medications, procedures)
    2. Builds 30-day hospital readmission cohort from inpatient index admissions
    3. Engineers comorbidity, polypharmacy, and utilization features
    4. Writes partitioned Parquet lakehouse storage
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

    print(f"    - Patients raw count:    {df_patients.count():,}")
    print(f"    - Encounters raw count:  {df_encounters.count():,}")

    # 2. Clean timestamps and patient demographics
    df_patients_clean = (
        df_patients
        .select(
            F.col("Id").alias("patient_id"),
            F.to_date(F.col("BIRTHDATE")).alias("birth_date"),
            F.lower(F.col("GENDER")).alias("gender"),
            F.lower(F.col("RACE")).alias("race"),
            F.lower(F.col("ETHNICITY")).alias("ethnicity"),
            F.col("CITY").alias("city"),
            F.col("ZIP").alias("zip_code"),
            F.col("HEALTHCARE_EXPENSES").cast("double").alias("healthcare_expenses"),
            F.col("HEALTHCARE_COVERAGE").cast("double").alias("healthcare_coverage")
        )
        .withColumn("gender", F.when(F.col("gender") == "f", "Female").otherwise("Male"))
        .withColumn("race", F.initcap(F.col("race")))
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

    # 5. Distributed Window Analytics for Prior Utilization & 30-Day Readmission Target
    patient_window = Window.partitionBy("patient_id").orderBy("enc_start")

    # Lead to identify next hospital encounter
    df_enc_windowed = (
        df_enc_with_pat
        .withColumn("next_enc_start", F.lead("enc_start", 1).over(patient_window))
        .withColumn("next_enc_class", F.lead("encounter_class", 1).over(patient_window))
        .withColumn("days_to_next_enc", 
            (F.col("next_enc_start").cast("long") - F.col("enc_stop").cast("long")) / 86400.0
        )
    )

    # Filter to index admissions (inpatient stays)
    df_index_admissions = df_enc_windowed.filter(F.col("encounter_class") == "inpatient")

    # Target: 30-day readmission = next encounter is inpatient/emergency and occurred within 30 days of discharge
    df_cohort_labeled = (
        df_index_admissions
        .withColumn(
            "readmission_30d",
            F.when(
                (F.col("days_to_next_enc") >= 0.0) & 
                (F.col("days_to_next_enc") <= 30.0) & 
                (F.col("next_enc_class").isin("inpatient", "emergency")),
                1
            ).otherwise(0)
        )
    )

    # 6. Distributed Feature Extraction: Conditions, Medications, Procedures
    # Chronic comorbidity indicators
    df_conditions_clean = (
        df_conditions
        .select(
            F.col("PATIENT").alias("patient_id"),
            F.lower(F.col("DESCRIPTION")).alias("condition_desc")
        )
    )

    df_comorbidities = (
        df_conditions_clean
        .groupBy("patient_id")
        .agg(
            F.max(F.when(F.col("condition_desc").contains("hypertension"), 1).otherwise(0)).alias("has_hypertension"),
            F.max(F.when(F.col("condition_desc").contains("diabetes"), 1).otherwise(0)).alias("has_diabetes"),
            F.max(F.when(F.col("condition_desc").contains("heart failure"), 1).otherwise(0)).alias("has_heart_failure"),
            F.max(F.when(F.col("condition_desc").contains("obstructive pulmonary"), 1).otherwise(0)).alias("has_copd"),
            F.max(F.when(F.col("condition_desc").contains("kidney"), 1).otherwise(0)).alias("has_ckd"),
            F.max(F.when(F.col("condition_desc").contains("coronary"), 1).otherwise(0)).alias("has_cad"),
            F.max(F.when(F.col("condition_desc").contains("depression"), 1).otherwise(0)).alias("has_depression"),
            F.countDistinct("condition_desc").alias("comorbidity_count")
        )
    )

    # Polypharmacy medication counts
    df_med_counts = (
        df_medications
        .groupBy(F.col("PATIENT").alias("patient_id"))
        .agg(F.countDistinct("CODE").alias("total_medications"))
    )

    # Procedures count
    df_proc_counts = (
        df_procedures
        .groupBy(F.col("PATIENT").alias("patient_id"))
        .agg(F.countDistinct("CODE").alias("total_procedures"))
    )

    # Prior utilization summary per patient
    df_prior_util = (
        df_enc_clean
        .groupBy("patient_id")
        .agg(
            F.sum(F.when(F.col("encounter_class") == "inpatient", 1).otherwise(0)).alias("num_prior_inpatient"),
            F.sum(F.when(F.col("encounter_class") == "emergency", 1).otherwise(0)).alias("num_prior_emergency"),
            F.sum(F.when(F.col("encounter_class") == "ambulatory", 1).otherwise(0)).alias("num_prior_ambulatory")
        )
    )

    # 7. Final Join to assemble the Analytical Feature Store
    df_final_cohort = (
        df_cohort_labeled
        .join(df_comorbidities, on="patient_id", how="left")
        .join(df_med_counts, on="patient_id", how="left")
        .join(df_proc_counts, on="patient_id", how="left")
        .join(df_prior_util, on="patient_id", how="left")
        .na.fill({
            "has_hypertension": 0, "has_diabetes": 0, "has_heart_failure": 0,
            "has_copd": 0, "has_ckd": 0, "has_cad": 0, "has_depression": 0,
            "comorbidity_count": 0, "total_medications": 0, "total_procedures": 0,
            "num_prior_inpatient": 1, "num_prior_emergency": 0, "num_prior_ambulatory": 0
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
    print(f"    - Index Admissions:     {cohort_count:,}")
    print(f"    - 30-Day Readmissions:  {readm_count:,} ({readm_rate:.2f}%)")

    # 8. Partitioned Parquet Lakehouse Storage Design
    # Partition by age_group and race to optimize demographic query pruning & subgroup isolation
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
