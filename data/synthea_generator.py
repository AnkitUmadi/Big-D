"""
Synthea-Style Synthetic EHR Generator for Hospital 30-Day Readmission Analysis
Generates realistic EHR cohorts conforming to Synthea's relational CSV schema.
Includes authentic demographic distributions, chronic disease progressions,
and documented socioeconomic disparity mechanisms in clinical readmissions.

Author: Ankith Umadi, Haneesh E, Vikas R (PES University BDSRAE Team)
"""

import os
import sys
import uuid
import random
import datetime
import argparse
import numpy as np
import pandas as pd


# Set deterministic random seed
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

CHRONIC_CONDITIONS = [
    {"code": "38341003", "name": "Essential hypertension", "prob_base": 0.32, "age_factor": 1.8},
    {"code": "44054006", "name": "Type 2 diabetes mellitus", "prob_base": 0.18, "age_factor": 1.6},
    {"code": "84114007", "name": "Heart failure", "prob_base": 0.08, "age_factor": 2.2},
    {"code": "13645005", "name": "Chronic obstructive pulmonary disease", "prob_base": 0.07, "age_factor": 1.9},
    {"code": "709044004", "name": "Chronic kidney disease stage 3", "prob_base": 0.06, "age_factor": 2.1},
    {"code": "53741008", "name": "Coronary artery disease", "prob_base": 0.09, "age_factor": 2.0},
    {"code": "35489007", "name": "Depressive disorder", "prob_base": 0.14, "age_factor": 0.9},
    {"code": "195967001", "name": "Asthma", "prob_base": 0.11, "age_factor": 0.7},
]

MEDICATIONS = [
    {"code": "310798", "name": "Lisinopril 10 MG Oral Tablet", "condition": "Essential hypertension"},
    {"code": "860975", "name": "Metformin hydrochloride 500 MG Oral Tablet", "condition": "Type 2 diabetes mellitus"},
    {"code": "855332", "name": "Atorvastatin 20 MG Oral Tablet", "condition": "Coronary artery disease"},
    {"code": "200845", "name": "Furosemide 40 MG Oral Tablet", "condition": "Heart failure"},
    {"code": "745678", "name": "Albuterol 0.09 MG/ACTUAT Inhalation Aerosol", "condition": "Asthma"},
    {"code": "312961", "name": "Sertraline 50 MG Oral Tablet", "condition": "Depressive disorder"},
    {"code": "311904", "name": "Amlodipine 5 MG Oral Tablet", "condition": "Essential hypertension"},
]

PROCEDURES = [
    {"code": "40701008", "name": "Echocardiography", "base_cost": 750.0},
    {"code": "261272006", "name": "Diagnostic chest X-ray", "base_cost": 220.0},
    {"code": "415070008", "name": "Percutaneous coronary intervention", "base_cost": 6500.0},
    {"code": "73761001", "name": "Colonoscopy", "base_cost": 1200.0},
    {"code": "104091002", "name": "Measurement of respiratory function", "base_cost": 310.0},
    {"code": "252465000", "name": "Electrocardiogram", "base_cost": 180.0},
]

CITIES_MA = [
    ("Boston", "02115", 42.3384, -71.1026),
    ("Worcester", "01608", 42.2626, -71.8023),
    ("Springfield", "01103", 42.1015, -72.5898),
    ("Cambridge", "02138", 42.3736, -71.1097),
    ("Lowell", "01852", 42.6334, -71.3162),
    ("Brockton", "02301", 42.0834, -71.0184),
    ("New Bedford", "02740", 41.6362, -70.9342),
    ("Quincy", "02169", 42.2529, -71.0023),
]


def generate_synthea_cohort(n_patients=30000, output_dir="data/raw"):
    """
    Generate synthetic EHR relational tables matching Synthea CSV standards.
    """
    os.makedirs(output_dir, exist_ok=True)
    print(f"[*] Generating {n_patients:,} synthetic patient EHR records...")

    # Reference study period: 2021-01-01 to 2024-12-31
    study_start = datetime.date(2021, 1, 1)
    study_end = datetime.date(2024, 12, 31)
    total_days = (study_end - study_start).days

    # 1. Patients table
    patients = []
    races = ["white", "black", "hispanic", "asian", "native", "other"]
    race_weights = [0.63, 0.15, 0.13, 0.06, 0.015, 0.015]

    for _ in range(n_patients):
        pid = str(uuid.uuid4())
        gender = np.random.choice(["M", "F"], p=[0.49, 0.51])
        race = np.random.choice(races, p=race_weights)
        ethnicity = "hispanic" if race == "hispanic" else ("hispanic" if random.random() < 0.05 else "nonhispanic")
        
        # Age distribution realistic for healthcare populations:
        # 10% pediatric (<18), 30% young adult (18-44), 35% middle (45-64), 25% geriatric (65+)
        age_tier = np.random.choice(["pediatric", "young_adult", "middle", "geriatric"], p=[0.10, 0.30, 0.35, 0.25])
        if age_tier == "pediatric":
            age = int(np.random.uniform(1, 17))
        elif age_tier == "young_adult":
            age = int(np.random.uniform(18, 44))
        elif age_tier == "middle":
            age = int(np.random.uniform(45, 64))
        else:
            age = int(np.random.uniform(65, 92))
        
        birth_year = 2024 - age
        birth_month = random.randint(1, 12)
        birth_day = random.randint(1, 28)
        birth_date = datetime.date(birth_year, birth_month, birth_day)

        city_info = random.choice(CITIES_MA)
        marital = "M" if age > 25 and random.random() < 0.58 else ("S" if age > 18 else "")

        # Healthcare expense disparity (synthetic insurance coverage disparity)
        base_expense = float(age * 180 + np.random.gamma(shape=2.0, scale=1200))
        coverage_rate = 0.85 if race == "white" else (0.76 if race == "black" else 0.72)
        coverage = float(base_expense * coverage_rate * random.uniform(0.8, 1.0))

        patients.append({
            "Id": pid,
            "BIRTHDATE": str(birth_date),
            "DEATHDATE": "",
            "SSN": f"999-{random.randint(10,99)}-{random.randint(1000,9999)}",
            "FIRST": f"First_{pid[:6]}",
            "LAST": f"Last_{pid[-6:]}",
            "MARITAL": marital,
            "RACE": race,
            "ETHNICITY": ethnicity,
            "GENDER": gender,
            "CITY": city_info[0],
            "STATE": "Massachusetts",
            "ZIP": city_info[1],
            "LAT": city_info[2],
            "LON": city_info[3],
            "HEALTHCARE_EXPENSES": round(base_expense, 2),
            "HEALTHCARE_COVERAGE": round(coverage, 2),
            "age": age,
            "age_tier": age_tier
        })

    df_patients = pd.DataFrame(patients)

    # 2. Conditions, Encounters, Medications, Procedures
    encounters = []
    conditions = []
    medications = []
    procedures = []

    print("[*] Generating clinical encounters, chronic disease trajectories & admissions...")

    for _, pat in df_patients.iterrows():
        pid = pat["Id"]
        age = pat["age"]
        race = pat["RACE"]
        gender = pat["GENDER"]

        # Determine patient chronic disease profile
        patient_conditions = []
        for cond in CHRONIC_CONDITIONS:
            # Chronic disease probability increases with age and socioeconomic risk
            race_multiplier = 1.35 if (race in ["black", "hispanic"] and cond["name"] in ["Essential hypertension", "Type 2 diabetes mellitus"]) else 1.0
            prob = min(0.85, cond["prob_base"] * (1.0 + (age / 70.0) * cond["age_factor"]) * race_multiplier)
            if random.random() < prob:
                onset_years_ago = random.randint(1, max(2, min(age - 10, 20)))
                onset_date = study_end - datetime.timedelta(days=int(onset_years_ago * 365.25))
                cid = str(uuid.uuid4())
                cond_record = {
                    "START": str(onset_date),
                    "STOP": "",
                    "PATIENT": pid,
                    "ENCOUNTER": cid,
                    "CODE": cond["code"],
                    "DESCRIPTION": cond["name"]
                }
                conditions.append(cond_record)
                patient_conditions.append(cond["name"])

                # Add corresponding medication
                for med in MEDICATIONS:
                    if med["condition"] == cond["name"] and random.random() < 0.75:
                        medications.append({
                            "START": str(onset_date),
                            "STOP": "",
                            "PATIENT": pid,
                            "PAYER": "Medicare" if age >= 65 else "Commercial",
                            "ENCOUNTER": cid,
                            "CODE": med["code"],
                            "DESCRIPTION": med["name"],
                            "TOTALCOST": round(random.uniform(45.0, 320.0), 2)
                        })

        # Number of encounters over the study period
        # Baseline ambulatory visits
        n_ambulatory = int(np.random.poisson(lam=max(1, int(len(patient_conditions) * 1.8 + age / 25))))
        for _ in range(n_ambulatory):
            enc_date = study_start + datetime.timedelta(days=random.randint(0, total_days - 60))
            eid = str(uuid.uuid4())
            encounters.append({
                "Id": eid,
                "START": f"{enc_date}T09:30:00Z",
                "STOP": f"{enc_date}T10:15:00Z",
                "PATIENT": pid,
                "ORGANIZATION": "MGH_Ambulatory_Clinic",
                "PROVIDER": "Dr_Provider",
                "PAYER": "Medicare" if age >= 65 else "Commercial",
                "ENCOUNTERCLASS": "ambulatory",
                "CODE": "185349003",
                "DESCRIPTION": "Encounter for checkup / routine evaluation",
                "BASE_ENCOUNTER_COST": 150.0,
                "TOTAL_CLAIM_COST": 185.0,
                "REASONCODE": "185349003",
                "REASONDESCRIPTION": "Routine medical visit"
            })

        # Inpatient Admissions (The core index admissions for 30-day readmission prediction)
        # Inpatient probability is higher for elderly, multimorbid, and underserved minority groups
        comorbidity_score = len(patient_conditions)
        base_inpatient_prob = 0.04 + (0.04 * comorbidity_score) + (0.05 if age >= 65 else 0.0)
        
        # Add emergency/inpatient disparity factor
        if race in ["black", "hispanic"]:
            base_inpatient_prob *= 1.25

        if random.random() < min(0.85, base_inpatient_prob):
            num_admissions = 1 if random.random() < 0.70 else 2
            
            for adm_idx in range(num_admissions):
                adm_day_offset = random.randint(60, total_days - 120)
                adm_date = study_start + datetime.timedelta(days=adm_day_offset)
                
                # Length of stay (1 to 14 days)
                los_days = int(np.random.gamma(shape=2.5, scale=1.8)) + 1
                los_days = min(los_days, 18)
                disch_date = adm_date + datetime.timedelta(days=los_days)

                primary_reason = random.choice(patient_conditions) if patient_conditions else "Acute respiratory distress"
                eid = str(uuid.uuid4())
                cost = float(los_days * 1850.0 + random.uniform(800, 3500))

                encounters.append({
                    "Id": eid,
                    "START": f"{adm_date}T14:00:00Z",
                    "STOP": f"{disch_date}T11:00:00Z",
                    "PATIENT": pid,
                    "ORGANIZATION": "Mass_General_Hospital",
                    "PROVIDER": "Dr_Inpatient_Attending",
                    "PAYER": "Medicare" if age >= 65 else "Commercial",
                    "ENCOUNTERCLASS": "inpatient",
                    "CODE": "32485007",
                    "DESCRIPTION": "Hospital admission, inpatient stay",
                    "BASE_ENCOUNTER_COST": round(cost * 0.85, 2),
                    "TOTAL_CLAIM_COST": round(cost, 2),
                    "REASONCODE": "84114007" if "Heart failure" in primary_reason else "44054006",
                    "REASONDESCRIPTION": primary_reason
                })

                # Inpatient diagnostic procedure
                if random.random() < 0.80:
                    proc = random.choice(PROCEDURES)
                    procedures.append({
                        "DATE": str(adm_date),
                        "PATIENT": pid,
                        "ENCOUNTER": eid,
                        "CODE": proc["code"],
                        "DESCRIPTION": proc["name"],
                        "BASE_COST": proc["base_cost"]
                    })

                # 30-Day Readmission Occurrence Simulation:
                # Disparity mechanisms:
                # - Geriatric patients: High frailty & polymedication -> higher readmission
                # - Minority patients: Structural gaps (lack of post-discharge PCP follow-up, social determinants) -> higher readmission
                # - High comorbidity count: Steep increase in readmission risk
                readmission_risk = 0.08 + (0.035 * comorbidity_score) + (0.015 * (los_days / 5.0))
                if age >= 65:
                    readmission_risk += 0.09
                if race in ["black", "hispanic"]:
                    readmission_risk += 0.07  # Reflects unaddressed social determinants in raw EHR
                if "Heart failure" in patient_conditions or "Chronic kidney disease stage 3" in patient_conditions:
                    readmission_risk += 0.08
                
                readmission_risk = min(0.65, max(0.04, readmission_risk))

                # If readmission occurs within 30 days:
                if random.random() < readmission_risk:
                    readm_gap = random.randint(3, 28)
                    readm_start = disch_date + datetime.timedelta(days=readm_gap)
                    readm_los = random.randint(2, 8)
                    readm_stop = readm_start + datetime.timedelta(days=readm_los)
                    readm_eid = str(uuid.uuid4())
                    
                    readm_class = "inpatient" if random.random() < 0.75 else "emergency"
                    encounters.append({
                        "Id": readm_eid,
                        "START": f"{readm_start}T16:00:00Z",
                        "STOP": f"{readm_stop}T12:00:00Z",
                        "PATIENT": pid,
                        "ORGANIZATION": "Mass_General_Hospital",
                        "PROVIDER": "Dr_Emergency_Attending",
                        "PAYER": "Medicare" if age >= 65 else "Commercial",
                        "ENCOUNTERCLASS": readm_class,
                        "CODE": "50849002" if readm_class == "emergency" else "32485007",
                        "DESCRIPTION": f"Acute 30-day readmission ({readm_class})",
                        "BASE_ENCOUNTER_COST": 2800.0,
                        "TOTAL_CLAIM_COST": 3400.0,
                        "REASONCODE": "84114007",
                        "REASONDESCRIPTION": f"Readmission secondary to {primary_reason}"
                    })

    # Save to CSV
    df_patients_clean = df_patients.drop(columns=["age", "age_tier"])
    df_encounters = pd.DataFrame(encounters)
    df_conditions = pd.DataFrame(conditions)
    df_medications = pd.DataFrame(medications)
    df_procedures = pd.DataFrame(procedures)

    print(f"[*] Writing Synthea tables to {output_dir}...")
    df_patients_clean.to_csv(os.path.join(output_dir, "patients.csv"), index=False)
    df_encounters.to_csv(os.path.join(output_dir, "encounters.csv"), index=False)
    df_conditions.to_csv(os.path.join(output_dir, "conditions.csv"), index=False)
    df_medications.to_csv(os.path.join(output_dir, "medications.csv"), index=False)
    df_procedures.to_csv(os.path.join(output_dir, "procedures.csv"), index=False)

    print(f"[+] Generation complete:")
    print(f"    - Patients:    {len(df_patients_clean):,}")
    print(f"    - Encounters:  {len(df_encounters):,}")
    print(f"    - Conditions:  {len(df_conditions):,}")
    print(f"    - Medications: {len(df_medications):,}")
    print(f"    - Procedures:  {len(df_procedures):,}")
    return df_patients_clean, df_encounters


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Synthea Synthetic EHR Data")
    parser.add_argument("--n_patients", type=int, default=30000, help="Number of patients to generate")
    parser.add_argument("--output_dir", type=str, default="data/raw", help="Output directory")
    args = parser.parse_args()

    generate_synthea_cohort(n_patients=args.n_patients, output_dir=args.output_dir)
