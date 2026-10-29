# Data Card: Synthea-Style Synthetic Longitudinal EHR Readmission Cohort

Following the Gebru et al. (2021) *Datasheets for Datasets* standard.

---

## 1. Motivation

- **Purpose:** Created to evaluate algorithmic fairness, demographic subgroup disparities, probability calibration, and mitigation strategies in hospital 30-day all-cause readmission risk models without exposing real Protected Health Information (PHI) or violating HIPAA.
- **Course Context:** Developed for Big Data Systems & Responsible AI Engineering (BDSRAE) at PES University (Dept. of CSE AI & ML), under the supervision of Dr. Manjula Devananda.
- **Authors & Maintainers:** Ankith Umadi (PES1UG24AM422), Haneesh E (PES1UG24AM471), Vikas R (PES1UG24AM477).

---

## 2. Data Provenance & Generation Methodology

> [!IMPORTANT]
> **Data Provenance Disclosure:**
> This dataset was programmatically generated using a local synthetic clinical generator (`data/synthea_generator.py`) implementing Synthea-style relational schemas and clinical hazard distributions. It is **not** an export from an official Synthea Massachusetts public download; rather, it is locally generated synthetic EHR data designed to emulate clinical workflows and documented real-world healthcare disparities (e.g., Obermeyer et al., *Science* 2019).

- **Generation Method:** Epidemiological hazard functions, Poisson utilization processes, and state transitions parameterized by age, chronic comorbidity profile, and socioeconomic indicators.
- **Reference Observation Period:** `2021-01-01 09:30:00 UTC` to `2024-11-01 15:45:00 UTC` (~3.83 years).
- **Synthetic Guarantee:** 100% synthetic data generated from mathematical distributions and pseudo-random generators with deterministic seed `SEED=42`. Contains zero real patient records, zero HIPAA identifiers, and zero real Protected Health Information.

---

## 3. Dataset Composition & Relational Schema

Recomputed actual counts from raw tables (`data/raw/`):

| Raw Table | File Size | Record Count | Description & Primary Keys |
| :--- | :--- | :--- | :--- |
| `patients.csv` | 2.58 MB | 15,000 | Unique patient demographics (`Id`, `BIRTHDATE`, `GENDER`, `RACE`, `ETHNICITY`, `HEALTHCARE_COVERAGE`, etc.) |
| `encounters.csv` | 23.58 MB | 87,382 | Longitudinal encounters (`Id`, `START`, `STOP`, `PATIENT`, `ENCOUNTERCLASS`, `TOTAL_CLAIM_COST`, etc.) |
| `conditions.csv` | 4.02 MB | 33,796 | Chronic clinical diagnoses mapped to SNOMED-CT codes |
| `medications.csv` | 4.29 MB | 29,967 | Active outpatient and inpatient prescriptions mapped to RxNorm codes |
| `procedures.csv` | 0.30 MB | 2,130 | Diagnostic and therapeutic procedures |

### Encounter Class Distribution
- **Ambulatory:** 83,535 encounters (95.60%)
- **Inpatient:** 3,632 encounters (4.16%) across 2,286 unique patients
- **Emergency:** 215 encounters (0.25%)
- **Total:** 87,382 encounters

### Demographic Breakdown (N = 15,000 Patients)
- **Sex / Gender:** Female: 7,746 (51.64%), Male: 7,254 (48.36%)
- **Race:** White: 9,460 (63.07%), Black: 2,253 (15.02%), Hispanic: 1,951 (13.01%), Asian: 873 (5.82%), Other: 250 (1.67%), Native American: 213 (1.42%)
- **Ethnicity:** Non-Hispanic: 12,389 (82.59%), Hispanic: 2,611 (17.41%)
- **Age Distribution:** Pediatric (<18): ~10%, Young Adult (18–44): ~30%, Middle Age (45–64): ~35%, Geriatric (65+): ~25%

---

## 4. Cohort Construction, Target Definition & Quality Audit

### Target Variable: `readmission_30d`
- **Index Encounter:** Acute inpatient hospital admission (`ENCOUNTERCLASS == 'inpatient'`).
- **Prediction Point:** Hospital discharge time (`STOP`).
- **Target Construction:** Next qualifying *inpatient* admission for the same patient that begins strictly after index discharge (`nxt.START > cur.STOP`) and occurs within 30 days (`0.0 < gap_days <= 30.0`).
- **Audit Fixes Applied:**
  1. *Intervening Visits:* Intervening ambulatory and emergency encounters do not mask subsequent inpatient readmissions.
  2. *Overlapping Encounters:* 9 overlapping inpatient episodes (where `START < PREV_STOP`) were audited and prevented from being treated as false post-discharge readmissions.
  3. *Boundary Censoring:* 2 index admissions occurring within 30 days of the dataset horizon (`2024-11-01`) without an observed readmission lacked sufficient follow-up and were excluded to prevent false-negative label noise.
- **Valid Analytical Cohort:** 3,630 index admissions across 2,286 unique patients.
- **Readmission Prevalence:** 667 readmissions (18.37%).

### Foreign-Key Consistency Audit
- `encounters.csv` $\rightarrow$ `patients.csv`: 0 unknown patient IDs (100% linkage).
- `conditions.csv` $\rightarrow$ `patients.csv`: 0 unknown patient IDs (100% linkage).
- `medications.csv` $\rightarrow$ `patients.csv`: 0 unknown patient IDs (100% linkage).
- `procedures.csv` $\rightarrow$ `patients.csv`: 0 unknown patient IDs (100% linkage).
- *Observation on Encounter Linkage:* While procedures match `encounters.Id` with 100% linkage, condition and medication tables in this synthetic generator assign unique condition/prescription episode IDs in the `ENCOUNTER` column. However, because both conditions and medications contain valid patient IDs and exact chronological `START` dates, features are joined on `patient_id` filtered strictly by `cond_start <= enc_stop` and `med_start <= enc_stop`, guaranteeing zero future leakage.

---

## 5. Storage Design & Big Data Systems Architecture

- **Ingestion & Transformation Engine:** Apache PySpark 4.2.1 running on local multi-threading (`master='local[*]'`).
- **Analytical Storage:** Partitioned Parquet Lakehouse stored in `data/processed/readmission_lakehouse.parquet/`.
- **Partitioning Strategy:** Multi-level directory partitioning by `[age_group, race]`. This enables demographic partition pruning and predicate pushdown during subgroup fairness audits, avoiding full-table scans.
- **Compression Efficiency:** Columnar Snappy Parquet achieves a **53.7x compression ratio** over uncompressed CSV tables (13.1 MB CSV $\rightarrow$ 0.24 MB Parquet at 6,000 scale).

---

## 6. Known Limitations

1. **Synthetic Nature:** Does not contain unmeasured clinical confounders (e.g., detailed nursing notes, vital signs telemetry, social determinants of health / SDOH) present in real clinical EHR registries.
2. **Geographical Specificity:** Simulates Massachusetts regional health networks; demographic distributions and disease hazards will drift if evaluated against rural or international patient cohorts.
3. **Small Sample Sparsity:** Native American and Other racial categories have small cohort sizes in the held-out test set ($N=7$ and $N=13$), yielding wide confidence intervals for subgroup metrics.
