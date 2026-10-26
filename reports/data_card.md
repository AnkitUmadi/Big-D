# Data Card: Synthea-Generated Longitudinal EHR Readmission Cohort

Following the Gebru et al. (2021) *Datasheets for Datasets* standard.

---

## 1. Motivation
- **Purpose:** Created to evaluate subgroup performance disparities, calibration errors, and algorithmic fairness mitigation strategies in hospital 30-day readmission risk models without exposing real Protected Health Information (PHI).
- **Course Context:** Developed for the Big Data Systems & Responsible AI Engineering (BDSRAE) project track at PES University (Dept. of CSE AI & ML), under the guidance of Dr. Manjula Devananda.
- **Creator / Maintainer:** Ankith Umadi, Haneesh E, Vikas R.

---

## 2. Dataset Composition
- **Total Patients Generated:** 30,000 synthetic individuals (scalable to 100,000+).
- **Total Relational Tables:**
  1. `patients.csv`: Core demographics (Age, Gender, Race, Ethnicity, City, Zip, Healthcare Coverage & Expenses).
  2. `encounters.csv`: Chronological healthcare encounters (Inpatient admissions, Emergency room visits, Ambulatory visits, Costs, Length of Stay).
  3. `conditions.csv`: Chronic diagnoses mapped to SNOMED-CT (Hypertension, Type 2 Diabetes, Heart Failure, COPD, CKD, CAD, Depression).
  4. `medications.csv`: Active outpatient and inpatient prescriptions mapped to RxNorm.
  5. `procedures.csv`: Diagnostic and therapeutic procedures (Echocardiogram, Chest X-Ray, PCI, ECG).
- **Target Variable:** `readmission_30d` (Binary: 1 if an acute inpatient or emergency admission occurred within 30 days of inpatient index discharge; 0 otherwise).
- **Overall Inpatient Readmission Prevalence:** ~21.6%.

### Demographic Stratification
| Demographic Attribute | Categories & Proportion |
| :--- | :--- |
| **Race** | White (63%), Black/African American (15%), Hispanic/Latino (13%), Asian (6%), Native American (1.5%), Other (1.5%) |
| **Age Group** | Pediatric <18 (10%), Young Adult 18–44 (30%), Middle Age 45–64 (35%), Geriatric 65+ (25%) |
| **Gender** | Female (51%), Male (49%) |
| **Geography** | Massachusetts urban, suburban, and rural healthcare regions (Boston, Worcester, Springfield, Cambridge, Lowell) |

---

## 3. Collection & Generation Process
- **Mechanism:** Programmatic generation using realistic epidemiological hazard functions and clinical state-transition models based on open-source Synthea specifications.
- **Disparity Encoding:** To realistically mirror real-world healthcare disparities documented in clinical AI literature (e.g., Obermeyer et al., Science 2019):
  1. Minority populations reflect historical barriers to preventive care, resulting in higher emergency utilization.
  2. Geriatric populations reflect true clinical multimorbidity distributions and polypharmacy interactions.
  3. Small demographic groups (Asian, Native American) reflect representation sparsity commonly observed in regional hospital electronic records.
- **Data Privacy & Licensing:** 100% synthetic data generated from probabilistic distributions. Contains zero real personal data or HIPAA-protected health identifiers. Freely reusable under Apache 2.0 / MIT licenses.

---

## 4. Pre-processing & Storage Design
- **Engine:** Apache PySpark 3.5.1 distributed processing.
- **Storage Format:** Partitioned Parquet Lakehouse storage partitioned by `[age_group, race]`.
- **Compression Efficiency:** Columnar Snappy Parquet achieves a **30.5x storage footprint reduction** over raw uncompressed CSV (43.9 MB CSV -> 1.4 MB Parquet), enabling rapid partition pruning and predicate pushdown during subgroup queries.
