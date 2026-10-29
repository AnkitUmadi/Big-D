# 8 to 10-Minute Capstone Presentation Deck Outline: Subgroup Performance in Healthcare Risk Models

**Course:** Big Data Systems & Responsible AI Engineering (BDSRAE)  
**Department:** Computer Science & Engineering (AI & ML), PES University  
**Topic 19:** Subgroup Performance in Healthcare Risk Models  
**Faculty Evaluator:** Dr. Manjula Devananda, Professor, Dept. of CSE (AI & ML)  
**Team Members:**  
- **Ankith Umadi** (PES1UG24AM422)  
- **Haneesh E** (PES1UG24AM471)  
- **Vikas R** (PES1UG24AM477)  

**Presentation Duration:** 8 to 10 Minutes (Target: 9 Minutes + 2 Mins Q&A)  
**Template Standard:** PES University Capstone Defense Template  

---

## Slide 1: Title & Team Credentials (0:00 – 0:45)
- **Slide Content:**
  - Project Title: *Subgroup Performance, Calibration Disparities, and Scalable Fairness Mitigation in Healthcare Risk Models*
  - Subtitle: *An End-to-End Big Data Systems and Responsible AI Framework*
  - PES University Crest & CSE (AI & ML) Department Banner
  - Team Details: Ankith Umadi (PES1UG24AM422), Haneesh E (PES1UG24AM471), Vikas R (PES1UG24AM477)
  - Faculty Guide: Dr. Manjula Devananda, Professor, Dept. of CSE (AI & ML)
- **Speaker:** Ankith Umadi
- **Key Talking Points:**
  - Introduce team, SRNs, and project topic.
  - State central thesis: Aggregate metrics in clinical machine learning hide life-threatening subgroup disparities; combining Big Data distributed engineering with Responsible AI auditing is necessary for equitable clinical deployment.

---

## Slide 2: Clinical Problem Statement & Case Study (0:45 – 1:30)
- **Slide Content:**
  - Case Scenario: Hospital 30-Day Readmission Risk Scoring (CMS HRRP context).
  - Clinical Decision-Support Pipeline: Inpatient discharge -> Risk score -> Triage care (phone calls, home nurse visits, medication reconciliation).
  - The Core Dilemma: False Negatives ($\text{FN}$) kill—patients discharged unmonitored experience acute decompensation. Miscalibration distorts triage tier thresholds.
- **Speaker:** Ankith Umadi
- **Key Talking Points:**
  - 30-day readmissions are the primary quality and reimbursement metric.
  - Asymmetric error costs: false alarms cost a modest phone check-in; missed readmissions cost tens of thousands in emergency care and compromise patient survival.

---

## Slide 3: Big Data Systems Architecture & Parquet Lakehouse (1:30 – 2:30)
- **Slide Content:**
  - Architecture Diagram: Synthea-style Relational CSVs -> PySpark Distributed ETL -> Partitioned Parquet Lakehouse (`[age_group, race]`) -> Analytical Cohort.
  - Technical Implementation:
    - Distributed Window Analytics: Self-join and chronological filtering for leak-free readmission target construction.
    - Censor Handling: Identifying encounters within 30 days of study horizon without readmission.
    - Leak-Free Feature Aggregation: Strict cutoff at index discharge.
- **Speaker:** Ankith Umadi
- **Key Talking Points:**
  - Fulfills Big Data systems mandate with industrial-grade PySpark distributed execution.
  - Partitioned Lakehouse design (`[age_group, race]`) enables predicate pushdown and rapid subgroup partition pruning.

---

## Slide 4: Measured Big Data Scalability Evidence (2:30 – 3:30)
- **Slide Content:**
  - Benchmark Table & Plots (`reports/figures/big_data_scaling.png`).
  - Empirical Evidence Metrics (PySpark 4.2.1, 16 CPU Cores, 23.7 GB RAM):
    - Storage Compression: 13.1 MB CSV -> 0.24 MB Parquet (**53.7x compression ratio**).
    - Throughput: Scaling to **1,047.3 patients/second** on local multi-threading.
    - Reproducible, isolated benchmarks with automatic directory cleanup.
- **Speaker:** Ankith Umadi
- **Key Talking Points:**
  - Demonstrates empirical scalability benchmarking satisfying course requirements.
  - Highlights how Snappy columnar compression dramatically reduces storage footprint.

---

## Slide 5: Baseline Healthcare Risk Models & The Accuracy Paradox (3:30 – 4:15)
- **Slide Content:**
  - Models Evaluated on Patient-Independent Splits (Train 70%, Val 15%, Test 15%):
    - LightGBM, Logistic Regression, Random Forest.
  - Overall Test Performance Table (N = 550 Encounters across 343 Patients):
    - LightGBM: ROC-AUC = 0.6710 | PR-AUC = 0.3057 | Accuracy = 80.73% | Brier Score = 0.1437.
    - Logistic Regression: ROC-AUC = 0.6824 | Accuracy = 81.09%.
    - Random Forest: ROC-AUC = 0.6835 | Accuracy = 80.73%.
- **Speaker:** Haneesh E
- **Key Talking Points:**
  - The "Accuracy Paradox": models achieve ~81% accuracy, but near-zero sensitivity under default 0.5 thresholds on imbalanced readmission data.

---

## Slide 6: Responsible AI Subgroup Performance Disparity Audit (4:15 – 5:30)
- **Slide Content:**
  - Subgroup Disparity Table & Bar Chart (`reports/figures/subgroup_disparities.png`).
  - Key Findings Highlighted:
    - Black ($N=110$), Hispanic ($N=78$), and Asian ($N=25$): Sensitivity collapses to **0.0%** (100% false negatives under default cutoff).
    - Pediatric (<18): Discrimination collapses to **ROC-AUC = 0.4530** (worse than chance).
    - Geriatric (65+): High base rate (25.96%), sensitivity only 3.70%.
- **Speaker:** Haneesh E
- **Key Talking Points:**
  - Default decision thresholds completely abandon minority high-risk patients.
  - Proves that adult clinical models cannot be generalized to pediatric populations.

---

## Slide 7: Subgroup Calibration Disparities (ECE) & Intersectional Analysis (5:30 – 6:30)
- **Slide Content:**
  - Subgroup Expected Calibration Error Table.
  - Intersectional ECE Heatmap (`reports/figures/intersectional_ece_heatmap.png`).
  - Key Intersectional Failures:
    - Geriatric Hispanic Females: Base Rate 28.57%, **$\text{ECE} = 0.2514$**, 100% miss rate.
    - Pediatric Black Females: Base Rate 27.27%, **$\text{ECE} = 0.2312$**, 100% miss rate.
    - Geriatric Black Males: Base Rate 38.10%, **$\text{ECE} = 0.1908$**, 100% miss rate.
- **Speaker:** Haneesh E
- **Key Talking Points:**
  - Explain why probability calibration is the cornerstone of clinical triage.
  - Single-axis audits conceal compounding harms that explode at demographic intersections.

---

## Slide 8: Algorithmic Bias Mitigation Strategies (6:30 – 7:30)
- **Slide Content:**
  - Three Mitigation Architectures:
    1. Pre-Processing: Demographic Sample Reweighing (fitted on training data).
    2. Post-Processing: Subgroup Isotonic Calibration (fitted on validation data).
    3. Post-Processing: Fairlearn ThresholdOptimizer for Equal Opportunity (fitted on validation data).
  - Methodological rigor: zero test-set leakage during tuning; fallback for sparse single-class groups.
- **Speaker:** Vikas R
- **Key Talking Points:**
  - Explain the difference between probability calibration and threshold shifting.
  - Emphasize that all mitigators were tuned on an independent validation set and evaluated on frozen test data.

---

## Slide 9: Mitigation Trade-Off Evaluation & Pareto Frontier (7:30 – 8:30)
- **Slide Content:**
  - Mitigation Trade-Off Summary Table & Pareto Plot (`reports/figures/fairness_utility_tradeoff.png`).
  - Comparative Results on Held-Out Test Set:
    - Subgroup Isotonic Calibration: Reduces mean subgroup ECE from **0.1207 to 0.1004** (-16.8%) and cuts Equal Opportunity Disparity from **1.0000 to 0.1250** (-87.5% reduction) with virtually zero loss in discrimination (AUC 0.6627 vs 0.6710).
    - Fairlearn Threshold Optimization: Shifting decision cutoffs boosts F1-score from **0.0536 to 0.3762**, increasing high-risk recall at the expense of accuracy (40.91%).
- **Speaker:** Vikas R
- **Key Talking Points:**
  - Subgroup calibration delivers the most balanced Pareto improvement for clinical decision support.
  - Transparent trade-off reporting: we acknowledge that aggressive threshold post-processing increases false alarms.

---

## Slide 10: Model Governance, Safety Warnings & Conclusion (8:30 – 9:30)
- **Slide Content:**
  - Model Card (Mitchell et al.) & Data Card (Gebru et al.) deliverables.
  - Mandatory Safety Warnings:
    - Automated scoring strictly prohibited for pediatric patients (<18).
    - Decision support only; never to be used for denying care or insurance claims.
  - Summary of Deliverables: Docker setup, unit test suite (10/10 passed), reproducibility commands.
- **Speaker:** Vikas R
- **Key Talking Points:**
  - Responsible AI engineering requires governance and intended-use boundaries, not just mathematical algorithms.
  - Conclude with project impact and invite faculty questions.
