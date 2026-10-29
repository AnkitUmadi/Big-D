# 3-Minute Demo Video Script: Subgroup Performance in Healthcare Risk Models

**Course:** Big Data Systems & Responsible AI Engineering (BDSRAE) - PES University  
**Topic 19:** Subgroup Performance in Healthcare Risk Models  
**Faculty Evaluator:** Dr. Manjula Devananda, Professor, Dept. of CSE (AI & ML)  
**Team:**  
- **Speaker 1:** Ankith Umadi (PES1UG24AM422)  
- **Speaker 2:** Haneesh E (PES1UG24AM471)  
- **Speaker 3:** Vikas R (PES1UG24AM477)  

**Total Duration:** Exactly 3 Minutes (180 Seconds)  
**Deliverable Requirement:** Option A - A 3-minute demo video  

---

## Storyboard & Timing Breakdown

### Part 1: Problem Formulation & Big Data Systems (0:00 – 0:50)
- **Time:** 0:00 – 0:50 (50 seconds)
- **Speaker:** Ankith Umadi
- **On Screen:** 
  1. Title slide showing PES University department logo, project topic, and team SRNs.
  2. Terminal showing `run_pipeline.py` executing PySpark distributed ETL.
  3. Visual of `reports/figures/big_data_scaling.png` showing the 53.7x compression and linear throughput scaling.
- **Voiceover Script:**
  > *"Hello everyone. I am Ankith Umadi, presenting alongside Haneesh and Vikas for the Big Data Systems and Responsible AI Engineering course under Dr. Manjula Devananda.*  
  > *Our project addresses a critical challenge in clinical AI: **Subgroup Performance and Calibration Disparities in Healthcare Risk Models**.*  
  > *Hospitals deploy machine learning to predict 30-day readmissions and triage post-discharge transitional care. However, standard models optimizing global accuracy often hide lethal failures in minority and vulnerable subgroups.*  
  > *To satisfy the Big Data mandate with empirical evidence, we built an Apache PySpark distributed pipeline. Our ETL ingests multi-table Synthea-style EHR tables, eliminates temporal data leakage, handles overlapping encounters and boundary censoring, and writes a partitioned Parquet Lakehouse.*  
  > *As shown in our empirical benchmark, columnar partitioned storage achieved a **53.7-fold storage footprint reduction** over raw CSV, scaling to over **1,000 patients per second** on local multi-threading."*

---

### Part 2: The Responsible AI Audit & Clinical Failure Modes (0:50 – 1:45)
- **Time:** 0:50 – 1:45 (55 seconds)
- **Speaker:** Haneesh E
- **On Screen:**
  1. Results notebook scrolling through the patient-split subgroup evaluation table.
  2. Focus on `reports/figures/subgroup_disparities.png` (Sensitivity bar chart highlighting minority collapse).
  3. Focus on `reports/figures/intersectional_ece_heatmap.png` showing dark red miscalibration zones.
- **Voiceover Script:**
  > *"I am Haneesh. When we train our baseline LightGBM model on patient-independent splits, it achieves a seemingly strong 80.7% accuracy. But our granular subgroup audit reveals why accuracy alone is dangerous in healthcare.*  
  > *First, under default decision thresholds, clinical sensitivity collapses to **0.0% across Black, Hispanic, and Asian minority cohorts**—meaning **100% of readmitted minority patients are missed** by the unadjusted baseline.*  
  > *Second, in pediatric patients under 18, discrimination collapses to an ROC-AUC of **0.4530**—worse than random guessing—proving that general adult readmission models cannot be safely applied to pediatric admissions.*  
  > *Third, we evaluated probabilistic calibration using Expected Calibration Error (ECE). In clinical medicine, risk probabilities dictate care tiers. While majority White patients have an ECE of 0.0375, intersectional cohorts face severe miscalibration—reaching **0.2514 in Geriatric Hispanic Females**."*

---

### Part 3: Bias Mitigation, Pareto Trade-Off & Model Card Governance (1:45 – 3:00)
- **Time:** 1:45 – 3:00 (75 seconds)
- **Speaker:** Vikas R
- **On Screen:**
  1. Terminal showing mitigation running in `run_pipeline.py`.
  2. `reports/figures/calibration_comparison.png` displaying straightened reliability curves.
  3. `reports/figures/fairness_utility_tradeoff.png` showcasing the Pareto frontier.
  4. Walkthrough of `MODEL_CARD.md` highlighting the Intended-Use and Clinical Safety warnings.
- **Voiceover Script:**
  > *"I am Vikas. To resolve these failures, we evaluated three Responsible AI mitigation strategies:*  
  > *Pre-processing demographic reweighing, post-processing subgroup-specific isotonic calibration, and Fairlearn threshold optimization under Equal Opportunity constraints.*  
  > *Crucially, all mitigators were tuned on an independent validation set and tested on frozen test data.*  
  > *Our reliability comparison demonstrates that Subgroup Isotonic Calibration successfully straightens probability curves, reducing mean calibration error by **16.8%** and cutting Equal Opportunity Disparity from **1.0000 down to 0.1250**—an **87.5% disparity reduction**—with negligible impact on global AUC (0.6627 vs 0.6710).*  
  > *Meanwhile, Fairlearn threshold optimization shifts the trade-off, boosting the clinical F1-score from 0.0536 to 0.3762 by prioritizing high-risk minority recall.*  
  > *We codified all operational safeguards into formal Model and Data Cards: automated scoring must be disabled for pediatric cohorts, and model flags must serve purely as decision support. Thank you!"*

---

## Recording Checklist for the Team:
1. **Resolution:** 1080p (1920x1080) at 30 or 60 fps.
2. **Audio:** Clear microphone, zero background noise.
3. **Screen Layout:** Left side terminal/VS Code, right side high-res figures and Jupyter notebook.
4. **Time Enforcement:** Exactly 3 minutes (180 seconds).
