# 3-Minute Demo Video Script: Subgroup Performance in Healthcare Risk Models

**Course:** Big Data Systems & Responsible AI Engineering (BDSRAE) - PES University  
**Topic 19:** Subgroup Performance in Healthcare Risk Models  
**Faculty Evaluator:** Dr. Manjula Devananda  
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
  3. Visual of `reports/figures/big_data_scaling.png` showing the $30.5\times$ compression and linear throughput scaling.
- **Voiceover Script:**
  > *"Hello everyone. I am Ankith Umadi, presenting alongside Haneesh and Vikas for the Big Data Systems and Responsible AI Engineering course under Dr. Manjula Devananda.*  
  > *Our project addresses a critical challenge in clinical AI: **Subgroup Performance and Calibration Disparities in Healthcare Risk Models**.*  
  > *Hospitals deploy machine learning to predict 30-day readmissions and triage post-discharge care. However, standard models optimizing global loss often hide lethal failures in minority and vulnerable subgroups.*  
  > *To meet Requirement 1's Big Data mandate with empirical evidence, we built an Apache PySpark distributed pipeline. Our ETL ingests multi-table Synthea electronic health records, runs longitudinal window queries to identify inpatient index hospitalizations, extracts chronic comorbidities, and writes a partitioned Parquet Lakehouse.*  
  > *As shown in our benchmark, columnar partitioned storage achieved a **30.5-fold storage footprint reduction** over raw CSV, scaling effortlessly to over **5,600 records per second**."*

---

### Part 2: The Responsible AI Audit & Clinical Failure Modes (0:50 – 1:45)
- **Time:** 0:50 – 1:45 (55 seconds)
- **Speaker:** Haneesh E
- **On Screen:**
  1. Results notebook scrolling through the subgroup evaluation table.
  2. Focus on `reports/figures/subgroup_disparities.png` (Sensitivity bar chart highlighting Pediatric and Native American collapse).
  3. Focus on `reports/figures/intersectional_ece_heatmap.png` showing dark red miscalibration zones.
- **Voiceover Script:**
  > *"I am Haneesh. When we train our baseline LightGBM model, it achieves a seemingly exceptional global ROC-AUC of 0.9010. But our granular subgroup audit reveals why aggregate metrics are dangerous in healthcare.*  
  > *First, in Pediatric patients under 18, clinical sensitivity collapses to just **21.4%**, meaning nearly **79% of readmissions are missed**.*  
  > *In Native American patients, sensitivity drops to **zero percent**—every single readmitted patient is missed by the unweighted baseline.*  
  > *Second, we evaluated probabilistic calibration using Expected Calibration Error (ECE). In clinical medicine, risk thresholds dictate care tiers. While majority White patients have a low ECE of 0.028, Asian patients experience an ECE of **0.1107**—their readmission risk is over-estimated by almost double, wasting hospital triage beds.*  
  > *Finally, our intersectional audit revealed compounded failure in **Geriatric Black Males**, who suffer both an alarming ECE of 0.1796 and a 57% miss rate."*

---

### Part 3: Bias Mitigation, Pareto Trade-Off & Model Card Governance (1:45 – 3:00)
- **Time:** 1:45 – 3:00 (75 seconds)
- **Speaker:** Vikas R
- **On Screen:**
  1. Terminal showing mitigation running in `run_pipeline.py`.
  2. `reports/figures/calibration_comparison.png` displaying straightened reliability curves.
  3. `reports/figures/fairness_utility_tradeoff.png` showcasing the Pareto frontier.
  4. Quick walkthrough of `reports/model_card.md` highlighting the Intended-Use and Residual Risks warnings.
- **Voiceover Script:**
  > *"I am Vikas. To resolve these failures, we implemented three Responsible AI mitigation strategies:*  
  > *Pre-processing demographic reweighing, post-processing subgroup-specific isotonic calibration, and Fairlearn threshold optimization under Equalized Odds constraints.*  
  > *Here in our reliability comparison, subgroup calibration successfully straightens the curves, restoring probability fidelity for minority populations.*  
  > *Even more strikingly, our fairness-utility Pareto analysis demonstrates that Threshold Optimization slashes the Equal Opportunity Disparity from **0.7500 down to 0.3333**—a **55.6% reduction in clinical disparity**—while maintaining an identical **0.9010 ROC-AUC** and sacrificing only 2.75% in overall accuracy.*  
  > *We documented all operational boundaries in a comprehensive Model Card, including mandatory clinical safety warnings: automated scoring must be disabled for pediatric cohorts, and thresholds must adapt to ensure equal opportunity in post-discharge monitoring.*  
  > *Our reproducible repository includes complete Docker configs, PySpark scripts, and pre-executed results notebooks. Thank you!"*

---

## Recording Checklist for the Team:
1. **Resolution:** 1080p (1920x1080) at 30 or 60 fps.
2. **Audio:** Clear microphone, zero background noise.
3. **Screen Layout:** Left side terminal/VS Code, right side high-res figures and Jupyter notebook.
4. **Time Enforcement:** Exactly 3 minutes.
