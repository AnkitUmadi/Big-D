# 3-Minute Demo Video Script: Subgroup Performance in Healthcare Risk Models

**Course:** Big Data Systems & Responsible AI Engineering (BDSRAE) - PES University  
**Topic 19:** Subgroup Performance in Healthcare Risk Models  
**Faculty Evaluator:** Dr. Manjula Devananda, Professor, Dept. of CSE (AI & ML)  
**Total Video Duration:** 3 Minutes 28 Seconds (208.43 Seconds, 6,253 Frames @ 30 FPS)  
**Deliverable:** Synchronized AI Voiceover Matching Screen Recording Frame-by-Frame (No Individual Student Names)

---

## Exact Frame-by-Frame Storyboard & Voiceover Alignment

| Scene # | Timestamp | Visual On-Screen Element | Exact Audio Narration |
|---|---|---|---|
| **Scene 1** | `00:00.5 – 00:16.1` | **GitHub Repo (`AnkitUmadi/Big-D`) & Architecture Flowchart**<br>Project title, PES University course context, multi-table Synthea ingestion diagram. | *"Welcome to our demonstration of Subgroup Performance in Healthcare Risk Models under Dr. Manjula Devananda. On GitHub, we present an end-to-end clinical readmission pipeline combining Apache PySpark distributed ETL with rigorous subgroup auditing and algorithmic bias mitigation."* |
| **Scene 2** | `00:18.0 – 00:35.7` | **Scalability Table & Accuracy Paradox**<br>PySpark benchmark table (66x compression, 895 pts/sec) and Section 4 Clinical Safety statement. | *"Our scalability benchmarks show PySpark compressing raw clinical tables 66-fold into Parquet, processing 900 patients per second. We uncover the Accuracy Paradox, where models reporting 81 percent accuracy exhibit near-zero sensitivity in minority subgroups under default thresholds."* |
| **Scene 3** | `00:37.5 – 00:74.1` | **VS Code Terminal Execution**<br>`python run_pipeline.py --cohort_size 5000 --skip_benchmarks` executing live Stages 2 through 6. | *"Switching to VS Code, we execute our master pipeline, run pipeline dot py. Apache PySpark executes distributed ETL, ingesting 15,000 synthetic patients and 87,000 clinical encounters. It handles boundary censoring, extracts leak-free index admissions, and writes an analytical Parquet lakehouse. The pipeline performs patient-stratified splitting, trains baseline LightGBM models, evaluates subgroup fairness, and applies multi-stage bias mitigations, completing in real time."* |
| **Scene 4** | `00:82.5 – 01:47.0` | **Parquet Lakehouse Explorer**<br>Expanding `readmission_lakehouse.parquet` hierarchical folders (`age_group` and `race`), `_SUCCESS.crc`. | *"In the project explorer, we inspect our analytical Lakehouse under data processed. Apache PySpark partitions the cohort hierarchically by age group and race, containing sub-folders for Asian, Black, Hispanic, and White cohorts alongside Spark success metadata. This physical partition layout enables predicate pushdown and fast partition pruning for clinical querying."* |
| **Scene 5** | `01:50.2 – 01:56.9` | **Figure: `big_data_scaling.png`**<br>Three-panel scaling: ETL runtime, storage footprint comparison, patient throughput. | *"In big data scaling, benchmarks demonstrate sub-linear scaling, 30-fold storage compression, and high throughput."* |
| **Scene 6** | `01:57.6 – 02:01.3` | **Figure: `subgroup_disparities.png`**<br>Clinical sensitivity collapse across Black, Hispanic, Asian, and Pediatric groups. | *"In subgroup disparities, sensitivity drops to zero for minorities."* |
| **Scene 7** | `02:02.0 – 02:06.2` | **Figure: `intersectional_ece_heatmap.png`**<br>2D matrix showing Geriatric $\times$ Minority Expected Calibration Error. | *"In the intersectional heatmap, older minority patients face severe calibration error."* |
| **Scene 8** | `02:08.0 – 02:13.0` | **Figure: `calibration_comparison.png`**<br>Reliability diagrams comparing unmitigated baseline vs. subgroup isotonic calibration. | *"In calibration comparison, isotonic calibration straightens reliability curves across demographics."* |
| **Scene 9** | `02:13.5 – 02:17.1` | **Figure: `fairness_utility_tradeoff.png`**<br>Pareto frontier showing ThresholdOptimizer shifting into Low Disparity Region. | *"In fairness tradeoff, threshold optimization slashes disparity."* |
| **Scene 10** | `02:18.2 – 02:36.3` | **Data Card: `data_card.md`**<br>Gebru et al. standard, 30,000 patients, relational tables, HIPAA privacy assurance. | *"Next, our Data Card follows the Gebru et al. standard. It documents our 30,000 synthetic patient cohort, relational encounters, SNOMED diagnoses, and PySpark Lakehouse storage design, ensuring reproducible benchmarking while guaranteeing zero exposure of real personal health data."* |
| **Scene 11** | `02:39.0 – 02:53.5` | **Model Card: `model_card.md`**<br>Mitchell et al. standard, FairBoost-v1.0 details, Section 6 documented pediatric warnings. | *"In our Model Card, structured per Mitchell et al., we document model performance, intended clinical decision boundaries, and explicit safety warnings, including mandated human oversight and safeguards prohibiting unmonitored pediatric scoring."* |
| **Scene 12** | `02:54.8 – 03:24.2` | **Technical Report: `technical_report.md`**<br>Disaggregated results table, Section 6 ethical safeguards, Section 7 conclusion & references. | *"Finally, our formal Technical Report synthesizes the entire engineering journey. We document the distributed Big Data architecture, empirical discovery of the Accuracy Paradox across racial and age strata, and our multi-stage bias mitigations. Section 6 defines mandatory clinical safeguards, strictly prohibiting automated standalone triage for pediatric patients. All pipeline code, lakehouse scripts, and reports are fully reproducible on GitHub. Thank you for watching."* |

---

## Audio File Locations

- Primary Workspace: `C:\Users\Ankit Umadi\Documents\bigdproject\Big-D\presentation\voiceover_synced.wav`
- Backup Workspace: `C:\Users\Ankit Umadi\Documents\big\presentation\voiceover_synced.wav`
- Generator Script: `C:\Users\Ankit Umadi\.gemini\antigravity\scratch\generate_perfect_synced_audio_v4.py`
