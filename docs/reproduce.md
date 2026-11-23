# Reproduction

## Choose a workflow

Use the [results index](../results/README.md) to inspect saved findings. For a rebuild, obtain the input artifacts listed below and compare their SHA-256 checksums to the [artifact registry](../reproducibility/artifacts.tsv). The registry distinguishes data obtainable from the distributors from study assets held by the author. Keep the original names and internal paths of sealed packages.

| Workflow | Entry point | Inputs |
| --- | --- | --- |
| Protocol A training | [Protocol A notebook](../notebooks/protocol_a/Protocol_A_755_Revised_Run.ipynb) | MSLD v2.0, `protocol-a-inputs` extracted as a directory, and the release code dataset. The preflight checks the fixed 755-image design. For a pause, attach the full saved output. |
| Protocol B training | [Single model and fold](../notebooks/protocol_b/00_GPU_Train_Fold.ipynb) | MSLD v2.0 and `protocol-b-training`. Set the model and fold before execution. For a pause, attach the full saved output and set `RESUME=True`. |
| Protocol B publication results | [CPU publication notebook](../notebooks/protocol_b/01_CPU_Final_Publication.ipynb) | `protocol-b-predictions` and the release code dataset. Confirm 20 completed model-fold records, 215 test groups, 138 calibration groups, and 189 correct ensemble decisions. |
| Leakage reports | [CPU merger](../notebooks/leakage/04_CPU_Final_Report.ipynb) | `leakage-inputs` and `leakage-batch-1` through `leakage-batch-3`. The sealed inputs include the two fixed seed 0–4 baseline series. |
| Metadata and operating-point analysis | [CPU metadata and thresholds](../notebooks/extensions/00_CPU_Metadata_and_Thresholds.ipynb) | `extension-inputs` and MSLD v2.0 for file-feature verification. |
| Final extension tables | [CPU consolidation](../notebooks/extensions/04_CPU_Consolidate.ipynb) | `extension-consolidation-inputs`. This input contains the complete saved CPU and GPU results and locked extension design. |
| Duplicate and calibration audit | [CPU audit notebook](../notebooks/audit/00_CPU_Audit_and_Comparisons.ipynb) | `supplemental-inputs` and MSLD v2.0. The results index records the completed visual review separately. |
| Qualitative ensemble Grad-CAM | [GPU Grad-CAM notebook](../notebooks/attribution/02_GPU_Ensemble_GradCAM.ipynb) | MSLD v2.0, `protocol-b-predictions`, all 40 selected checkpoint tensors, and the release code dataset. |
| Quantitative ensemble attribution | [GPU validation and recovery notebook](../notebooks/attribution/04_GPU_Quantitative_Validation.ipynb) | MSLD v2.0, `supplemental-inputs`, all 40 selected checkpoints, `xai-recovery-cache`, and the release code dataset. |
| Supplemental report | [CPU report notebook](../notebooks/supplementary/05_CPU_Final_Report.ipynb) | `supplemental-inputs`, `intermediate-dose-inputs`, the release code dataset, and the six archives `supplemental-training-batch-1` through `supplemental-training-batch-3`, `supplemental-results-cpu`, `supplemental-results-xai`, and `supplemental-results-dose`. |

Use the CPU accelerator setting for saved-prediction notebooks. GPU workflows need the recorded software and compute environment from the corresponding job records. Saved notebooks pin their analysis dependencies. The fixed designs and archive readers check manuscript-era source identities at startup.

## Source layout

The [code source index](../reproducibility/code_sources.csv) maps each frozen runtime filename to its byte-identical source here. Workflows that share the same source use a single tracked file. Some workflows require a different version of the same module. The exporter checks every byte and restores import-compatible names on demand.

```bash
python tools/materialize_workflow.py protocol_b_analysis ./runtime/protocol_b_analysis
```

Available source groups are `protocol_a`, `protocol_b_training`, `protocol_b_analysis`, `qualitative_gradcam`, `leakage`, `extensions`, `supplemental`, `dose`, `xai_recovery` and `supplemental_report`. The sealed input packages remain the execution records for workflows that load their own code. The audit scripts under `src/audit/` retain their original Kaggle dataset paths and should be configured with the relevant data locations.

For notebooks that request a release code dataset, use a private Kaggle input containing the `src/`, `tools/` and `reproducibility/` directories from this checkout, with their structure intact. Upload a ZIP of those three folders and confirm Kaggle has expanded it. The notebook locates `tools/materialize_workflow.py` and the adjoining source index. Code exports and study outputs belong in an external working directory or an ignored `runtime/` directory.

For inference and training use the checkpoint and data sources corresponding to the recorded manifests. The saved study packages are versioned as a unit.
