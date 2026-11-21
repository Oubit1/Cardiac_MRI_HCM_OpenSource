# Revision experiments

This release contains the computation and reporting scripts used during the November 2026 revision, with portable GPU, R-interpreter and encoder-configuration defaults. `source_manifest.json` records source and released SHA-256 hashes. The scientific algorithms are retained. ROI-Frozen Word-table insertion was removed from the public compute entry point, and manuscript-specific Table 2 filling is optional.

## Results and analysis correspondence

| Analysis | Development / external | Implementation and outputs |
| --- | --- | --- |
| Updated common-fold benchmark | 1059 / 309 | `2026.11.06/benchmark_1059/code/`; full-cine Frozen, FT, ROI-FT, ROI-Frozen and CMR/Rscore combinations |
| Nested tabular classifier selection | 1059 / 309 | `2026.11.06/nested_validation/code/tabular_suite.py` and `nested_tabular.R`; full and non-shape radiomics |
| Historical matched benchmark | 1050 / 309 | `nested_validation/code/run_validation.py`, `prepare.py`, `prepare_frozen.py`; retained source dependencies and historical analysis |
| Calibration and recalibration | See result table model/cohort columns | `recalibration/code/`, `code/revision_statistics.py`; joint logistic calibration intercept/slope, bootstrap CIs, and development-fitted recalibration |
| Reviewer statistics | Branch-specific cohort | `reviewer_statistics/code/`; NRI, specificity, calibration summaries and report generation |
| Acquisition heterogeneity | Available metadata per subgroup | `acquisition_followup/code/`; metadata extraction, subgroup analysis, Rscore regression and slice-gap analysis |

The released `tabular_suite.py` executes one outer five-fold run with inner five-fold classifier selection. CMR selection and Rscore reconstruction run inside the relevant training partitions. Each supervised image fold selects epochs using an internal 10% validation split, refits the encoder on the outer training patients, then extracts representations for training and held-out patients. Fusion uses fixed elastic-net logistic parameters. Its training representations are extracted from the outer-training encoder. Frozen representations use the unchanged pretrained encoder. Consult the protocol and implementation for the distinction between the standalone MIL probability head and feature-based downstream models.

`summary_results/benchmark_1059/results/nine_model_table2_metrics.csv` and `roi_fusion_table2_metrics.csv` correspond to the updated Table 2 calculations. `summary_results/benchmark_1059/roi_frozen/performance.csv` adds the ROI-Frozen comparisons. `performance_1059.csv` includes standalone image-head results. Use the appropriate prediction head when comparing tables. Recalibration and reviewer-statistics tables retain their original analysis scope.

`figures/` contains the updated nine-model ROC and five-fold full-cine/ROI training curves in PNG and PDF. Individual-patient predictions, images, masks and training checkpoints are supplied locally by the researcher.

## Installation

From the repository root, with a Python 3.10+ environment and a CUDA-compatible PyTorch installation:

```bash
pip install -r revision/requirements.txt
Rscript -e 'install.packages(c("glmnet", "jsonlite", "caret"))'
python revision/run.py init
```

Optional environment variables are `JMRI_RSCRIPT` (Rscript executable), `JMRI_DATASET` (cine dataset directory) and `JMRI_SEGMENTATION_ROOT` (the separately installed automated segmentation project). Use `--gpu 0` or a CUDA UUID for the public launcher. Segmentation weights from the private training datasets must be obtained separately.

## Input layout

The public launcher preserves the relative directory relationships of the analysis project. Place locally authorized inputs under `revision/`:

```text
CMR parameter/train.xlsx, external.xlsx
Radiomics/Radiomics_train.xlsx, Radiomics_external.xlsx
2026.11.06/inputs/analysis_manifest.csv
2026.11.06/inputs/pretrained.ckpt
2026.11.06/benchmark_1059/inputs/prepared_patients.json
2026.11.06/benchmark_1059/inputs/roi_added_records.json
```

The manifest has `cohort,ID,event`, with `development` or `external_test` and binary LGE-presence labels. CMR and radiomics workbooks have `ID,event` and candidate feature columns. `nested_validation/plans/protocol.json` lists the released CMR/radiomics candidates and tuning grids. Preserve patient ordering and ID alignment when reproducing the original splits. The current benchmark enforces 1059 development and 309 external patients.

Each `prepared_patients.json` record requires `ID`, `cohort`, `event`, `full_paths` and `roi_paths`. Paths are absolute paths to prepared per-slice NumPy cine arrays with dimensions time × height × width. `code/prepare_inputs.py` contains the normalization, resize and ROI-crop implementation and requires the original slice inventory and masks. `roi_added_records.json` supplies replacement `roi_paths` for recovered cases; provide an empty JSON array if all ROI paths are already complete. The checkpoint must have the encoder state dictionary expected by `code/train.py`.

The full-cohort tabular stage generates folds. The launcher copies those folds into the image-training plan and checks that existing plans agree. Input schemas describe researcher-provided inputs; the release contains aggregate outputs rather than the clinical input workbooks. Exact scientific reproduction requires the original images, labels, split membership and pretrained weights.

## Execution order

```bash
# Fold-specific CMR selection, Rscore fitting, classifier search and no-shape analysis
python revision/run.py tabular
# Internal epoch selection, outer refit, held-out and external inference, full + ROI
python revision/run.py train --gpu 0
# Shared-fold full-cine and ROI downstream comparisons
python revision/run.py fusion --gpu 0
python revision/run.py roi-fusion --gpu 0
python revision/run.py roi-frozen --gpu 0
# Standalone metrics and ROC/training plots
python revision/run.py metrics --gpu 0
# Development-fitted logistic recalibration sensitivity analysis
python revision/run.py recalibration
```

The legacy `nested_validation/code/experiments.py` controller includes the historical 1050-patient matched branch. The public launcher above calls the updated full-cohort stages directly. Additional scripts such as `prepare_roi9.py`, acquisition audits, response builders and historical report builders preserve study-specific input dependencies. Review their file reads before use. Word-fill scripts require the corresponding local manuscript templates.

Thresholds are selected using development predictions and applied to the external cohort. Calibration intercept and slope are estimated jointly against logit probabilities; their ideal values are zero and one. Recalibration fits a logistic probability mapping from development data and evaluates transferred probabilities on the external cohort. Detailed model definitions, confidence-interval calculations and threshold conventions are in the scripts.

## Release checks

Python syntax, launcher initialization, source-hash integrity and synthetic statistical checks are validated during release preparation. Full clinical training requires local clinical inputs and is not rerun as part of publishing the source. Aggregate CSV tables preserve the calculated precision, while manuscript display formatting can round to four decimal places.
