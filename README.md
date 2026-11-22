# Non-contrast cine CMR for predicting LGE presence in HCM

Code for evaluating conventional cardiac MRI variables, cine radiomics scores and pretrained mViT representations for LGE-presence prediction in hypertrophic cardiomyopathy. The binary outcome is LGE presence on contrast-enhanced CMR. The repository supports research evaluation of non-contrast predictors.

## November 2026 revision update

The revision source, protocols, aggregate results and figures are in [`revision/`](revision/README.md). The updated common-fold benchmark includes **1059 development patients and 309 external patients**, covering full-cine Frozen/FT, myocardial ROI FT/Frozen and CMR/Rscore fusion. The release also includes fold-specific CMR/Rscore construction, nested classifier selection, removal of shape features, calibration/recalibration, NRI and acquisition-parameter analyses.

| Material | Location |
| --- | --- |
| Revised workflow and input requirements | [Revision guide](revision/README.md) |
| Public execution entry point | [revision/run.py](revision/run.py) |
| Updated Table 2 aggregate metrics | [Nine models](revision/summary_results/benchmark_1059/results/nine_model_table2_metrics.csv), [ROI fusion](revision/summary_results/benchmark_1059/results/roi_fusion_table2_metrics.csv), [ROI-Frozen](revision/summary_results/benchmark_1059/roi_frozen/performance.csv) |
| ROC and training curves, PNG/PDF | [Figures](revision/figures/) |
| Nested CMR and Rscore construction | [R implementation](revision/2026.11.06/nested_validation/code/nested_tabular.R), [Classifier search](revision/2026.11.06/nested_validation/code/tabular_suite.py) |
| Calibration, uncertainty and statistical functions | [Statistics](revision/2026.11.06/code/revision_statistics.py), [Recalibration](revision/2026.11.06/recalibration/code/run.py) |
| Acquisition metadata and heterogeneity | [Acquisition analyses](revision/2026.11.06/acquisition_followup/code/) |
| Source provenance | [SHA-256 manifest](revision/source_manifest.json) |

## Start here

```bash
git clone https://github.com/Oubit1/Cardiac_MRI_HCM_OpenSource.git
cd Cardiac_MRI_HCM_OpenSource
pip install -r revision/requirements.txt
python revision/run.py init
python revision/run.py --help
```

Install the R packages and supply the input workbooks, prepared cine/ROI arrays and encoder checkpoint described in the [revision guide](revision/README.md), then run the listed stages in order. GPU selection is explicit through `--gpu`. Aggregate results and figures can be inspected directly from the repository.

## Earlier workflow

The original modules remain organized under `data_preparation/`, `radiomics/`, `models/`, `multimodal_fusion/`, `configs/` and `scripts/`. Their original instructions are preserved in the [historical README](revision/README_legacy.md). The revision guide identifies updated entry points and distinguishes the earlier 1050-patient matched analyses from the completed 1059-patient benchmark. Previously published assets are available through the repository [Releases](https://github.com/Oubit1/Cardiac_MRI_HCM_OpenSource/releases).

## 中文说明

本次更新提供修稿使用的实验源代码、参数配置、汇总结果和训练图。复现实验请按 revision 使用说明准备本地数据，并通过统一入口选择实验与 GPU。新版 Table 2 应使用 1059 例 benchmark 的结果文件。独立 mViT 分类头与提取表征后的融合分类器采用不同的预测流程，对照表格时需对应具体模型。

## License

Project code is distributed under the [MIT License](LICENSE). Existing third-party notices in vendored model code remain applicable. Clinical images, individual-level data and pretrained model assets require their respective access permissions.
