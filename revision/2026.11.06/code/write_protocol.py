"""Write the prospective revision protocol before new training starts."""
from common import ROOT, GPU_UUID, write_json, timestamp


def main():
    plan = {
        'protocol_version': '2026-11-06-v1', 'created_at': timestamp(),
        'purpose': 'mViT-only reviewer supplement; no claim of first use of the external cohort',
        'reference_table': 'inputs/Main_tables_verified.xlsx / Table 4',
        'reference_auc': {'development': .7927354360758783, 'external': .8016922777669714},
        'data': {'development_n': 1050, 'external_n': 309, 'max_slices': 8, 'frames': 16,
                 'image_size': [224,224], 'labels': 'analysis_manifest.csv; matches current CMR/Radiomics spreadsheets',
                 'cine': 'migrated per-slice cine joined to historical index by cohort, ID, slice_index',
                 'historical_split': 'Explicit reproduction of archived five-fold development evaluation, seed 42; external cohort kept separate',
                 'roi': 'Time-union cine-mask box with 20% margin per side, reference-free QC; full-image fallback; all patients retained',
                 'mask_limitation': 'Existing binary masks have incompletely verified generation provenance; ROI results exploratory, never manual ground truth'},
        'training': {'outer_folds': 5, 'split_seed': 42, 'inner_validation_fraction': .1,
                     'max_epochs': 60, 'early_stopping': False, 'encoder_lrs': [1e-5, 3e-6, 3e-5],
                     'head_lr': 1e-4, 'weight_decay': 1e-4, 'dropout': .25, 'encoder_frozen_epochs': 2,
                     'adam_betas': [.9,.999], 'adam_eps': 1e-8, 'gradient_accumulation_patients': 4,
                     'encoder_microbatch_slices': 8, 'clip_grad_norm': 1., 'mixed_precision': 'float16 with GradScaler',
                     'primary_training_seed': 42, 'additional_training_seeds': [142,242],
                     'epoch_selection': 'Highest inner-validation AUC; improvement >1e-4; retain earlier checkpoint on ties',
                     'lr_selection': 'Per outer fold, highest best inner-validation AUC; lower LR on exact tie',
                     'outer_evaluation': 'Refit from pretrained initialization on all outer-training patients for the locked epoch count; then evaluate outer-held-out patients',
                     'final_selection': 'LR mode across five selected LRs (lower on ties); median selected epoch, minimum 3; full-development refit',
                     'seed_sensitivity': 'Refit using the fold-specific LR and epoch selected with seed 42; no new selection on seed results'},
        'arms': ['baseline20_full', 'extended60_full', 'tuned60_full', 'tuned60_roi',
                 'tuned60_full_seed142', 'tuned60_full_seed242'],
        'stages': ['Audit source labels and reproduce table values', 'Cache source-matched cine and exploratory ROI inputs',
                   'Five folds of three LR candidates, 60 epochs each',
                   'Fold-specific baseline20/extended60/tuned60 refits and held-out predictions',
                   'ROI 60-epoch trajectories with paired full-image LR, followed by fold refits',
                   'Additional-seed fixed-configuration refits',
                   'Lock all final model configurations and thresholds from development only',
                   'Full-development fits and external sensitivity evaluation',
                   'Generate statistics, figures, complete workbook, response Word'],
        'statistics': {'bootstrap_replicates': 2000, 'bootstrap_unit': 'patient, stratified by event',
                       'threshold': 'Maximum specificity subject to development OOF sensitivity >=0.95; lowest finite threshold on exact tie',
                       'metrics': ['AUC','sensitivity','specificity','PPV','NPV','accuracy','Brier',
                                   'calibration_intercept_joint','calibration_slope_joint','TP','FP','TN','FN'],
                       'comparisons': 'Paired patient bootstrap delta-AUC, paired DeLong, Holm-adjusted P for prespecified primary image comparisons',
                       'subgroups': ['external field strength','external in-plane resolution split at development median'],
                       'missing_policy': 'No outcome-driven exclusions; insufficient-class subgroup metrics marked not estimable'},
        'fusion_sensitivity': {'role': 'External-only exploratory fixed-classifier representation fusion, not new nested OOF claims',
                              'cmr_columns': ['MWT','LAs','LVGRS','LVGLS'],
                              'rscore': 'User-confirmed 26.9.19 R scripts and provided scores; do not silently replace with old Python Rscore',
                              'classifier': 'StandardScaler + elastic-net logistic regression, C=0.1, l1_ratio=0.5, fixed before new training',
                              'cv_limitation': 'Full-pipeline nested CMR/Rscore selection is outside the image-only training supplement; no independence claims for legacy fusion OOF',
                              'external_threshold': 'No new fused threshold claim; report AUC, Brier, calibration and paired incremental AUC only'},
        'gpu': {'uuid': GPU_UUID, 'physical_index_at_launch': 1, 'max_visible_devices': 1,
                'policy': 'One serial training process; never terminate other users jobs or change devices automatically'},
        'reporting': {'word': 'response/mViT_response.docx', 'workbook': 'tables/mViT_revision_results.xlsx',
                      'figure_dpi': 300, 'formats': ['PNG','PDF'],
                      'pending_results': 'Explicitly marked pending; never substitute archived results for uncompleted new experiments'},
    }
    dest = ROOT / 'plans/protocol.json'
    if dest.exists():
        raise FileExistsError('Protocol already written; do not overwrite after training begins.')
    write_json(dest, plan)
    text = '''# mViT 修订实验计划（2026-11-06）

所有新增代码、输入版本快照、日志、缓存、训练权重、表格、图与 Word 均保存在本目录。仅使用 GPU 1（UUID 写入 protocol.json）。各模型顺序训练，断点保存在各 run 的 latest.pt；不会占用其他卡。

## 版本及标签

Table 4 的 mViT-FT 开发 AUC=0.792735，外部 AUC=0.801692。逐例结果位于 inputs/submitted_ft_*.csv。旧归档的图像预测概率相同，但有 6 例外部标签不同；新版标签与当前临床及 radiomics 表一致。标签修订本身不能解释为模型改善。

用户确认 26.9.19 下两个 R 脚本为实际使用版本，原样存入 inputs。Rscore 使用重复 10 折外层、10 折内层、5 次重复、alpha 0.1–0.9 和 lambda.1se；CMR 脚本在全开发集进行 LASSO 并明确将开发性能标为 apparent。不得把它们写成包含后续 mViT 融合的完全嵌套流程。

## 必做实验

1. 五个外层折，各自保留 10% 外层训练患者用于内部验证。原学习率训练 60 轮，不早停；分别在前 20 轮和全部 60 轮选最佳 epoch。训练/验证 loss、AUC 均逐轮记录，训练评估使用无增强输入。
2. 编码器学习率 3e-6、1e-5、3e-5，分类头固定 1e-4。所有候选只在内部验证选择。选定后在全部外层训练患者重训，再预测未见过的外层保留患者。
3. 心肌区域裁剪使用现有 cine 掩膜的时间并集和固定边距，失败时回退整图并记录，不排除患者。现有掩膜来源尚未完整核实，结果只能标作探索性定位敏感性分析，不能称专家分割或证明公平性完全解决。
4. 额外训练种子 142、242 使用 seed 42 内部已选定的学习率/epoch 重训，数据划分固定。不得选外部表现最好的种子。
5. 最终配置由开发折选择结果的预定汇总规则锁定，全开发集重训后评价外部队列。外部队列此前已被使用，此次属于修订敏感性验证，不写成首次盲测。

## 图表和回复

图以英文轴标签、统一色板、清楚标明 inner validation/OOF/external，保存 300 DPI PNG 与矢量 PDF。主表记录 N、事件数、阈值、混淆矩阵、AUC/敏感度/特异度/NPV/PPV/Brier/校准及 CI；额外保存逐例预测、逐轮历史、参数、运行耗时和来源哈希。小亚组和不完整运行标记清楚。每完成一个训练阶段更新 Excel 与 response Word，全部完成后才生成 FINAL_COMPLETE 标记。

原始 Table 4 保留为历史对照，不自动覆盖。新的融合分析使用事先固定的正则化分类器，只提供外部敏感性结果，不以全局 OOF Rscore 冒充严格嵌套融合估计。

## 运行

使用 /home/wangjiahui/miniconda3/envs/myenv/bin/python code/run_queue.py。默认先检查 prepared_patients.json 完整性，再顺序训练。查看 status.json 和 logs/queue.log；所有卡由 GPU UUID 固定。无需反复启动队列；系统 flock 阻止重复运行。
'''
    (ROOT / 'README.md').write_text(text, encoding='utf-8')


if __name__ == '__main__':
    main()
