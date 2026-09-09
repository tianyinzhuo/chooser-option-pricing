# 第五周模型超参数维护

统一配置文件：`config/week_5_model_hyperparameters.json`。
本次维护保留已有的 14 个候选模型及其参数数值；修改参数前先保存配置版本。

## 配置分组

| 分组 | 用途 | 当前候选 |
| --- | --- | --- |
| volatility_random_forest | 预测未来 20 日实现波动率 | RF conservative / balanced |
| raw_dollar_gbdt | 预测美元权利金 | GBDT conservative / balanced |
| normalized_pricing_gbdt | 预测权利金与现货价之比 | normalized GBDT conservative / balanced |
| supervised_pricing_comparison | 在相同标准化数据上比较模型 | Ridge、RF、GBDT、MLP，各 2 组 |

原始美元 GBDT 与标准化 GBDT 是不同实验，参数并不相同。波动率 RF 与定价 RF 也分别维护。

## 可调整字段

| 参数 | 含义 | 当前配置 |
| --- | --- | --- |
| random_seed | 随机种子，帮助复现训练 | 42 |
| n_jobs | 随机森林并行任务数 | -1；设为 1 可限制为单个任务 |
| gbdt_loss | GBDT 训练损失 | huber |
| n_estimators | 树的数量 | 随候选而异 |
| max_depth | 每棵树允许的最大深度 | 随候选而异 |
| min_samples_leaf | 叶节点所需的最少样本数 | 随候选而异 |
| learning_rate | GBDT 每次提升的步长 | 0.05 或 0.04 |
| alpha | Ridge 或 MLP 的正则化系数；两类模型分别设置 | Ridge：0.1/1.0；MLP：0.01 |
| hidden_layer_sizes | MLP 各隐藏层的神经元数量 | [16] 或 [32, 16] |
| learning_rate_init | MLP 的初始学习率 | 0.001 |
| max_iter | MLP 最大迭代次数 | 2000 |
| early_stopping | MLP 是否启用内置提前停止 | false |
| selection_metric | 在验证集选择候选的指标，越小越好 | rmse；也支持 mae |

`target` 是目标的一致性约束。修改预测目标还涉及数据准备与训练代码，不能仅修改该名称。
`schema_version`、`scope`、`notes` 为说明字段。
本版本只接收表中对应模型已声明的参数键；新增未支持的参数会明确报错，避免参数被静默忽略。

## 使用步骤

1. 修改中央 JSON 中对应实验的候选值；每个候选的 model_name 必须唯一。
2. 运行对应训练脚本。模型按 selection_metric 在验证集选定，再在训练集加验证集上重新拟合。
3. 查看验证和测试 CSV，以及新生成的 selected_hyperparameters.json。
4. 将中央配置、训练代码及结果一起留存。测试集结果用于最终评估；本次复跑用于检查代码重构是否保留已有数值结果，不构成新的独立测试。

## 每次训练的参数记录

结果目录 `outputs/week_5/` 中增加：

- volatility_rf_selected_hyperparameters.json
- e2e_pricing_selected_hyperparameters.json
- normalized_pricing_selected_hyperparameters.json
- supervised_ridge_selected_hyperparameters.json
- supervised_random_forest_selected_hyperparameters.json
- supervised_gbdt_selected_hyperparameters.json
- supervised_mlp_selected_hyperparameters.json

这些记录包含选中模型、选模指标、候选参数、训练器实际参数（含库默认值）、scikit-learn 版本和完整中央配置快照。模型 joblib 同时保存配置与版本信息。
中央配置以后改变时，应以那次训练保存的快照解释旧结果。

横向比较可以移除某一模型族的全部候选，程序会按配置中实际保留的模型族训练。重新运行后，输出目录可能仍保留旧模型族的历史文件，判断本次参与模型应以本次比较 CSV 为准。

BSM 和 Heston 的金融模型参数继续维护于原有配置文件。第五周定价实验的目标来源为 Heston 蒙特卡洛合成参考价格。
