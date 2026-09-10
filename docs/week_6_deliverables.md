# 第六周交付物清单

## 1. 配置与方法

- `config/week_6_model_search.json`：集中记录随机种子、交叉验证方法、超参数搜索空间、评价指标、SHAP设置和研究限制。
- `src/week_6_cv.py`：时间序列交叉验证及按日期分组的扩展窗口切分工具。

## 2. 模型训练与评估代码

- `src/tune_week_6_volatility.py`：未来20日波动率随机森林随机搜索、最终训练和测试评估。
- `src/tune_week_6_pricing_models.py`：Ridge、随机森林、GBDT和MLP的分组时间交叉验证及超参数搜索。
- `src/evaluate_week_6_route_a.py`：机器学习波动率加BSM混合路线评估。
- `src/explain_week_6_gbdt_shap.py`：GBDT全局与局部SHAP解释。
- `src/plot_week_6_results.py`：生成第六周模型比较图。

## 3. 训练模型

- `models/week_6/week_6_tuned_volatility_random_forest.joblib`
- `models/week_6/week_6_tuned_ridge_pricing.joblib`
- `models/week_6/week_6_tuned_random_forest_pricing.joblib`
- `models/week_6/week_6_tuned_gbdt_pricing.joblib`
- `models/week_6/week_6_tuned_mlp_pricing.joblib`

## 4. 关键结果表

- `outputs/week_6/volatility_cv_search_results.csv`
- `outputs/week_6/volatility_cv_model_comparison.csv`
- `outputs/week_6/volatility_test_model_comparison.csv`
- `outputs/week_6/pricing_cv_search_results.csv`
- `outputs/week_6/pricing_cv_model_comparison.csv`
- `outputs/week_6/pricing_test_model_comparison.csv`
- `outputs/week_6/route_a_test_metrics.csv`
- `outputs/week_6/final_pricing_approach_test_comparison.csv`
- `outputs/week_6/final_pricing_approach_test_predictions.csv`
- `outputs/week_6/pricing_selected_hyperparameters.json`
- `outputs/week_6/route_a_evaluation_summary.json`

## 5. 图表与可解释性材料

- `outputs/week_6/week_6_final_test_metric_comparison.png`
- `outputs/week_6/week_6_cv_vs_test_rmse.png`
- `outputs/week_6/week_6_gbdt_test_scatter.png`
- `outputs/week_6/week_6_volatility_cv_vs_test_rmse.png`
- `outputs/week_6/shap/gbdt_shap_summary_bar_usd.png`
- `outputs/week_6/shap/gbdt_shap_beeswarm_usd.png`
- `outputs/week_6/shap/gbdt_shap_waterfall_representative_usd.png`
- `outputs/week_6/shap/gbdt_shap_global_importance_usd.csv`
- `outputs/week_6/shap/gbdt_shap_representative_contributions_usd.csv`
- `outputs/week_6/shap/gbdt_shap_analysis_summary.json`

## 6. 自动测试

- `tests/test_week_6_cv.py`：验证时间隔离和日期分组。
- `tests/test_week_6_pipeline.py`：验证模型、结果表、预测和SHAP交付物。
- 全项目测试结果：21项全部通过。

## 7. 重要披露

- Heston蒙特卡洛价格是合成参考标签，不是真实OTC成交权利金。
- 第五周测试期此前已经被观察，因此第六周结果属于探索性研究，并非完全纯净的最终留出检验。
- SHAP解释描述模型行为，不代表因果效应。
- 路线A存在20日波动率预测期限与1年期选择权期限不匹配的问题。

