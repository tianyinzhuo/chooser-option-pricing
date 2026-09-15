"""Week 7 Streamlit prototype for the JPM simple chooser option project."""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import plotly.express as px
import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.week_7_pricing_service import (  # noqa: E402
    MODEL_SPECS,
    load_market_history,
    load_model_bundle,
    load_pricing_training_ranges,
    price_contract,
    run_contract_scenarios,
    volatility_sensitivity,
)
from src.week_7_realtime_data import collect_realtime_snapshot  # noqa: E402


st.set_page_config(
    page_title="JPM简单选择权定价实验平台",
    page_icon="📈",
    layout="wide",
)


@st.cache_resource(show_spinner="正在加载定价模型……")
def cached_model(model_key: str) -> dict[str, Any]:
    return load_model_bundle(model_key, PROJECT_ROOT)


@st.cache_data(show_spinner=False)
def cached_history() -> pd.DataFrame:
    return load_market_history(PROJECT_ROOT)


@st.cache_data(show_spinner=False)
def cached_training_ranges(model_key: str) -> dict[str, tuple[float, float]]:
    bundle = load_model_bundle(model_key, PROJECT_ROOT)
    return load_pricing_training_ranges(bundle, PROJECT_ROOT)


@st.cache_data(show_spinner=False)
def cached_csv(relative_path: str) -> pd.DataFrame:
    return pd.read_csv(PROJECT_ROOT / relative_path, encoding="utf-8-sig")


@st.cache_data(ttl=3600, show_spinner=False)
def cached_live_snapshot() -> dict[str, Any]:
    return collect_realtime_snapshot(
        project_root=PROJECT_ROOT,
        save=True,
        timeout=30.0,
        retries=1,
        news_lookback_days=7,
    )


def show_project_image(relative_path: str, caption: str) -> None:
    path = PROJECT_ROOT / relative_path
    if path.exists():
        st.image(str(path), caption=caption, use_container_width=True)
    else:
        st.info(f"尚未找到图表：{relative_path}")


def display_snapshot(snapshot: dict[str, Any] | None) -> None:
    if not snapshot:
        st.info("尚无最新已发布日频快照。点击“刷新最新日频快照”开始获取。")
        return
    st.caption(f"快照获取时间（UTC）：{snapshot.get('generated_at_utc', '未知')}")
    labels = {
        "jpm_daily": "JPM日线",
        "vix": "VIX",
        "treasury_10y": "10年期美债收益率",
        "jpm_news_sentiment": "JPM新闻情绪",
    }
    icons = {
        "ok": "✅",
        "cached": "🟡",
        "stale": "⚠️",
        "no_data": "⚪",
        "not_configured": "⚙️",
        "error": "❌",
    }
    status_labels = {
        "ok": "成功",
        "cached": "当天缓存",
        "stale": "历史备用值",
        "no_data": "无可用数据",
        "not_configured": "未配置",
        "error": "失败",
    }
    records = []
    for source_name, source in snapshot.get("sources", {}).items():
        data = source.get("data") or {}
        observation_date = data.get("date")
        if source_name == "jpm_news_sentiment":
            observation_date = (data.get("window") or {}).get("to_utc")
        last_success = source.get("last_successful_data") or {}
        status = source.get("status", "unknown")
        endpoint = source.get("endpoint")
        method = source.get("request_method")
        route = " / ".join(str(item) for item in (endpoint, method) if item) or "默认接口"
        records.append(
            {
                "数据项": labels.get(source_name, source_name),
                "状态": f"{icons.get(status, '•')} {status_labels.get(status, status)}",
                "实际提供方": source.get("data_provider") or source.get("provider") or "—",
                "观测日期/窗口": observation_date or "—",
                "接口路径/方式": route,
                "说明": source.get("message") or "正常",
                "是否保留上次成功值": "是" if last_success else "否",
            }
        )
    st.dataframe(pd.DataFrame(records), hide_index=True, use_container_width=True)

    sources = snapshot.get("sources", {})
    jpm_source = sources.get("jpm_daily", {})
    vix_source = sources.get("vix", {})
    rate_source = sources.get("treasury_10y", {})
    news_source = sources.get("jpm_news_sentiment", {})
    jpm = (jpm_source.get("data") or jpm_source.get("last_successful_data") or {})
    vix = (vix_source.get("data") or vix_source.get("last_successful_data") or {})
    rate = (rate_source.get("data") or rate_source.get("last_successful_data") or {})
    news = (news_source.get("data") or news_source.get("last_successful_data") or {})
    columns = st.columns(4)
    columns[0].metric(
        "最新JPM收盘价",
        f"${jpm['close']:.2f}" if jpm.get("close") is not None else "未获得",
    )
    columns[1].metric(
        "VIX（历史备用）" if vix_source.get("status") == "stale" else "最新VIX",
        f"{vix['value']:.2f}" if vix.get("value") is not None else "未获得",
    )
    columns[2].metric(
        "最新10年期利率",
        f"{rate['value']:.2f}%" if rate.get("value") is not None else "未获得",
    )
    columns[3].metric(
        "近7日新闻数（当天缓存）"
        if news_source.get("status") == "cached"
        else "近7日新闻数",
        str(news.get("article_count")) if news.get("article_count") is not None else "未获得",
    )


def load_saved_snapshot() -> dict[str, Any] | None:
    path = (
        PROJECT_ROOT
        / "data"
        / "external"
        / "realtime"
        / "latest_market_snapshot.json"
    )
    if not path.exists():
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else None
    except (OSError, json.JSONDecodeError):
        return None


st.title("JPM 简单选择权定价实验平台")
st.caption("第七周原型 · 历史交互定价、压力测试、SHAP解释与最新日频数据检查")

try:
    history = cached_history()
except Exception as error:
    st.error(f"无法读取历史特征数据：{error}")
    st.stop()

label_to_key = {spec["label"]: key for key, spec in MODEL_SPECS.items()}
with st.sidebar:
    st.header("定价设置")
    selected_label = st.selectbox(
        "定价模型",
        list(label_to_key),
        index=0,
        help="无情绪的Week 6 GBDT是当前主模型；情绪模型仅用于探索。",
    )
    model_key = label_to_key[selected_label]
    chosen_date = st.date_input(
        "市场日期",
        value=history["date"].max().date(),
        min_value=history["date"].min().date(),
        max_value=history["date"].max().date(),
    )
    eligible = history[history["date"].dt.date <= chosen_date]
    if eligible.empty:
        st.error("所选日期早于数据起点。")
        st.stop()
    market_row = eligible.iloc[-1]
    actual_date = pd.Timestamp(market_row["date"]).date()
    if actual_date != chosen_date:
        st.caption(f"该日不是交易日，已使用之前最近的交易日：{actual_date}")
    moneyness = st.select_slider(
        "价内外程度 K/S",
        options=[0.9, 1.0, 1.1],
        value=1.0,
        help="执行价 K = JPM现价 S × K/S。",
    )
    choice_time = st.radio(
        "选择时点 T1",
        options=[0.25, 0.5],
        format_func=lambda value: f"{value:.2f} 年",
        horizontal=True,
    )
    maturity = 1.0
    st.text_input("最终到期 T2", value="1.00 年（训练范围固定）", disabled=True)
    st.success("合约条款处于离散训练网格内；市场变量会另行检查预测试范围。")

try:
    bundle = cached_model(model_key)
    training_ranges = cached_training_ranges(model_key)
    pricing = price_contract(
        bundle,
        market_row,
        float(moneyness),
        float(choice_time),
        maturity,
    )
except Exception as error:
    st.error(f"定价失败：{error}")
    st.stop()

if MODEL_SPECS[model_key]["experimental"]:
    st.warning(
        "当前选择的是情绪实验模型。它的测试MAE略低，但CV和测试RMSE均未改善，"
        "因此不能视为主模型升级。"
    )

tab_price, tab_scenario, tab_explain, tab_data = st.tabs(
    ["🧮 单份合约定价", "🌪️ 压力情景", "🔍 模型比较与解释", "🔄 数据更新与说明"]
)

with tab_price:
    st.subheader("市场与合约快照")
    top = st.columns(5)
    top[0].metric("实际使用日期", pricing["date"])
    top[1].metric("JPM现价", f"${pricing['spot']:.2f}")
    top[2].metric("执行价", f"${pricing['strike']:.2f}")
    top[3].metric("VIX", f"{float(market_row['vix']):.2f}")
    top[4].metric("10年期美债利率", f"{float(market_row['treasury_10y']):.2f}%")

    st.subheader("定价结果")
    result_columns = st.columns(4)
    result_columns[0].metric("ML预测权利金", f"${pricing['ml_price_usd']:.2f}")
    result_columns[1].metric("权利金 / JPM现价", f"{pricing['normalized_price']:.2%}")
    result_columns[2].metric("BSM解析基准", f"${pricing['bsm_price_usd']:.2f}")
    signed_difference = float(pricing["difference_usd"])
    formatted_difference = (
        f"-${abs(signed_difference):.2f}"
        if signed_difference < 0
        else f"+${signed_difference:.2f}"
    )
    result_columns[3].metric(
        "ML − BSM",
        formatted_difference,
    )
    model_rmse = float(MODEL_SPECS[model_key]["test_rmse"])
    st.caption(
        f"历史测试RMSE约为 ${model_rmse:.3f}。"
        "这只是探索性误差尺度，不是置信区间，也不保证未来误差。"
    )
    for warning in pricing["warnings"]:
        st.warning(warning)

    st.subheader("波动率敏感性")
    sensitivity = volatility_sensitivity(
        bundle,
        market_row,
        float(moneyness),
        float(choice_time),
        maturity,
    )
    sensitivity_long = sensitivity.rename(
        columns={
            "historical_volatility": "历史波动率",
            "ml_price_usd": "ML模型",
            "bsm_price_usd": "BSM基准",
        }
    ).melt(
        id_vars=["volatility_multiplier", "历史波动率"],
        value_vars=["ML模型", "BSM基准"],
        var_name="定价方法",
        value_name="权利金（美元）",
    )
    figure = px.line(
        sensitivity_long,
        x="历史波动率",
        y="权利金（美元）",
        color="定价方法",
        markers=True,
        title="一因素情景响应：历史波动率变化（其他特征固定）",
    )
    figure.update_traces(line_shape="hv", selector={"name": "ML模型"})
    volatility_range = training_ranges.get("rolling_vol_60")
    if volatility_range:
        figure.add_vrect(
            x0=volatility_range[0],
            x1=volatility_range[1],
            fillcolor="rgba(70, 170, 110, 0.08)",
            line_width=0,
            annotation_text="预测试样本范围",
            annotation_position="top left",
            layer="below",
        )
    figure.add_vline(
        x=float(market_row["rolling_vol_60"]),
        line_dash="dash",
        annotation_text="当前波动率",
    )
    st.plotly_chart(figure, use_container_width=True)
    st.info(
        "深蓝色ML曲线呈阶梯状，是GBDT树模型按分裂阈值作分段常数预测的结果，"
        "不是期权真实价格发生跳跃。该图用于一因素情景比较，不能直接当作Vega等Greek；"
        "超出绿色训练范围时，树模型也不会像解析公式那样平滑外推。"
    )

    with st.expander("查看模型自动使用的主要数据"):
        fields = {
            "60日年化历史波动率": market_row["rolling_vol_60"],
            "20日年化历史波动率": market_row["rolling_vol_20"],
            "VIX 20日均值": market_row["vix_ma_20"],
            "JPM与VIX变化20日相关性": market_row["jpm_vix_corr_20"],
            "成交量/20日均量": market_row["volume_to_ma20_ratio"],
        }
        if "news_article_count_ma_5" in market_row:
            fields["前期新闻数5日均值"] = market_row["news_article_count_ma_5"]
            fields["前期新闻情绪20日均值"] = market_row["news_sentiment_ma_20"]
        st.dataframe(
            pd.DataFrame(
                {"变量": list(fields), "当前值": [float(value) for value in fields.values()]}
            ),
            hide_index=True,
            use_container_width=True,
        )

with tab_scenario:
    st.subheader("当前合约的五种情景（含基准）")
    scenarios = run_contract_scenarios(
        bundle,
        market_row,
        float(moneyness),
        float(choice_time),
        maturity,
        feature_ranges=training_ranges,
    )
    scenario_chart = px.bar(
        scenarios,
        x="预测权利金（美元）",
        y="情景",
        orientation="h",
        color="相对基准变化率（%）",
        color_continuous_scale="Blues",
        text_auto=".2f",
        title="当前合约在预设冲击下的预测权利金",
    )
    scenario_chart.update_layout(yaxis={"autorange": "reversed"})
    st.plotly_chart(scenario_chart, use_container_width=True)
    st.dataframe(
        scenarios.drop(columns=["scenario"]).style.format(
            {
                "预测权利金（美元）": "{:.4f}",
                "相对基准变化（美元）": "{:+.4f}",
                "相对基准变化率（%）": "{:+.2f}%",
            }
        ),
        hide_index=True,
        use_container_width=True,
    )
    st.info(
        "上图是当前单份合约的响应；第七周实验中的30/60份合约汇总结果是另一种统计口径，"
        "两者不应直接混为一谈。"
    )
    st.caption(
        "利率情景采用收益率曲线水平平行上移2个百分点：只改变treasury_10y水平，"
        "日变化与20日动量保持不变。VIX及其20日均值同时提高代表持续高VIX状态。"
    )
    if (scenarios["训练范围检查"] == "⚠ 超出").any():
        st.warning(
            "带“⚠ 超出”的情景含有预测试样本范围之外的输入。GBDT在范围外仍只能落入既有叶节点，"
            "所以平台或很小的变化不能解释为真实经济敏感度很小。"
        )
    cohort_path = "outputs/week_7/week_7_scenario_summary.csv"
    if (PROJECT_ROOT / cohort_path).exists():
        with st.expander("查看第七周完整测试集合约的汇总结果"):
            st.dataframe(cached_csv(cohort_path), hide_index=True, use_container_width=True)
            show_project_image(
                "outputs/week_7/week_7_scenario_mean_percentage_change.png",
                "五种情景的测试集平均价格变化",
            )

with tab_explain:
    st.subheader("有无新闻情绪的受控比较")
    st.info(
        "结论：加入情绪后测试MAE小幅改善约0.83%，但CV RMSE恶化1.98%、测试RMSE恶化1.07%，"
        "目前没有稳定证据证明它提高了定价精度。"
    )
    cv_path = "outputs/week_7/week_7_sentiment_model_cv_comparison.csv"
    test_path = "outputs/week_7/week_7_sentiment_model_test_comparison.csv"
    if (PROJECT_ROOT / cv_path).exists() and (PROJECT_ROOT / test_path).exists():
        left, right = st.columns(2)
        with left:
            st.markdown("**按日期交叉验证**")
            cv_table = cached_csv(cv_path).replace(
                {
                    "gbdt_without_sentiment": "GBDT（无情绪）",
                    "gbdt_with_sentiment": "GBDT（加入情绪）",
                }
            ).rename(
                columns={
                    "model_name": "模型",
                    "feature_count": "特征数",
                    "mae": "MAE（越低越好）",
                    "rmse": "RMSE（越低越好）",
                    "r2": "R²（越高越好）",
                }
            )
            st.dataframe(cv_table, hide_index=True, use_container_width=True)
        with right:
            st.markdown("**时间测试集**")
            test_table = cached_csv(test_path).replace(
                {
                    "gbdt_without_sentiment": "GBDT（无情绪）",
                    "gbdt_with_sentiment": "GBDT（加入情绪）",
                }
            ).rename(
                columns={
                    "model_name": "模型",
                    "feature_count": "特征数",
                    "mae": "MAE（越低越好）",
                    "rmse": "RMSE（越低越好）",
                    "r2": "R²（越高越好）",
                }
            )
            st.dataframe(test_table, hide_index=True, use_container_width=True)
    show_project_image(
        "outputs/week_7/week_7_sentiment_cv_test_rmse_comparison.png",
        "情绪特征加入前后的CV与测试RMSE",
    )

    st.subheader("SHAP解释")
    impact = st.columns(6)
    for column, (label, value) in zip(
        impact,
        [
            ("JPM历史波动率", "39.39%"),
            ("其他市场变量", "7.04%"),
            ("合约条款", "44.32%"),
            ("VIX核心", "5.55%"),
            ("新闻情绪", "2.82%"),
            ("JPM–VIX交互", "0.89%"),
        ],
    ):
        column.metric(label, value, help="测试集平均绝对SHAP幅度占比")
    st.caption("以上百分比是平均绝对SHAP幅度占比，不是模型精度贡献率。")
    st.warning(
        "非零SHAP值只说明模型在当前预测中使用了该特征；它不等于预测精度改善，"
        "也不表示因果关系。测试集虽有60份合约，但只有10个独立市场日期；"
        "高度相关的VIX、移动均值和情绪移动平均还可能彼此分摊归因。"
    )
    image_columns = st.columns(2)
    with image_columns[0]:
        show_project_image(
            "outputs/week_7/shap/week_7_shap_group_importance_usd.png",
            "特征组平均绝对SHAP贡献",
        )
        show_project_image(
            "outputs/week_7/shap/week_7_shap_beeswarm_usd.png",
            "测试集合约的SHAP贡献分布",
        )
    with image_columns[1]:
        show_project_image(
            "outputs/week_7/shap/week_7_shap_vix_sentiment_focus_usd.png",
            "VIX与新闻情绪重点特征",
        )
        show_project_image(
            "outputs/week_7/shap/week_7_shap_waterfall_representative_usd.png",
            "代表性合约的局部解释",
        )
    show_project_image(
        "outputs/week_7/shap/week_7_shap_vix_sentiment_dependence_usd.png",
        "VIX与情绪特征的描述性依赖关系",
    )

with tab_data:
    st.subheader("历史数据状态")
    status_columns = st.columns(3)
    status_columns[0].metric("特征数据行数", f"{len(history):,}")
    status_columns[1].metric("起始日期", str(history["date"].min().date()))
    status_columns[2].metric("截止日期", str(history["date"].max().date()))

    st.subheader("最新已发布日频数据检查")
    st.caption(
        "这里获取的是各供应商最新发布的日频观测，不是逐笔实时行情。不同数据源的观测日期可能不同。"
    )
    refresh = st.button("刷新最新日频快照", type="primary")
    snapshot: dict[str, Any] | None = load_saved_snapshot()
    if refresh:
        cached_live_snapshot.clear()
        try:
            with st.spinner("正在连接 Twelve Data、FRED 和 Alpha Vantage……"):
                snapshot = cached_live_snapshot()
            st.success("数据源检查完成。")
        except Exception as error:
            st.error(f"刷新失败：{error}")
            snapshot = load_saved_snapshot()
    display_snapshot(snapshot)

    auto_refresh = st.toggle("每60分钟自动检查一次（实验性）", value=False)
    if auto_refresh:
        @st.fragment(run_every="60m")
        def automatic_snapshot_fragment() -> None:
            try:
                latest = cached_live_snapshot()
                st.caption(
                    "自动检查已开启；最近一次缓存快照："
                    f"{latest.get('generated_at_utc', '未知')}"
                )
            except Exception as error:
                st.caption(f"自动检查暂时失败，仍保留上一次快照：{error}")

        automatic_snapshot_fragment()

    st.info(
        "本原型不会直接用最新快照覆盖训练数据或历史特征。完整的最新日期定价仍需至少60个交易日"
        "的同口径行情、VIX和利率数据，并重新生成滚动特征。"
    )

st.divider()
st.caption(
    "免责声明：监督学习标签是Heston蒙特卡洛合成参考价格，并非真实OTC成交权利金；"
    "结果仅用于课程实验与研究，不构成投资建议。SHAP是模型归因而非因果证明。"
    "新闻缺失表示未观测到新闻，不代表市场情绪中性。模型训练数据截止到2024年，"
    "更新日期属于样本外推。当前产品为欧式简单选择权，T2固定1年，BSM基准假定零股息与固定波动率。"
)
