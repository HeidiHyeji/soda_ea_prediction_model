# =========================================================
# Streamlit Rolling Forecast App (ALL-IN-ONE 최종본)
# =========================================================

import streamlit as st
import pandas as pd
import joblib
import altair as alt

# -------------------------------------------------
# 1) 페이지 설정
# -------------------------------------------------
st.set_page_config(
    page_title="탄산음료 판매량 Rolling Forecast",
    layout="centered"
)

st.title("🥤 탄산음료 판매량 Rolling Forecast")
st.caption("lag/rolling 기반 장기 예측 + 기온 시나리오 + 제품 유형 비교")

# -------------------------------------------------
# 2) 리소스 로드
# -------------------------------------------------
@st.cache_resource
def load_model():
    return joblib.load("./outputs/models/final_lgbm_model.pkl")

@st.cache_data
def load_data():
    df = pd.read_csv("./outputs/data/df_4.csv")
    df["판매일"] = pd.to_datetime(df["판매일"])
    df = df.sort_values("판매일").reset_index(drop=True)
    return df

model = load_model()
df = load_data()

# -------------------------------------------------
# 3) 월별 평균 기온 (미래 대체용)
# -------------------------------------------------
monthly_temp = (
    df.assign(month=df["판매일"].dt.month)
      .groupby("month")["기온"]
      .mean()
)

# -------------------------------------------------
# 4) 사이드바 UI
# -------------------------------------------------
with st.sidebar:
    st.header("⚙️ 예측 설정")

    seed_date = st.date_input(
        "기준일 (마지막 실측일)",
        value=df["판매일"].max().date(),
        min_value=df["판매일"].min().date(),
        max_value=df["판매일"].max().date()
    )

    end_date = st.date_input(
        "예측 종료일",
        value=pd.to_datetime("2027-12-31")
    )

    HISTORY_WINDOW = st.number_input(
        "Seed window (일)",
        min_value=30, max_value=90, value=30, step=7
    )

    temp_delta = st.slider(
        "🌡️ 기온 시나리오 (Δ℃)",
        min_value=-3.0, max_value=3.0, value=0.0, step=0.5
    )

    product_type = st.radio(
        "🥤 제품 유형",
        options=["전체", "일반", "제로"],
        horizontal=True
    )

    run_btn = st.button("🔮 예측 실행")

# -------------------------------------------------
# 5) 보조 함수
# -------------------------------------------------
def get_season(month):
    if month in [12, 1, 2]:
        return "겨울"
    elif month in [3, 4, 5]:
        return "봄"
    elif month in [6, 7, 8]:
        return "여름"
    else:
        return "가을"

feature_cols = [
    '기온',
    '계절_봄', '계절_여름', '계절_가을', '계절_겨울',
    '주말여부',
    '제로구분_제로',
    'EA_lag1', 'EA_lag7', 'EA_lag14',
    'EA_ma7', 'EA_ma14', 'EA_ma30'
]

# -------------------------------------------------
# 6) Rolling Forecast 함수
# -------------------------------------------------
def run_forecast(seed_df, future_dates, zero_flag):
    history = seed_df[["EA"]].copy()
    results = []

    for current_date in future_dates:
        row = {}

        m = current_date.month
        row["기온"] = monthly_temp[m] + temp_delta
        row["주말여부"] = int(current_date.weekday() >= 5)
        row["제로구분_제로"] = zero_flag

        season = get_season(m)
        row["계절_봄"]   = int(season == "봄")
        row["계절_여름"] = int(season == "여름")
        row["계절_가을"] = int(season == "가을")
        row["계절_겨울"] = int(season == "겨울")

        row["EA_lag1"]  = history["EA"].iloc[-1]
        row["EA_lag7"]  = history["EA"].iloc[-7]
        row["EA_lag14"] = history["EA"].iloc[-14]
        row["EA_ma7"]   = history["EA"].iloc[-7:].mean()
        row["EA_ma14"]  = history["EA"].iloc[-14:].mean()
        row["EA_ma30"]  = history["EA"].iloc[-30:].mean()

        X = pd.DataFrame([row])[feature_cols]
        y_pred = model.predict(X)[0]

        results.append({"판매일": current_date, "예측_EA": y_pred})

        history = pd.concat(
            [history, pd.DataFrame({"EA": [y_pred]})],
            ignore_index=True
        ).iloc[-HISTORY_WINDOW:]

    return pd.DataFrame(results)

# -------------------------------------------------
# 7) 예측 실행
# -------------------------------------------------
if run_btn:
    seed_df = df[df["판매일"] <= pd.to_datetime(seed_date)].tail(HISTORY_WINDOW)

    if len(seed_df) < 30:
        st.error("Seed window는 최소 30일 이상 필요합니다.")
        st.stop()

    forecast_start = pd.to_datetime(seed_date) + pd.Timedelta(days=1)
    future_dates = pd.date_range(start=forecast_start, end=end_date, freq="D")

    if product_type == "일반":
        df_pred = run_forecast(seed_df, future_dates, zero_flag=0)

    elif product_type == "제로":
        df_pred = run_forecast(seed_df, future_dates, zero_flag=1)

    else:  # 전체
        df_pred_n = run_forecast(seed_df, future_dates, zero_flag=0)
        df_pred_z = run_forecast(seed_df, future_dates, zero_flag=1)

        df_pred = df_pred_n.copy()
        df_pred["예측_EA"] = (
            df_pred_n["예측_EA"].values + df_pred_z["예측_EA"].values
        )

    # -------------------------------------------------
    # 8) 실측 데이터 (2024)
    # -------------------------------------------------
    df_actual = (
        df[df["판매일"].dt.year == 2024]
        [["판매일", "EA"]]
        .rename(columns={"EA": "실측_EA"})
    )

    # -------------------------------------------------
    # 9) 요약
    # -------------------------------------------------
    st.subheader("📈 예측 요약")
    st.metric("총 예측 판매량(EA)", f"{int(df_pred['예측_EA'].sum()):,}")
    st.caption(
        f"🌡️ 기온 시나리오: {temp_delta:+.1f}℃ | "
        f"🥤 제품 유형: {product_type}"
    )

    # -------------------------------------------------
    # 10) 그래프 (tooltip 완전 적용)
    # -------------------------------------------------
    actual_line = (
        alt.Chart(df_actual)
        .mark_line(color="#1f77b4", point=True)
        .encode(
            x="판매일:T",
            y=alt.Y("실측_EA:Q", title="판매량(EA)"),
            tooltip=[
                alt.Tooltip("판매일:T", title="판매일"),
                alt.Tooltip("실측_EA:Q", title="실측 판매량", format=",.0f")
            ]
        )
    )

    forecast_line = (
        alt.Chart(df_pred)
        .mark_line(color="#ff7f0e", point=True)
        .encode(
            x="판매일:T",
            y="예측_EA:Q",
            tooltip=[
                alt.Tooltip("판매일:T", title="판매일"),
                alt.Tooltip("예측_EA:Q", title="예측 판매량", format=",.0f")
            ]
        )
    )

    vline = (
        alt.Chart(pd.DataFrame({"판매일": [forecast_start]}))
        .mark_rule(strokeDash=[6, 4], color="red")
        .encode(x="판매일:T")
    )

    vline_text = (
        alt.Chart(pd.DataFrame({
            "판매일": [forecast_start],
            "label": ["예측 시작"]
        }))
        .mark_text(dx=5, dy=-5, color="red")
        .encode(x="판매일:T", text="label:N")
    )

    chart = (
        (actual_line + forecast_line + vline + vline_text)
        .properties(
            title=f"2024 실측 + 2025–2027 예측 판매량 추이 ({product_type})",
            height=400
        )
        .interactive()
    )

    st.altair_chart(chart, use_container_width=True)

    st.markdown("---")
    st.subheader("📄 예측 데이터")
    st.dataframe(df_pred, use_container_width=True)

