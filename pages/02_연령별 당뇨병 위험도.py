import streamlit as st
import pandas as pd
import plotly.express as px

# 페이지 설정
st.set_page_config(page_title="연령별 당뇨병 위험도", layout="wide")
st.title("📊 나이에 따른 당뇨병 발병률 대시보드")

# 데이터 로드 함수
@st.cache_data
def load_data():
    df = pd.read_csv("diabetes_prediction_dataset - diabetes_prediction_dataset.csv")
    df = df[df['age'] <= 120]  # 이상치 제거
    df['age_group'] = pd.cut(
        df['age'],
        bins=[0, 20, 30, 40, 50, 60, 70, 120],
        labels=['10대 이하', '20대', '30대', '40대', '50대', '60대', '70대 이상'],
        right=False
    )
    return df

# 데이터 불러오기
df = load_data()

# 사용자 입력: 성별 + 흡연이력
col1, col2 = st.columns(2)
with col1:
    gender = st.selectbox("성별 선택", sorted(df['gender'].dropna().unique()), index=0)
with col2:
    smoke = st.selectbox("흡연 이력 선택", sorted(df['smoking_history'].dropna().unique()), index=0)

# 데이터 필터링
filtered = df[(df['gender'] == gender) & (df['smoking_history'] == smoke)]

# 연령대별 당뇨병 유병률 계산
summary = filtered.groupby('age_group', observed=True)['diabetes'].mean().reset_index()
summary['diabetes'] *= 100

# Plotly 시각화
fig = px.bar(
    summary,
    x='age_group',
    y='diabetes',
    labels={'age_group': '연령대', 'diabetes': '당뇨병 유병률 (%)'},
    title=f"연령대별 당뇨병 발병률 ({gender}, 흡연: {smoke})",
    color='diabetes',
    color_continuous_scale='Reds',
    text=summary['diabetes'].round(1).astype(str) + '%'
)

fig.update_traces(textposition='outside')
fig.update_layout(yaxis_range=[0, summary['diabetes'].max() * 1.2])

# 차트 출력
st.plotly_chart(fig, use_container_width=True)

# 해설 출력
st.markdown("---")
st.markdown("### 🔬 생명과학적 해설")
st.markdown("""
- **고령층**은 인슐린 분비량이 감소하고 감수성도 낮아져 당뇨병 발병률이 높습니다.
- **흡연**은 인슐린 저항성을 유발하고 대사 이상을 심화시켜 위험도를 높입니다.
- 이 분석은 연령과 생활습관이 당뇨병 발병에 어떤 영향을 미치는지 시각적으로 확인할 수 있게 해 줍니다.
""")
