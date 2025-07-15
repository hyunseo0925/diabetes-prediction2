import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np

st.set_page_config(page_title="상관관계 분석", layout="wide")
st.title("📈 당뇨병 관련 변수 간의 상관관계 분석")

# 데이터 불러오기
@st.cache_data
def load_data():
    return pd.read_csv("diabetes_prediction_dataset.csv")

try:
    df = load_data()
except FileNotFoundError:
    st.error("❌ 'diabetes_prediction_dataset.csv' 파일이 없습니다.")
    st.stop()

# 주요 변수 선택
features = ['age', 'bmi', 'blood_glucose_level', 'hypertension', 'heart_disease']
if not all(col in df.columns for col in features + ['diabetes']):
    st.error("필요한 컬럼이 누락되었습니다. 다음 컬럼이 필요합니다: " + ", ".join(features + ['diabetes']))
    st.dataframe(df.head())
    st.stop()

# 상관계수 계산
corr = df[features + ['diabetes']].corr()

st.subheader("🔗 변수 간 상관계수")
st.dataframe(corr.style.background_gradient(cmap='RdBu_r', axis=None), height=350)

# 산점도 행렬 (Plotly)
st.subheader("🧬 산점도 행렬 (Scatter Matrix)")

fig = px.scatter_matrix(
    df,
    dimensions=['age', 'bmi', 'blood_glucose_level'],
    color='diabetes',
    title="주요 변수 간의 산점도 행렬 (당뇨병 여부 색상)",
    labels={col: col.upper() for col in ['age', 'bmi', 'blood_glucose_level', 'diabetes']}
)

fig.update_traces(diagonal_visible=False, showupperhalf=False, marker=dict(opacity=0.6, size=4))
fig.update_layout(height=700)
st.plotly_chart(fig, use_container_width=True)

# 해설
st.markdown("---")
st.markdown("### 🔬 생명과학적 해설")
st.markdown("""
- **BMI와 혈당**은 높은 상관관계를 보일 수 있으며, 이는 **비만 → 인슐린 저항성** → **고혈당**이라는 경로를 반영합니다.
- **혈압(고혈압)** 역시 대사 증후군의 일부로, 당뇨병과 동반 발생 가능성이 높습니다.
- **고령(노화)**는 인슐린 분비 감소, 인슐린 감수성 저하와 연관되어 있습니다.

➡️ 상관계수와 산점도는 기계학습 모델이 어떤 특성에 더 주목할지를 설명하는 데 도움을 줍니다.
""")
