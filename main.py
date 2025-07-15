import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

st.set_page_config(page_title="당뇨병 예측기", layout="centered")
st.title("🧬 당뇨병 발병 확률 예측기")

# CSV 경로
csv_path = "diabetes_prediction_dataset - diabetes_prediction_dataset.csv"

# 데이터 불러오기
@st.cache_data
def load_data():
    df = pd.read_csv(csv_path)
    
    # 범주형 데이터 인코딩
    df = df.dropna()
    label_cols = ['gender', 'smoking_history']
    encoders = {}
    for col in label_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        encoders[col] = le
    return df, encoders

df, encoders = load_data()

# 특성과 타겟
X = df.drop(columns=['diabetes'])
y = df['diabetes']

# 모델 훈련
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)

# 사용자 입력 폼
st.subheader("👤 환자 정보 입력")
col1, col2 = st.columns(2)

with col1:
    age = st.slider("나이", 1, 100, 40)
    bmi = st.number_input("BMI (체질량지수)", 10.0, 50.0, 22.0)
    gender = st.selectbox("성별", encoders['gender'].classes_)
    smoking = st.selectbox("흡연 이력", encoders['smoking_history'].classes_)

with col2:
    hypertension = st.radio("고혈압 여부", ["없음", "있음"])
    heart_disease = st.radio("심장질환 여부", ["없음", "있음"])
    hba1c = st.number_input("당화혈색소 수치 (HbA1c)", 3.0, 15.0, 5.5)
    glucose = st.number_input("혈당 수치 (mg/dL)", 50, 300, 100)

# 예측 버튼
if st.button("🔍 당뇨병 발병 확률 예측"):
    # 입력값 변환
    input_data = pd.DataFrame([[
        encoders['gender'].transform([gender])[0],
        age,
        1 if hypertension == "있음" else 0,
        1 if heart_disease == "있음" else 0,
        encoders['smoking_history'].transform([smoking])[0],
        bmi,
        hba1c,
        glucose
    ]], columns=X.columns)

    # 예측
    proba = model.predict_proba(input_data)[0][1] * 100

    # 결과 시각화 (Plotly 게이지 차트)
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=proba,
        title={'text': "당뇨병 발병 확률 (%)"},
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': "crimson"},
            'steps': [
                {'range': [0, 30], 'color': "lightgreen"},
                {'range': [30, 70], 'color': "yellow"},
                {'range': [70, 100], 'color': "orangered"},
            ],
        }
    ))

    st.plotly_chart(fig, use_container_width=True)

    # 간단한 해석
    if proba >= 70:
        st.error("⚠️ 당뇨병 위험이 매우 높습니다. 전문의 상담을 권장합니다.")
    elif proba >= 30:
        st.warning("⚠️ 당뇨병 위험이 중간입니다. 식습관/생활 습관 개선이 필요합니다.")
    else:
        st.success("✅ 현재 당뇨병 위험이 낮은 편입니다.")
