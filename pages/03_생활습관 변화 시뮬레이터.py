import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

st.set_page_config(page_title="생활습관 변화 시뮬레이터", layout="centered")
st.title("🧠 생활습관 변화 시뮬레이터: 당뇨병 위험 예측")

# 데이터 불러오기
@st.cache_data
def load_data():
    df = pd.read_csv("diabetes_prediction_dataset.csv")
    df = df.dropna()
    df['gender'] = df['gender'].map({'Male': 0, 'Female': 1, 'Other': 2})
    df['smoking_history'] = df['smoking_history'].astype('category').cat.codes
    df['physical_activity'] = (
        np.where(df['physical_activity'] == "Yes", 1, 0)
        if 'physical_activity' in df.columns
        else np.random.randint(0, 2, len(df))
    )
    return df

df = load_data()

# 모델 학습
features = ['age', 'bmi', 'blood_glucose_level', 'gender', 'smoking_history', 'hypertension', 'heart_disease']
X = df[features]
y = df['diabetes']
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# 사용자 입력
st.subheader("💡 현재 상태 입력")
age = st.slider("나이", 10, 100, 45)
bmi = st.slider("BMI (체질량지수)", 15.0, 45.0, 24.0)
glucose = st.slider("혈당 수치", 70, 300, 110)
gender = st.selectbox("성별", ['Female', 'Male', 'Other'])
smoking = st.selectbox("흡연 여부", ['never', 'former', 'current', 'ever', 'not current', 'No Info'])
hypertension = st.checkbox("고혈압 있음", value=False)
heart_disease = st.checkbox("심장질환 있음", value=False)

# 시뮬레이션
st.subheader("🔁 생활습관 변화 시뮬레이션")
smoking_new = st.selectbox("변경 후 흡연 여부", ['never', 'former', 'current', 'ever', 'not current', 'No Info'], index=0)

# 인코딩
input_data = pd.DataFrame([{
    'age': age,
    'bmi': bmi,
    'blood_glucose_level': glucose,
    'gender': {'Male': 0, 'Female': 1, 'Other': 2}[gender],
    'smoking_history': pd.Series([smoking]).astype('category').cat.codes[0],
    'hypertension': int(hypertension),
    'heart_disease': int(heart_disease)
}])

input_new = input_data.copy()
input_new['smoking_history'] = pd.Series([smoking_new]).astype('category').cat.codes[0]

# 예측
prob_before = model.predict_proba(input_data)[0][1] * 100
prob_after = model.predict_proba(input_new)[0][1] * 100

# 시각화
fig = go.Figure()
fig.add_trace(go.Indicator(
    mode="gauge+number+delta",
    value=prob_after,
    delta={'reference': prob_before, 'increasing': {'color': "red"}, 'decreasing': {'color': "green"}},
    gauge={
        'axis': {'range': [0, 100]},
        'bar': {'color': "darkblue"},
        'steps': [
            {'range': [0, 30], 'color': "lightgreen"},
            {'range': [30, 70], 'color': "yellow"},
            {'range': [70, 100], 'color': "red"}
        ]
    },
    title={'text': "예상 당뇨병 위험도 (%)"}
))
st.plotly_chart(fig, use_container_width=True)

# 결과 해석
st.markdown("### 🩺 결과 해석")
if prob_after < prob_before:
    st.success(f"생활습관 개선으로 당뇨병 위험이 **{prob_before - prob_after:.1f}% 감소**하였습니다.")
else:
    st.warning(f"생활습관 변화로 당뇨병 위험이 **{prob_after - prob_before:.1f}% 증가**하였습니다.")

st.markdown("---")
st.markdown("### 🔬 예방의학적 시사점")
st.markdown("""
- 흡연은 인슐린 저항성을 증가시키며, 이는 당뇨병 발생 위험을 높이는 주요 요인입니다.
- 생활습관 개선(금연, 체중 감량, 혈압 관리)은 당뇨병을 **예방하거나 지연**시키는 데 매우 효과적입니다.
- 본 시뮬레이터는 건강한 선택이 어떻게 질병 위험을 낮추는지를 **직관적으로 보여줍니다.**
""")
