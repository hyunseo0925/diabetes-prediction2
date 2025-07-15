import streamlit as st
import pandas as pd
import numpy as np
import os
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

st.set_page_config(page_title="생활습관 변화 시뮬레이터", layout="centered")
st.title("🧠 생활습관 변화 시뮬레이터: 당뇨병 위험 예측")

# CSV 파일 경로
file_path = "diabetes_prediction_dataset - diabetes_prediction_dataset.csv"

# 데이터 불러오기
@st.cache_data
def load_data():
    if not os.path.exists(file_path):
        st.error(f"❌ '{file_path}' 파일이 없습니다. 같은 디렉토리에 CSV 파일을 넣어주세요.")
        st.stop()

    df = pd.read_csv(file_path)
    df = df.dropna()

    # 인코딩
    df['gender'] = df['gender'].map({'Male': 0, 'Female': 1, 'Other': 2})
    df['smoking_history'] = df['smoking_history'].astype('category').cat.codes

    if 'physical_activity' in df.columns:
        df['physical_activity'] = np.where(df['physical_activity'] == "Yes", 1, 0)
    else:
        df['physical_activity'] = np.random.randint(0, 2, len(df))

    return df

df = load_data()

# 모델 학습
features = ['age', 'bmi', 'blood_glucose_level', 'gender', 'smoking_history', 'hypertension', 'heart_disease']
X = df[features]
y = df['diabetes']
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# 사용자 입력 받기
st.subheader("💡 현재 상태 입력")
age = st.slider("나이", 10, 100, 45)
bmi = st.slider("BMI (체질량지수)", 15.0, 45.0, 24.0)
glucose = st.slider("혈당 수치", 70, 300, 110)
gender = st.selectbox("성별", ['Female', 'Male', 'Other'])
smoking = st.selectbox("흡연 여부", ['never', 'former', 'current', 'ever', 'not current', 'No Info'])
hypertension = st.checkbox("고혈압 있음", value=False)
heart_disease = st.checkbox("심장질환 있음", value=False)

st.subheader("🔁 생활습관 변화 시뮬레이션")
smoking_new = st.selectbox("변경 후 흡연 여부", ['never', 'former', 'current', 'ever', 'not current', 'No Info'], index=0)

# 고정 인코딩 매핑
gender_map = {'Male': 0, 'Female': 1, 'Other': 2}
smoking_map = {
    'never': 0,
    'former': 1,
    'current': 2,
    'ever': 3,
    'not current': 4,
    'No Info': 5
}

# 입력 데이터프레임 생성
input_data = pd.DataFrame([{
    'age': age,
    'bmi': bmi,
    'blood_glucose_level': glucose,
    'gender': gender_map[gender],
    'smoking_history': smoking_map[smoking],
    'hypertension': int(hypertension),
    'heart_disease': int(heart_disease)
}])

input_new = input_data.copy()
input_new['smoking_history'] = smoking_map[smoking_new]

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
diff = prob_after - prob_before
if abs(diff) < 0.01:
    st.info("생활습관 변화로 인한 당뇨병 위험 변화가 거의 없습니다.")
elif diff < 0:
    st.success(f"생활습관 개선으로 당뇨병 위험이 **{abs(diff):.2f}% 감소**하였습니다.")
else:
    st.warning(f"생활습관 변화로 당뇨병 위험이 **{diff:.2f}% 증가**하였습니다.")

st.markdown("---")
st.markdown("### 🔬 예방의학적 시사점")
st.markdown("""
- 흡연은 인슐린 저항성을 증가시켜 당뇨병 위험을 높이는 주요 원인입니다.
- 금연, 체중 감량, 혈당·혈압 관리 등 생활습관 개선은 당뇨병을 예방하거나 지연시킬 수 있습니다.
- 이 시뮬레이터는 건강한 선택이 질병 위험도에 어떤 영향을 주는지 **직관적으로 보여줍니다**.
""")
