import streamlit as st
import pandas as pd
import plotly.express as px
import os

st.set_page_config(page_title="당뇨병 시각화", layout="centered")

st.title("📊 연령대별 당뇨병 유병률 시각화")

# CSV 경로 설정
csv_path = "diabetes_prediction_dataset - diabetes_prediction_dataset.csv"

# 파일 존재 여부 확인
if not os.path.exists(csv_path):
    st.error(f"❌ CSV 파일이 존재하지 않습니다: {csv_path}")
else:
    # CSV 불러오기
    df = pd.read_csv(csv_path)

    # 나이 그룹 나누기
    df['age_group'] = pd.cut(df['age'],
                             bins=[0, 20, 30, 40, 50, 60, 70, 120],
                             labels=['10대 이하', '20대', '30대', '40대', '50대', '60대', '70대 이상'])

    # 연령대별 당뇨병 유병률 계산
    age_diabetes_rate = df.groupby('age_group')['diabetes'].mean().reset_index()
    age_diabetes_rate['diabetes'] *= 100  # 퍼센트 변환

    # 그래프 생성
    fig = px.bar(age_diabetes_rate,
                 x='age_group',
                 y='diabetes',
                 labels={'age_group': '연령대', 'diabetes': '당뇨병 유병률 (%)'},
                 title='연령대별 당뇨병 유병률',
                 color='diabetes',
                 color_continuous_scale='reds')

    # 그래프 출력
    st.plotly_chart(fig, use_container_width=True)

    # 데이터표 보기 토글
    with st.expander("📄 유병률 데이터표 보기"):
        st.dataframe(age_diabetes_rate, use_container_width=True)
