import pandas as pd
import plotly.express as px
import os

# 현재 작업 디렉토리 확인
print("현재 디렉토리:", os.getcwd())

# 파일 경로 직접 지정
csv_path = "/mount/src/diabetes-prediction/data/diabetes_prediction_dataset.csv"  # 실제 위치에 맞게 수정

df = pd.read_csv(csv_path)

# 나이 구간 생성
df['age_group'] = pd.cut(df['age'], bins=[0, 20, 30, 40, 50, 60, 70, 120],
                         labels=['10대 이하', '20대', '30대', '40대', '50대', '60대', '70대 이상'])

# 당뇨병 발병률 계산
age_diabetes_rate = df.groupby('age_group')['diabetes'].mean().reset_index()
age_diabetes_rate['diabetes'] *= 100

# 그래프 시각화
fig = px.bar(age_diabetes_rate,
             x='age_group',
             y='diabetes',
             labels={'age_group': '연령대', 'diabetes': '당뇨병 유병률 (%)'},
             title='연령대별 당뇨병 유병률',
             color='diabetes',
             color_continuous_scale='reds')

# 그래프 저장 (GUI 없는 환경 대비)
fig.write_html("diabetes_age_plot.html")
