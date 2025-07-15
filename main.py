import pandas as pd
import plotly.express as px

# 데이터 불러오기
df = pd.read_csv("diabetes_prediction_dataset - diabetes_prediction_dataset.csv")

# 나이 구간 설정
df['age_group'] = pd.cut(df['age'], bins=[0, 20, 30, 40, 50, 60, 70, 120],
                         labels=['10대 이하', '20대', '30대', '40대', '50대', '60대', '70대 이상'])

# 연령대별 당뇨병 발병률(%) 계산
age_diabetes_rate = df.groupby('age_group')['diabetes'].mean().reset_index()
age_diabetes_rate['diabetes'] *= 100

# 그래프 그리기
fig = px.bar(age_diabetes_rate,
             x='age_group',
             y='diabetes',
             labels={'age_group': '연령대', 'diabetes': '당뇨병 유병률 (%)'},
             title='연령대별 당뇨병 유병률',
             color='diabetes',
             color_continuous_scale='reds')

fig.show()
