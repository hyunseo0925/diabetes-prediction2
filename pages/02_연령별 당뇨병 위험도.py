import pandas as pd
import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.express as px

# 데이터 불러오기
df = pd.read_csv("diabetes_prediction_dataset.csv")

# 전처리
df = df[df['age'] <= 120]  # 이상치 제거

# 연령대 컬럼 생성
df['age_group'] = pd.cut(
    df['age'],
    bins=[0, 20, 30, 40, 50, 60, 70, 120],
    labels=['10대 이하', '20대', '30대', '40대', '50대', '60대', '70대 이상'],
    right=False
)

# Dash 앱 생성
app = dash.Dash(__name__)
app.title = "당뇨병 발병률 대시보드"

# 앱 레이아웃
app.layout = html.Div([
    html.H1("📊 나이에 따른 당뇨병 발병률 대시보드", style={'textAlign': 'center'}),

    html.Div([
        html.Label("성별 선택"),
        dcc.Dropdown(
            id='gender_filter',
            options=[{'label': g, 'value': g} for g in sorted(df['gender'].unique())],
            value='Female',
            clearable=False
        ),
        html.Label("흡연 여부 선택"),
        dcc.Dropdown(
            id='smoke_filter',
            options=[{'label': s, 'value': s} for s in sorted(df['smoking_history'].dropna().unique())],
            value='never',
            clearable=False
        ),
    ], style={'width': '30%', 'display': 'inline-block', 'padding': '20px'}),

    dcc.Graph(id='bar_chart', style={'width': '100%', 'height': '600px'}),

    html.Div([
        html.P("※ 인슐린 민감도는 나이가 들수록 감소하며, 흡연은 인슐린 저항성과 대사 이상을 악화시켜 당뇨병 위험을 높입니다.",
               style={'fontSize': 16, 'padding': '10px'})
    ])
])

# 콜백 함수
@app.callback(
    Output('bar_chart', 'figure'),
    [Input('gender_filter', 'value'),
     Input('smoke_filter', 'value')]
)
def update_chart(gender, smoke):
    filtered = df[(df['gender'] == gender) & (df['smoking_history'] == smoke)]
    summary = filtered.groupby('age_group', observed=True)['diabetes'].mean().reset_index()
    summary['diabetes'] *= 100

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
    return fig

# 실행
if __name__ == '__main__':
    app.run_server(debug=True)
