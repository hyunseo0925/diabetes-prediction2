import streamlit as st
import pandas as pd
import numpy as np
import os
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

st.set_page_config(page_title="생활습관 변화 시뮬레이터", layout="centered")
st.title("🧠 생활습관 변화 시뮬레이터: 당뇨병 위험 예측")

# 데이터 불러오기
@st.cache_data
def load_data():
    file_path = "diabetes_prediction_dataset.csv"
    if not os.path.exists(file_path):
        st.error(f"❌ 파일 '{file_path}'이 존재하지 않습니다. 올바른 경로에 CSV 파일을 업로드했는지 확인해주세요.")
        st.stop()
    
    df = pd.read_csv(file_path)
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
