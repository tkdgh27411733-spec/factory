import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

st.set_page_config(page_title="공장 통합 전력 관제 시스템", layout="wide")

st.title("🏭 공장 통합 전력 관제 시스템")
st.markdown("전력 수요 예측, 피크 관리 및 생산 효율 최적화 대시보드")

st.sidebar.header("시뮬레이션 설정")
start_date = st.sidebar.date_input("시작 일자", datetime.today())
end_date = st.sidebar.date_input("종료 일자", datetime.today() + timedelta(days=7))

base_load = st.sidebar.slider("기본 부하 (kWh)", 100, 2000, 500)
op_rate = st.sidebar.slider("설비 가동률 (%)", 10, 100, 80)
peak_limit = st.sidebar.slider("피크 임계치 (kWh)", 500, 2500, 1500)

unit = st.sidebar.radio("데이터 단위", ["시간별", "일별"])

def calculate_data():
    days = (end_date - start_date).days + 1
    periods = days * 24 if unit == "시간별" else days
    
    # 데이터 생성
    data = []
    current_time = start_date
    for i in range(periods):
        val = base_load * (op_rate / 100) * (0.7 + np.random.rand() * 0.6)
        data.append(val)
        
    df = pd.DataFrame({'전력량 (kWh)': data})
    return df

if st.button("데이터 분석 실행"):
    df = calculate_data()
    
    # 지표 계산
    total_power = df['전력량 (kWh)'].sum()
    peak_power = df['전력량 (kWh)'].max()
    load_factor = (df['전력량 (kWh)'].mean() / peak_power) * 100
    
    # 메트릭 표시
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("총 누적 전력", f"{total_power/1000:.2f} MWh")
    col2.metric("최대 부하(Peak)", f"{peak_power:.0f} kWh")
    col3.metric("전력 효율(부하율)", f"{load_factor:.1f}%")
    
    # 피크 경보
    if peak_power > peak_limit:
        col4.metric("운영 상태", "경고", delta="초과", delta_color="inverse")
        st.error(f"주의: 피크 임계치({peak_limit} kWh)를 초과했습니다!")
    else:
        col4.metric("운영 상태", "정상")
        st.success("운영 상태가 안정적입니다.")
    
    # 그래프
    st.line_chart(df)
else:
    st.info("왼쪽 패널에서 설정을 변경한 후 '데이터 분석 실행'을 클릭하세요.")
