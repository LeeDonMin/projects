import streamlit as st
import pandas as pd
from datetime import date

df = pd.read_csv("avocado.csv")

# 날짜 데이터를 문자열에서 날짜 데이터 형식으로 옮깁니다.
df['date'] = df['date'].apply(lambda x: date.fromisoformat(x))  ## YYYY-MM-DD 형태로 날짜를 표현하는 걸 ISO format 이라고 함

# [TODO] 
plot_table = {
    'geography': [
        "Boston",
        "California",
        "Los Angeles"
    ]
}

line_chart = []
for i in plot_table['geography']:
    line_chart.append(list(df[df['geography']==i]['average_price']))
    
plot_table['line'] = line_chart

st.data_editor(plot_table,
               column_config={
                   'line':st.column_config.LineChartColumn(
                       'Average Price',
                       width='medium'
                   )
               })