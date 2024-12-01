import streamlit as st
import pandas as pd
from datetime import date

df = pd.read_csv("avocado.csv")
df['date'] = df['date'].apply(lambda x: date.fromisoformat(x))
df.drop('type', axis=1, inplace=True)

# Boston, California, Los Angeles 에 대한 line chart를 갖는 그래프
plot_table = {
    "geography": [
        "Boston",
        "California",
        "Los Angeles"
    ]
}

line_chart = []

# [TODO] 선 그래프로 그려질 숫자들을 담아봅니다.
for i in plot_table["geography"]:
    line_chart.append(list(df[df['geography']==i]['average_price']))

plot_table["line"] = line_chart

st.dataframe(plot_table,
             column_config={
                 "line": st.column_config.LineChartColumn(
                     "Average Price",
                     width="medium"
                 )
             })


# [TODO] line chart를 위한 날짜(date)에 따른 pivot table을 만듭니다.  Pivot table column, index가 어떻게 설정되는지 step by step으로 보여주기
pivot_df = pd.pivot_table(df, index='date',columns=["geography"])

# # line chart를 시각화합니다.
st.line_chart(pivot_df["average_price"][plot_table["geography"]]) 


# # bar chart를 시각화합니다.
st.bar_chart(pivot_df["average_price"][plot_table["geography"]]) 

