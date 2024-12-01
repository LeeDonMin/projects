import streamlit as st
import pandas as pd
from datetime import date

df = pd.read_csv("avocado.csv")

# 날짜 데이터를 문자열에서 날짜 데이터 형식으로 옮깁니다.
df['date'] = df['date'].apply(lambda x: date.fromisoformat(x))  ## YYYY-MM-DD 형태로 날짜를 표현하는 걸 ISO format 이라고 함

# [TODO] 테이블을 출력해봅니다.
st.dataframe(df)

# [TODO] 테이블을 수정 가능한 테이블로 출력해봅니다.
new_df = st.data_editor(df)

# [TODO] 달력창이 뜨는 수정가능한 테이블 삽입 (시간까지 뜨도록 설정 - DateColumn / DatetimeColumn)
new_df = st.data_editor(df,
                        column_config={
                            "date":st.column_config.DatetimeColumn("Date")
                        })