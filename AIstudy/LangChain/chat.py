import streamlit as st



st.title('챗봇 연습')
st.chat_message('assistant').write('쳇봇입니다. 궁금한 점을 질문해주세요.')
chat = st.chat_input('궁금한 점을 작성해주세요')
if chat:
    st.chat_message('user').write(chat)