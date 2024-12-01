import streamlit as st
import random

st.title(':game_die: 숫자 추측 게임 :game_die:')

# 컴퓨터가 무작위로 1에서 100까지의 숫자를 생성
if 'random_number' not in st.session_state:
    st.session_state.random_number = random.randint(1, 100)

# 사용자가 추측한 숫자를 입력할 수 있는 곳
user_guess = st.number_input('1에서 100 사이의 숫자를 입력하세요:', min_value=1, max_value=100, step=1)

# 버튼을 눌러 숫자를 제출
guess_button = st.button('추측하기')

# 피드백을 제공하는 기능
if guess_button:
    if user_guess < st.session_state.random_number:
        st.write('너무 낮습니다! 다시 시도하세요.')
    elif user_guess > st.session_state.random_number:
        st.write('너무 높습니다! 다시 시도하세요.')
    else:
        st.success(f'축하합니다! 정답은 {user_guess}입니다!')
        # 게임을 재시작할 수 있도록 랜덤 숫자 재생성
        st.session_state.random_number = random.randint(1, 100)

# 게임을 재시작할 수 있는 버튼
if st.button('게임 재시작'):
    st.session_state.random_number = random.randint(1, 100)
    st.write('게임이 재시작되었습니다!')

