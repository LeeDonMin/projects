import streamlit as st 
from langchain.callbacks import get_openai_callback
from langchain.chat_models import ChatOpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.chains import AnalyzeDocumentChain
from langchain.schema import ChatMessage
from PyPDF2 import PdfReader
import time
import os

os.environ["OPENAI_API_KEY"] = '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'


########## qa bot 함수를 그대로 가져와서 활용합니다. 
def qa_bot(source, question, model='gpt-3.5-turbo', temperature=0, chain_type="map_reduce"):




########### 받은 pdf를 raw text로 변환하는 함수를 그대로 가져와서 활용합니다. 
def pdf_to_txt(uploaded_file):
    
    
    
    

############ 채팅 메시지들을 웹에 띄워줍니다. 
def print_messages():
    # conversation history print 
    for msg in st.session_state.messages:
        st.chat_message(msg['role']).write(msg['content'])



# page title


# file upload
uploaded_file = st.file_uploader('Upload an document', type=['pdf'])
print(uploaded_file)


# file을 session state에 저장



#  chatbot greatings


# chat bot들과의 대화를 messages 라는 이름으로 session state에 저장


# input prompt 받기 


# 프롬프트 입력 시 챗봇 구동


