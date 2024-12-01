import streamlit as st 
from langchain.callbacks import get_openai_callback
from langchain.chat_models import ChatOpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.chains import AnalyzeDocumentChain
from langchain.schema import ChatMessage
from PyPDF2 import PdfReader
import time
import os

os.environ["OPENAI_API_KEY"] = ''


########## qa bot 함수를 그대로 가져와서 활용합니다. 
def qa_bot(source, question, model='gpt-3.5-turbo', temperature=0, chain_type="map_reduce"):
    model = ChatOpenAI(model = model , temperature = 0)
    qa_chain = load_qa_chain(model, chain_type= chain_type)
    qa_document_chain = AnalyzeDocumentChain(combine_docs_chain = qa_chain)
    answer = qa_document_chain.run(input_document=source, question=question)
    return answer


########### 받은 pdf를 raw text로 변환하는 함수를 그대로 가져와서 활용합니다. 
def pdf_to_txt(uploaded_file):
    raw_text =''
    reader = PdfReader(uploaded_file)
    for i, page in enumerate(reader.pages):
        text = page.extract_text()
        if text:
            raw_text += (text + ' ')
    return raw_text
if 'messages' not in st.session_state:
    st.session_state['messages'] =[]   
    #or st.session_state.messages = []

############ 채팅 메시지들을 웹에 띄워줍니다. 
def print_messages():
    # conversation history print 
    for msg in st.session_state.messages:
        st.chat_message(msg['role']).write(msg['content'])



# page title
st.title('문서 기반 요약 및 QA쳇봇')
# file upload
uploaded_file = st.file_uploader('Upload an document', type=['pdf'])
print(uploaded_file)


# file을 session state에 저장
if 'raw' not in st.session_state:
    if uploaded_file:
        raw_text = pdf_to_txt(uploaded_file)
        st.session_state.raw = raw_text
with st.chat_message('assistant'):
    st.write('저는 문서에 대한 이해를 도와주는 챗봇입니다. 무엇을 도와드릴까요?')
prompt = st.chat_input('문서에 대해 궁금한 점이 있으시면 알려주세요')
if prompt:
    st.session_state.messages.append({'role':'user', 'content':prompt})
    response = qa_bot(st.session_state.raw, prompt)
    st.session_state.messages.append({'role':'assistant', 'content':response})
    print_messages()
    
#  chatbot greatings


# chat bot들과의 대화를 messages 라는 이름으로 session state에 저장


# input prompt 받기 


# 프롬프트 입력 시 챗봇 구동


