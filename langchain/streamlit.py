#!/usr/bin/env python
# coding: utf-8

# pip install faiss-cpu langchain streamlit streamlit-chat pypdf langchain_ollama langchain-community


import streamlit as st
from streamlit_chat import message
from langchain.embeddings.ollama import OllamaEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain.vectorstores import FAISS
import tempfile
from langchain.document_loaders import PyPDFLoader

import os
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama.llms import OllamaLLM
from langchain_ollama.chat_models import ChatOllama


# sample pdf 
# https://www.yeonsu.go.kr/UpFiles/board_252/%EC%82%AC%ED%9A%8C%EC%A0%81%EA%B8%B0%EC%97%85_%EC%A0%95%EA%B4%80_%EC%98%88%EC%8B%9C.pdf

# source 참조
# https://github.com/gilbutITbook/080413/blob/main/%EC%8B%A4%EC%8A%B5/5%EC%9E%A5/5_5_%EB%8C%80%ED%99%94%ED%98%95_%EC%B1%97%EB%B4%87_%EB%A7%8C%EB%93%A4%EA%B8%B0.py

model_name = "llama3.2-vision"
# llm = OllamaLLM(model=model_name)
llm = ChatOllama(model=model_name)

uploaded_file = st.sidebar.file_uploader("upload", type="pdf")



if uploaded_file :
    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_file_path = tmp_file.name
    
    loader = PyPDFLoader(tmp_file_path)
    data = loader.load()

    embeddings = OllamaEmbeddings(model="llama3.2")
    vectors = FAISS.from_documents(data, embeddings)

    chain = ConversationalRetrievalChain.from_llm(llm = llm, retriever=vectors.as_retriever())

    def conversational_chat(query):  #문맥 유지를 위해 과거 대화 저장 이력에 대한 처리      
        result = chain({"question": query, "chat_history": st.session_state['history']})
        st.session_state['history'].append((query, result["answer"]))        
        return result["answer"]
    
    if 'history' not in st.session_state:
        st.session_state['history'] = []

    if 'generated' not in st.session_state:
        st.session_state['generated'] = ["안녕하세요! " + uploaded_file.name + "에 관해 질문주세요."]

    if 'past' not in st.session_state:
        st.session_state['past'] = ["안녕하세요!"]
        
    #챗봇 이력에 대한 컨테이너
    response_container = st.container()
    #사용자가 입력한 문장에 대한 컨테이너
    container = st.container()

    with container: #대화 내용 저장(기억)
        with st.form(key='Conv_Question', clear_on_submit=True):           
            user_input = st.text_input("Query:", placeholder="PDF파일에 대해 얘기해볼까요? (:", key='input')
            submit_button = st.form_submit_button(label='Send')
            
        if submit_button and user_input:
            output = conversational_chat(user_input)
            
            st.session_state['past'].append(user_input)
            st.session_state['generated'].append(output)

    if st.session_state['generated']:
        with response_container:
            for i in range(len(st.session_state['generated'])):
                message(st.session_state["past"][i], is_user=True, key=str(i) + '_user', avatar_style = "fun-emoji", seed = "Nala")
                message(st.session_state["generated"][i], key=str(i), avatar_style = "bottts", seed = "Fluffy")




