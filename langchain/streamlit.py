#!/usr/bin/env python
# coding: utf-8

# pip install faiss-cpu langchain streamlit streamlit-chat pypdf langchain_ollama
# pip install pdfplumber langchain-text-splitters

import streamlit as st
from streamlit_chat import message
from streamlit.logger import get_logger
from langchain.embeddings.ollama import OllamaEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain.vectorstores import FAISS, VectorStore
import tempfile
#from langchain.document_loaders import PyPDFLoader
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import PDFPlumberLoader
from langchain_community.document_loaders import UnstructuredExcelLoader
import logging
# from logging import getLogger

from langchain_text_splitters import RecursiveCharacterTextSplitter

import os
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama.llms import OllamaLLM
from langchain_ollama.chat_models import ChatOllama


# 환경 변수 
from dotenv import load_dotenv
load_dotenv(override=True)


from langchain_community.retrievers import TavilySearchAPIRetriever
from langchain_community.tools.tavily_search import TavilySearchResults

import os
import random
def get_tavily_api_key() -> str:
    random_number = random.randint(1, 7)
    # random_number = 7
    print(f"--> TAVILY_API_KEY_{random_number}")
    return os.getenv(f"TAVILY_API_KEY_{random_number}")

def get_tavily_retriever() -> TavilySearchAPIRetriever:
    os.environ["TAVILY_API_KEY"] = get_tavily_api_key()    
    return TavilySearchAPIRetriever(k=2, api_key=get_tavily_api_key())

def get_tavily_search_results() -> TavilySearchResults:    
    os.environ["TAVILY_API_KEY"] = get_tavily_api_key()
    return TavilySearchResults(max_results=2)

app_logger = get_logger(__name__)
# app_logger.addHandler(logging.StreamHandler())
# app_logger.setLevel(logging.INFO)

# sample pdf 
# https://www.yeonsu.go.kr/UpFiles/board_252/%EC%82%AC%ED%9A%8C%EC%A0%81%EA%B8%B0%EC%97%85_%EC%A0%95%EA%B4%80_%EC%98%88%EC%8B%9C.pdf

# source 참조
# https://github.com/gilbutITbook/080413/blob/main/%EC%8B%A4%EC%8A%B5/5%EC%9E%A5/5_5_%EB%8C%80%ED%99%94%ED%98%95_%EC%B1%97%EB%B4%87_%EB%A7%8C%EB%93%A4%EA%B8%B0.py

# model_name = "llama3.3:latest"
model_name = "llama3.2:latest"
#model_name = "hf.co/MLP-KTLim/llama-3-Korean-Bllossom-8B-gguf-Q4_K_M:latest"
# llm = OllamaLLM(model=model_name)
llm = ChatOllama(model=model_name, temperature=0)
app_logger.info(f" llm : {llm} ")


chain = None
#
st.title(f"Pdf 파일 서비스(model:{model_name})") 


def make_file_vector(tmp_file_path : str) -> VectorStore:
    loader = PDFPlumberLoader(tmp_file_path)
    pages = loader.load()
    
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=10
    )
    
    data = splitter.split_documents(pages)
    
    embeddings = OllamaEmbeddings(model=model_name)
    vectors = FAISS.from_documents(data, embeddings)
    
    return vectors

uploaded_file = st.sidebar.file_uploader("upload", type="pdf", accept_multiple_files=False)
if uploaded_file :
    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_file_path = tmp_file.name
        file_name = uploaded_file.name
        app_logger.info(f" uploaded_file : {uploaded_file.name}")
        app_logger.info(f" make_file_vector : {tmp_file_path}")
        pdf_vector = make_file_vector(tmp_file_path=tmp_file_path)

        from langchain.chains.combine_documents import create_stuff_documents_chain
        from langchain.chains import create_retrieval_chain
        from langchain import hub
        system_prompt = """
        - Answer is always in Korean
        - Answer any use questions based solely on the context below:

        <context>
        {context}
        </context>
        """
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt),
                ("human" , "{input}")
            ]
        )
        combine_docs_chain = create_stuff_documents_chain(
            llm, prompt
        )
        chain = create_retrieval_chain(pdf_vector.as_retriever(), combine_docs_chain)

        # chain = ConversationalRetrievalChain.from_llm(llm = llm, retriever=pdf_vector.as_retriever())
        
        # ai_message = st.chat_message("ai")
        # user_message = st.chat_message("user")
        
        # ai_message.write("안녕하세요! '" + uploaded_file.name + "'에 관해 질문주세요.(model:" + model_name+")")
        
    
if chain:
    if 'history' not in st.session_state:
        st.session_state['history'] = []

    if 'generated' not in st.session_state:
        init_message = f"{uploaded_file.name}에 관해 질문주세요."
        st.session_state['generated'] = [init_message]
        

    if 'past' not in st.session_state:
        st.session_state['past'] = ["안녕하세요!"]
        
    def conversational_chat(query):  #문맥 유지를 위해 과거 대화 저장 이력에 대한 처리 
        app_logger.info(f" st.session_state['history'] : {st.session_state['history']}")
        
        # history = [("PDF 파일 이름" , file_name)]    
        history = []   
        if st.session_state['history'] :
            history.append(st.session_state['history'][-1])  
 
        
        app_logger.info(f" conversational_chat - query : {query}, history : {history}") 
        result = chain.invoke({"input": query, "chat_history": history })
        st.session_state['history'].append((query, result["answer"]))        
        return result["answer"]
        
    #챗봇 이력에 대한 컨테이너
    response_container = st.container()
    #사용자가 입력한 문장에 대한 컨테이너
    container = st.container()


    with container: #대화 내용 저장(기억)
        with st.form(key='Conv_Question', clear_on_submit=True) as f:   
            
            user_input = st.text_area("파일에 대해 얘기해볼까요?")
            submit_button = st.form_submit_button(label='Send')
                
            if submit_button and user_input:
                
                app_logger.info(f" user_input : {user_input}")
                st.session_state['past'].append(user_input)
              

    if st.session_state['generated']:
        with response_container:
            for i in range(len(st.session_state['generated'])):
                message(st.session_state["past"][i], is_user=True, key=str(i) + '_user', avatar_style = "fun-emoji", seed = "Nala")
                message(st.session_state["generated"][i], key=str(i), avatar_style = "bottts", seed = "Fluffy")

            if len(st.session_state['past']) > len(st.session_state['generated']) :
                i = len(st.session_state['past']) - 1
                message(st.session_state["past"][i], is_user=True, key=str(i) + '_user', avatar_style = "fun-emoji", seed = "Nala")
                user_input = st.session_state['past'][-1]
                output = conversational_chat(user_input)
                st.session_state['generated'].append(output)
                message(st.session_state["generated"][i], key=str(i), avatar_style = "bottts", seed = "Fluffy")
