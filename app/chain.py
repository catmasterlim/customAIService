from langchain_ollama import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from llm import llm

# 프롬프트 설정
prompt = ChatPromptTemplate.from_template("{topic} 에 대하여 간략히 설명해 줘.")

# LangChain 표현식 언어 체인 구문을 사용합니다.
chain = prompt | llm | StrOutputParser()