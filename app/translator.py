from langchain_community.chat_models import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from llm import llm

# 프롬프트 설정
prompt = ChatPromptTemplate.from_template(
    "Translate following sentences into Korean:\n{input}"
)

# LangChain 표현식 언어 체인 구문을 사용합니다.
chain = prompt | llm | StrOutputParser()