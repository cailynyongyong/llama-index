# pip install beautifulsoup4
# pip install faiss-cpu
from dotenv import load_dotenv
load_dotenv()
import openai
import os
openai.api_key= os.environ.get("OPENAI_API_KEY")

from langchain_community.llms import Ollama

llm = Ollama(model="llama3")

# 1) 데이터 로딩하기
from langchain_community.document_loaders import WebBaseLoader
loader = WebBaseLoader("https://ko.wikipedia.org/wiki/%EC%82%AC%EA%B3%BC_%EC%A3%BC%EC%8A%A4")

docs = loader.load()

# 2) 텍스트 토큰화하기
from langchain_openai import OpenAIEmbeddings

embeddings = OpenAIEmbeddings()

# 3) 데이터 쪼개기
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter()
documents = text_splitter.split_documents(docs)

# 4) 데이터베이스에 업로드하기
vector = FAISS.from_documents(documents, embeddings)

# 5) 데이터에 대해 질문과 답변할 수 있는 프롬프트 및 모델 만들기
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

prompt = ChatPromptTemplate.from_template("""Answer the following question based only on the provided context:

<context>
{context}
</context>

Question: {input}""")

document_chain = create_stuff_documents_chain(llm, prompt)

# 6) 업로드한 데이터를 불러올 수 있는 retriever 체인 만들기
from langchain.chains import create_retrieval_chain

retriever = vector.as_retriever()
retrieval_chain = create_retrieval_chain(retriever, document_chain)

# 7) 답변받기
response = retrieval_chain.invoke({"input": "사과주스는 건겅에 어떻게 좋은가요?"})
print(response["answer"])