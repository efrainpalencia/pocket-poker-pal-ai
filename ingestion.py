import os
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader, WebBaseLoader
from langchain_pinecone import PineconeVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings

load_dotenv()

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

vectorstore = PineconeVectorStore(
    index_name=os.getenv("INDEX_NAME"), embedding=embeddings)

file_path = "./assets/docs/2024_Poker_TDA_Rules_PDF_Longform_Vers_1.0_FINAL.pdf"
loader = PyPDFLoader(file_path)

docs = loader.load()
# print(docs[0].metadata)

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=4000, chunk_overlap=200, add_start_index=True
)
all_splits = text_splitter.split_documents(docs)
# ids = vectorstore.add_documents(documents=all_splits)

retriever = vectorstore.as_retriever()
