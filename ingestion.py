import os

from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_text_splitters import CharacterTextSplitter

load_dotenv()

if __name__ == '__main__':
    print("Ingesting...")
    loader = TextLoader("D:\RAG LangChain\langchain-course\mediumblog1.txt", encoding = 'UTF-8')

    # -----------------

    # loader = TextLoader("D:\RAG LangChain\langchain-course\mediumblog1.txt")
    # if not including encoding gives error, insert encoding, if encoding doesn't work use the flag "autodetect_encoding=True"

    # ------------------
    document = loader.load()

    print("splitting....")
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_documents(document)
    print(f"created {len(texts)} chunks")

    embeddings = OllamaEmbeddings(model="embeddinggemma")

    print("ingesting...")

    PineconeVectorStore.from_documents(texts, embeddings,index_name = os.environ["INDEX_NAME"])
    print("Finish")
