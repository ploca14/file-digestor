from langchain_core.vectorstores import VectorStore
from langchain_core.embeddings import Embeddings
from langchain_core.retrievers import BaseRetriever
from langchain_openai import OpenAIEmbeddings
from langchain_iris import IRISVector
from config import COLLECTION_NAME, CONNECTION_STRING

def get_embeddings() -> Embeddings:
    return OpenAIEmbeddings()

def get_vector_store(embeddings: Embeddings) -> VectorStore:
    return IRISVector(
        embedding_function=embeddings,
        dimension=1536,
        collection_name=COLLECTION_NAME,
        connection_string=CONNECTION_STRING,
    )
    
def get_retriever(vector_store: VectorStore, patient_id: str) -> BaseRetriever:
    return vector_store.as_retriever(
        search_kwargs={"filter": {"patient_id": patient_id}}
    )