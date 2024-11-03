from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import Document
from pydantic import BaseModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain_openai import ChatOpenAI
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain


class ChatMessage(BaseModel):
    is_organisation_message: bool
    content: str

def create_suggestion_chain():
    pass

class Suggestion(BaseModel):
    suggestion: str
    sources: list[Document]
    
# Function to convert chat history to string
def format_chat_history(dict_input: dict) -> dict:
    return {
        "input": "",
        "chat_history": "\n".join([
            f"{'Doctor' if msg.is_organisation_message else 'Patient'}: {msg.content}" 
            for msg in dict_input["chat_history"]
        ])
    }
    
def create_retriever(retriever: BaseRetriever):
    prompt = ChatPromptTemplate.from_messages([
        ("system", """Given the chat history, formulate a search query that will help find 
         relevant information to make a suggestion for the next message or action.
         Focus on key topics and themes from the conversation.{input}
         """),
        ("human", "{chat_history}")
    ])
    
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    
    return create_history_aware_retriever(llm, retriever, prompt)

def create_suggestion_chain():
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are an expert at making suggestions based on medical information.
        You are given a chat history and a list of documents that are relevant to the conversation.
        Make a suggestion for the next message or action based on the chat history and the documents.
        
        {context}
        """),
        ("human", "{chat_history}")
    ])
    
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    
    return create_stuff_documents_chain(llm, prompt)

def create_rag_chain(retriever: BaseRetriever):
    history_aware_retriever = create_retriever(retriever)
    suggestion_chain = create_suggestion_chain()
    
    return RunnableLambda(format_chat_history) | create_retrieval_chain(history_aware_retriever, suggestion_chain)
    

def generate_suggestions(chat_history: list[ChatMessage], retriever: BaseRetriever) -> Suggestion:
    """Generate suggestions based on the chat history and the retriever
    
    Args:
        chat_history: The chat history
        retriever: The retriever to use to get the documents
    Returns:
        A Suggestion object containing the suggestion and the sources
    """
    
    rag_chain = create_rag_chain(retriever)
    
    response = rag_chain.invoke({"chat_history": chat_history})
    
    return {
        "answer": response["answer"],
        "sources": response["context"]
    }
    
    
    
    
