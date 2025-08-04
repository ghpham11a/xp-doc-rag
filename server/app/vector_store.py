from fastapi import FastAPI
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain.retrievers.multi_vector import MultiVectorRetriever
from langchain.storage import InMemoryByteStore

def init_vector_store(app: FastAPI):
    # Initialize vector_store attribute if it doesn't exist
    if not hasattr(app.state, 'vector_store'):
        app.state.vector_store = None
    
    # if "vector_stores" not in app.state or app.state.vector_stores is None:
    try:
        app.state.vector_stores = {
            "subject_one": Chroma(
                persist_directory=str(app.state.SUBJECT_ONE_VECTOR_DIR),
                embedding_function=app.state.embeddings,
                collection_name="subject_one_vector"
            ),
            "subject_two": Chroma(
                persist_directory=str(app.state.SUBJECT_TWO_VECTOR_DIR),
                embedding_function=app.state.embeddings,
                collection_name="subject_two_vector"
            )
        }
        app.state.multi_vector_stores = {
            "subject_one": Chroma(
                persist_directory=str(app.state.SUBJECT_ONE_MULTI_VECTOR_DIR),
                collection_name="subject_one_multi_vector", 
                embedding_function=OpenAIEmbeddings()
            ),
            "subject_two": Chroma(
                persist_directory=str(app.state.SUBJECT_TWO_MULTI_VECTOR_DIR),
                collection_name="subject_two_multi_vector", 
                embedding_function=OpenAIEmbeddings()
            ),
        }
    except Exception as e:
        print(f"Error loading vector store: {e}")

def close_vector_store(app: FastAPI):
    if hasattr(app.state, 'vector_store'):
        app.state.vector_store = None