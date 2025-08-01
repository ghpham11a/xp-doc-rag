from fastapi import FastAPI
from langchain_chroma import Chroma

def init_vector_store(app: FastAPI):
    # Initialize vector_store attribute if it doesn't exist
    if not hasattr(app.state, 'vector_store'):
        app.state.vector_store = None
    
    if app.state.vector_store is None and app.state.VECTOR_STORE_DIR.exists():
        try:
            app.state.vector_store = {
                "dz1": Chroma(
                    persist_directory=str(app.state.VECTOR_STORE_DIR),
                    embedding_function=app.state.embeddings,
                    collection_name="dz1"
                ),
                "dz2": Chroma(
                    persist_directory=str(app.state.VECTOR_STORE_DIR),
                    embedding_function=app.state.embeddings,
                    collection_name="dz2"
                ),
                "dz3": Chroma(
                    persist_directory=str(app.state.VECTOR_STORE_DIR),
                    embedding_function=app.state.embeddings,
                    collection_name="dz3"
                ),
            }
        except Exception as e:
            print(f"Error loading vector store: {e}")

def close_vector_store(app: FastAPI):
    if hasattr(app.state, 'vector_store'):
        app.state.vector_store = None