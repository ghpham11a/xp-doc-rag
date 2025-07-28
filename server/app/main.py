from fastapi import FastAPI
from contextlib import asynccontextmanager
from fastapi.middleware.cors import CORSMiddleware
from routers import files
from routers import chats
from pathlib import Path
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

from conversation_chain import init_conversation_chain
from vector_store import init_vector_store, close_vector_store

load_dotenv()

@asynccontextmanager
async def lifespan(app: FastAPI):

    # Shared configuration
    app.state.UPLOAD_DIR = Path("uploads")
    app.state.UPLOAD_DIR.mkdir(exist_ok=True)
    app.state.VECTOR_STORE_DIR = Path("chroma_db")
    app.state.VECTOR_STORE_DIR.mkdir(exist_ok=True)
    app.state.chat_history = []

    # Embeddings
    app.state.embeddings = OpenAIEmbeddings()

    init_vector_store(app)
    init_conversation_chain(app)

    print("Startup")
    yield

    close_vector_store(app)
    print("Shutdown")
    # await init_db()
    # yield
    # await close_db()

def create_app() -> FastAPI:
    """
    Creates and configures the FastAPI application.
    """
    app = FastAPI(lifespan=lifespan)

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:3000"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Register our router(s)
    app.include_router(files.router)
    app.include_router(chats.router)

    @app.get("/")
    def root():
        return {"message": "up"}

    return app

app = create_app()