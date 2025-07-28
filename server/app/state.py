from typing import Optional
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Shared state
vector_store: Optional[Chroma] = None
retrieval_chain = None
chat_history = []

# Shared configuration
UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)
VECTOR_STORE_DIR = Path("chroma_db")
VECTOR_STORE_DIR.mkdir(exist_ok=True)

# Shared embeddings instance
embeddings = OpenAIEmbeddings()