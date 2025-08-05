import uuid
import os
import shutil

from fastapi import Request, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.storage import InMemoryByteStore

from langchain.retrievers.multi_vector import MultiVectorRetriever
from ragatouille import RAGPretrainedModel

async def run(request: Request, file: UploadFile = File(...), zone: str = ""):

    try:
        # Save uploaded file temporarily
        file_extension = os.path.splitext(file.filename)[1].lower()
        unique_filename = f"{uuid.uuid4()}{file_extension}"
        file_path = request.app.state.UPLOAD_DIR / unique_filename
        
        with file_path.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Load and process document
        if file_extension == ".pdf":
            loader = PyPDFLoader(str(file_path))
        elif file_extension == ".txt":
            loader = TextLoader(str(file_path))
        elif file_extension in [".docx", ".doc"]:
            loader = Docx2txtLoader(str(file_path))
        else:
            raise ValueError(f"Unsupported file type: {file_extension}")
        
        documents = loader.load()

        RAG = RAGPretrainedModel.from_pretrained("colbert-ir/colbertv2.0")

        RAG.index(
            collection=documents,
            index_name=f"{zone}_colbert",
            max_document_length=180,
            split_documents=True,
        )
        
        # results = RAG.search(query="What animation studio did Miyazaki found?", k=3)

        # retriever = RAG.as_langchain_retriever(k=3)
        # retriever.invoke("What animation studio did Miyazaki found?")

    except Exception as e:
        print("error", str(e))
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        file.file.close()
