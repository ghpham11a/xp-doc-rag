from fastapi import APIRouter, File, UploadFile, HTTPException, Path, Request
from fastapi.responses import JSONResponse

from typing import List
import os
import shutil
import uuid

from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma

import state

router = APIRouter(prefix="/files", tags=["files"])

def process_document(file_path: Path, file_type: str):
    if file_type == ".pdf":
        loader = PyPDFLoader(str(file_path))
    elif file_type == ".txt":
        loader = TextLoader(str(file_path))
    elif file_type in [".docx", ".doc"]:
        loader = Docx2txtLoader(str(file_path))
    else:
        raise ValueError(f"Unsupported file type: {file_type}")
    
    documents = loader.load()
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
    )
    
    splits = text_splitter.split_documents(documents)
    return splits

@router.post("/upload")
async def upload_file(request: Request, file: UploadFile = File(...)):

    vector_store = request.app.state.vector_store["dz1"]
    
    try:
        file_extension = os.path.splitext(file.filename)[1].lower()
        if file_extension not in [".pdf", ".txt", ".doc", ".docx"]:
            raise HTTPException(status_code=400, detail=f"Unsupported file type: {file_extension}")
        
        unique_filename = f"{uuid.uuid4()}{file_extension}"
        file_path = request.app.state.UPLOAD_DIR / unique_filename
        
        with file_path.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        documents = process_document(file_path, file_extension)
        
        vector_store.add_documents(documents)

        print("after add documents " + str(vector_store._collection.count()))
        
        return JSONResponse(
            status_code=200,
            content={
                "filename": file.filename,
                "saved_as": unique_filename,
                "size": file_path.stat().st_size,
                "content_type": file.content_type,
                "chunks_processed": len(documents)
            }
        )
    except Exception as e:
        print("error", str(e))
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        file.file.close()

@router.post("/upload-multiple")
async def upload_multiple_files(request: Request, files: List[UploadFile] = File(...)):

    vector_store = request.app.state.vector_store

    uploaded_files = []
    
    for file in files:
        try:
            file_extension = os.path.splitext(file.filename)[1].lower()
            if file_extension not in [".pdf", ".txt", ".doc", ".docx"]:
                uploaded_files.append({
                    "filename": file.filename,
                    "error": f"Unsupported file type: {file_extension}"
                })
                continue
                
            unique_filename = f"{uuid.uuid4()}{file_extension}"
            file_path = request.app.state.UPLOAD_DIR / unique_filename
            
            with file_path.open("wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
            
            documents = process_document(file_path, file_extension)
            
            vector_store.add_documents(documents)
            
            uploaded_files.append({
                "filename": file.filename,
                "saved_as": unique_filename,
                "size": file_path.stat().st_size,
                "content_type": file.content_type,
                "chunks_processed": len(documents)
            })
        except Exception as e:
            uploaded_files.append({
                "filename": file.filename,
                "error": str(e)
            })
        finally:
            file.file.close()
    
    return JSONResponse(
        status_code=200,
        content={"uploaded_files": uploaded_files}
    )

@router.delete("/clear-vector-store")
async def clear_vector_store(request: Request):
    try:
        vector_store = request.app.state.vector_store
        if vector_store:
            # Get all document IDs and delete them
            collection = vector_store._collection
            all_docs = collection.get()
            if all_docs['ids']:
                collection.delete(ids=all_docs['ids'])
            
            return JSONResponse(
                status_code=200,
                content={"message": "Vector store cleared successfully", "documents_deleted": len(all_docs['ids'])}
            )
        else:
            return JSONResponse(
                status_code=404,
                content={"message": "Vector store not initialized"}
            )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error clearing vector store: {str(e)}")

@router.get("/")
async def list_files():
    files = []
    for file_path in state.UPLOAD_DIR.iterdir():
        if file_path.is_file():
            files.append({
                "filename": file_path.name,
                "size": file_path.stat().st_size,
                "created": file_path.stat().st_ctime
            })
    return {"files": files}