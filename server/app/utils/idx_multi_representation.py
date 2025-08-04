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
        
        # Split documents into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
        )
        
        docs = text_splitter.split_documents(documents)
        
        chain = (
            {"doc": lambda x: x.page_content}
            | ChatPromptTemplate.from_template("Summarize the following document:\n\n{doc}")
            | ChatOpenAI(model="gpt-3.5-turbo",max_retries=0)
            | StrOutputParser()
        )

        summaries = chain.batch(docs, {"max_concurrency": 5})

        # The storage layer for the parent documents
        store = InMemoryByteStore()
        id_key = "doc_id"

        # The retriever
        retriever = MultiVectorRetriever(
            vectorstore=request.app.state.multi_vector_stores[zone],
            byte_store=InMemoryByteStore(),
            id_key="doc_id",
        )

        doc_ids = [str(uuid.uuid4()) for _ in docs]

        # Docs linked to summaries
        summary_docs = [
            Document(page_content=s, metadata={id_key: doc_ids[i]})
            for i, s in enumerate(summaries)
        ]

        # Add
        retriever.vectorstore.add_documents(summary_docs)
        retriever.docstore.mset(list(zip(doc_ids, docs)))
        
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

    ########################################

    print(f"run_indexing_multi_representation {zone}")

    

    # query = "Memory in agents"
    # sub_docs = vectorstore.similarity_search(query,k=1)
    # sub_docs[0]

async def run_indexing_raptor(request: Request, file: UploadFile = File(...), zone: str = ""):
    pass