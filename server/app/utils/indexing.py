import uuid

from fastapi import Request, UploadFile
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from langchain.storage import InMemoryByteStore
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.retrievers.multi_vector import MultiVectorRetriever

async def run_indexing_multi_representation(request: Request, file: UploadFile = File(...)):

    docs = [file]
    
    chain = (
        {"doc": lambda x: x.page_content}
        | ChatPromptTemplate.from_template("Summarize the following document:\n\n{doc}")
        | ChatOpenAI(model="gpt-3.5-turbo",max_retries=0)
        | StrOutputParser()
    )

    summaries = chain.batch(docs, {"max_concurrency": 5})

    # The vectorstore to use to index the child chunks
    vectorstore = Chroma(collection_name="summaries",
                        embedding_function=OpenAIEmbeddings())

    # The storage layer for the parent documents
    store = InMemoryByteStore()
    id_key = "doc_id"

    # The retriever
    retriever = MultiVectorRetriever(
        vectorstore=vectorstore,
        byte_store=store,
        id_key=id_key,
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

    query = "Memory in agents"
    sub_docs = vectorstore.similarity_search(query,k=1)
    sub_docs[0]