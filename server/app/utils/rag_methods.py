from fastapi import APIRouter, File, UploadFile, HTTPException, Request

from langchain_core.messages import HumanMessage, AIMessage
from models import ChatRequest, ChatResponse
from langchain.prompts import ChatPromptTemplate

# Multi Query: Different Perspectives
async def run_query_translation_multi_query(chat_request: ChatRequest, request: Request):

    retriever = request.app.state.vector_store.as_retriever()

    template = """You are an AI language model assistant. Your task is to generate five 
    different versions of the given user question to retrieve relevant documents from a vector 
    database. By generating multiple perspectives on the user question, your goal is to help
    the user overcome some of the limitations of the distance-based similarity search. 
    Provide these alternative questions separated by newlines. Original question: {question}"""
    prompt_perspectives = ChatPromptTemplate.from_template(template)

    from langchain_core.output_parsers import StrOutputParser
    from langchain_openai import ChatOpenAI

    generate_queries = (
        prompt_perspectives 
        | ChatOpenAI(temperature=0) 
        | StrOutputParser() 
        | (lambda x: x.split("\n"))
    )

    from langchain.load import dumps, loads

    def get_unique_union(documents: list[list]):
        """ Unique union of retrieved docs """
        # Flatten list of lists, and convert each Document to string
        flattened_docs = [dumps(doc) for sublist in documents for doc in sublist]
        # Get unique documents
        unique_docs = list(set(flattened_docs))
        # Return
        return [loads(doc) for doc in unique_docs]

    # Retrieve
    question = chat_request.message
    retrieval_chain = generate_queries | retriever.map() | get_unique_union
    docs = retrieval_chain.invoke({"question":question})
    
    from operator import itemgetter
    from langchain_openai import ChatOpenAI
    from langchain_core.runnables import RunnablePassthrough

    # RAG
    template = """Answer the following question based on this context:

    {context}

    Question: {question}
    """

    prompt = ChatPromptTemplate.from_template(template)

    llm = ChatOpenAI(temperature=0)

    final_rag_chain = (
        {"context": retrieval_chain, 
        "question": itemgetter("question")} 
        | prompt
        | llm
        | StrOutputParser()
    )

    response = final_rag_chain.invoke({"question":question})

    return ChatResponse(
        answer=response,
        sources=[]
    )