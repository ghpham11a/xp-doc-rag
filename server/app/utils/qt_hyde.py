from fastapi import Request

from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.vectorstores.base import VectorStore


from models import ChatRequest, ChatResponse, QueryTranslationResponse

# HyDE
async def run_query_translation_hyde(chat_request: ChatRequest, request: Request, vector_store: VectorStore):

    retriever = request.app.state.vector_store.as_retriever()

    # HyDE document generation
    template = """Please write a scientific paper passage to answer the question
    Question: {question}
    Passage:"""
    prompt_hyde = ChatPromptTemplate.from_template(template)

    generate_docs_for_retrieval = (
        prompt_hyde | ChatOpenAI(temperature=0) | StrOutputParser() 
    )

    question = chat_request.message

    # Run
    question = "What is task decomposition for LLM agents?"
    generate_docs_for_retrieval.invoke({"question": question})

    # Retrieve
    retrieval_chain = generate_docs_for_retrieval | retriever 
    retrieved_docs = retrieval_chain.invoke({"question":question})
    
    # RAG
    template = """Answer the following question based on this context:

    {context}

    Question: {question}
    """

    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
    prompt = ChatPromptTemplate.from_template(template)

    final_rag_chain = (
        prompt
        | llm
        | StrOutputParser()
    )

    answer = final_rag_chain.invoke({"context": retrieved_docs,"question": question})

    return ChatResponse(
        answer=answer,
        sources=[]
    )