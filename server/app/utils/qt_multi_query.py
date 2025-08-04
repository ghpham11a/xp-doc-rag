from operator import itemgetter
from fastapi import Request

from langchain_core.messages import HumanMessage, AIMessage
from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain.load import dumps, loads
from langchain_core.vectorstores.base import VectorStore


from models import ChatRequest, ChatResponse, QueryTranslationResponse

# Multi Query: Different Perspectives
async def run(chat_request: ChatRequest, request: Request, vector_store: VectorStore) -> QueryTranslationResponse:

    retriever = vector_store.as_retriever()

    template = """You are an AI language model assistant. Your task is to generate five 
    different versions of the given user question to retrieve relevant documents from a vector 
    database. By generating multiple perspectives on the user question, your goal is to help
    the user overcome some of the limitations of the distance-based similarity search. 
    Provide these alternative questions separated by newlines. Original question: {question}"""
    prompt_perspectives = ChatPromptTemplate.from_template(template)

    generate_queries = (
        prompt_perspectives 
        | ChatOpenAI(temperature=0) 
        | StrOutputParser() 
        | (lambda x: x.split("\n"))
    )

    def get_unique_union(documents: list[list]):
        """ Unique union of retrieved docs """
        # Flatten list of lists, and convert each Document to string
        flattened_docs = [dumps(doc) for sublist in documents for doc in sublist]
        # Get unique documents
        unique_docs = list(set(flattened_docs))
        # Return
        return [loads(doc) for doc in unique_docs]

    # Retrieve
    retrieval_chain = generate_queries | retriever.map() | get_unique_union

    # RAG
    system_prompt = """Answer the following question based on this context:

    {context}

    Question: {question}
    """

    chain_params = {
        "context": retrieval_chain, 
        "question": itemgetter("question")
    }

    return QueryTranslationResponse(chain_params=chain_params, system_prompt=system_prompt)