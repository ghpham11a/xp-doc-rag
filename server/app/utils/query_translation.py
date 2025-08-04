from operator import itemgetter
from typing import Any
from fastapi import APIRouter, File, UploadFile, HTTPException, Request

from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain.load import dumps, loads
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_core.vectorstores.base import VectorStore


from models import ChatRequest, ChatResponse, QueryTranslationResponse

# Multi Query: Different Perspectives
async def run_query_translation_multi_query(chat_request: ChatRequest, request: Request, vector_store: VectorStore) -> QueryTranslationResponse:

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


# RAG-Fusion: Related
async def run_query_translation_rag_fusion(chat_request: ChatRequest, request: Request, vector_store: VectorStore):

    retriever = vector_store.as_retriever()

    template = """You are a helpful assistant that generates multiple search queries based on a single input query. \n
    Generate multiple search queries related to: {question} \n
    Output (4 queries):"""
    prompt_rag_fusion = ChatPromptTemplate.from_template(template)

    generate_queries = (
        prompt_rag_fusion 
        | ChatOpenAI(temperature=0)
        | StrOutputParser() 
        | (lambda x: x.split("\n"))
    )

    def reciprocal_rank_fusion(results: list[list], k=60):
        """ Reciprocal_rank_fusion that takes multiple lists of ranked documents 
            and an optional parameter k used in the RRF formula """
        
        # Initialize a dictionary to hold fused scores for each unique document
        fused_scores = {}

        # Iterate through each list of ranked documents
        for docs in results:
            # Iterate through each document in the list, with its rank (position in the list)
            for rank, doc in enumerate(docs):
                # Convert the document to a string format to use as a key (assumes documents can be serialized to JSON)
                doc_str = dumps(doc)
                # If the document is not yet in the fused_scores dictionary, add it with an initial score of 0
                if doc_str not in fused_scores:
                    fused_scores[doc_str] = 0
                # Retrieve the current score of the document, if any
                previous_score = fused_scores[doc_str]
                # Update the score of the document using the RRF formula: 1 / (rank + k)
                fused_scores[doc_str] += 1 / (rank + k)

        # Sort the documents based on their fused scores in descending order to get the final reranked results
        reranked_results = [
            (loads(doc), score)
            for doc, score in sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
        ]

        # Return the reranked results as a list of tuples, each containing the document and its fused score
        return reranked_results
    
    question = chat_request.message
    
    retrieval_chain_rag_fusion = generate_queries | retriever.map() | reciprocal_rank_fusion
    docs = retrieval_chain_rag_fusion.invoke({"question": question})
    len(docs)

    # RAG
    system_prompt = """Answer the following question based on this context:

    {context}

    Question: {question}
    """

    chain_params = {
        "context": retrieval_chain_rag_fusion, 
        "question": itemgetter("question")
    }

    return QueryTranslationResponse(chain_params=chain_params, system_prompt=system_prompt)

# Decomposition
async def run_query_translation_decomposition(chat_request: ChatRequest, request: Request, vector_store: VectorStore):

    retriever = vector_store.as_retriever()

    template = """You are a helpful assistant that generates multiple sub-questions related to an input question. \n
    The goal is to break down the input into a set of sub-problems / sub-questions that can be answers in isolation. \n
    Generate multiple search queries related to: {question} \n
    Output (3 queries):"""
    prompt_decomposition = ChatPromptTemplate.from_template(template)

    # LLM
    decomposition_llm = ChatOpenAI(temperature=0)

    # Chain
    generate_queries_decomposition = ( prompt_decomposition | decomposition_llm | StrOutputParser() | (lambda x: x.split("\n")))

    # Run
    question = "What are the main components of an LLM-powered autonomous agent system?"
    question = chat_request.message
    questions = generate_queries_decomposition.invoke({"question":question})

    print(questions)

    # Prompt
    template = """Here is the question you need to answer:

    \n --- \n {question} \n --- \n

    Here is any available background question + answer pairs:

    \n --- \n {q_a_pairs} \n --- \n

    Here is additional context relevant to the question: 

    \n --- \n {context} \n --- \n

    Use the above context and any background question + answer pairs to answer the question: \n {question}
    """

    decomposition_prompt = ChatPromptTemplate.from_template(template)

    def format_qa_pair(question, answer):
        """Format Q and A pair"""
        
        formatted_string = ""
        formatted_string += f"Question: {question}\nAnswer: {answer}\n\n"
        return formatted_string.strip()

    # llm
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)

    q_a_pairs = ""
    answer = ""
    for q in questions:
        
        rag_chain = (
        {"context": itemgetter("question") | retriever, 
        "question": itemgetter("question"),
        "q_a_pairs": itemgetter("q_a_pairs")} 
        | decomposition_prompt
        | llm
        | StrOutputParser())

        answer = rag_chain.invoke({"question":q,"q_a_pairs":q_a_pairs})
        q_a_pair = format_qa_pair(q,answer)
        q_a_pairs = q_a_pairs + "\n---\n"+  q_a_pair

    print("answer", answer)

    return ChatResponse(
        answer=answer,
        sources=[]
    )

# Step back
async def run_query_translation_step_back(chat_request: ChatRequest, request: Request, vector_store: VectorStore):

    from langchain_core.prompts import FewShotChatMessagePromptTemplate

    retriever = request.app.state.vector_store.as_retriever()

    examples = [
        {
            "input": "Could the members of The Police perform lawful arrests?",
            "output": "what can the members of The Police do?",
        },
        {
            "input": "Jan Sindel’s was born in what country?",
            "output": "what is Jan Sindel’s personal history?",
        },
    ]
    # We now transform these to example messages
    example_prompt = ChatPromptTemplate.from_messages(
        [
            ("human", "{input}"),
            ("ai", "{output}"),
        ]
    )
    few_shot_prompt = FewShotChatMessagePromptTemplate(
        example_prompt=example_prompt,
        examples=examples,
    )
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """You are an expert at world knowledge. Your task is to step back and paraphrase a question to a more generic step-back question, which is easier to answer. Here are a few examples:""",
            ),
            # Few shot examples
            few_shot_prompt,
            # New question
            ("user", "{question}"),
        ]
    )

    generate_queries_step_back = prompt | ChatOpenAI(temperature=0) | StrOutputParser()
    question = chat_request.message
    generate_queries_step_back.invoke({"question": question})

    system_prompt = """You are an expert of world knowledge. I am going to ask you a question. Your response should be comprehensive and not contradicted with the following context if they are relevant. Otherwise, ignore them if they are not relevant.

    # {normal_context}
    # {step_back_context}

    # Original Question: {question}
    # Answer:"""

    chain_params = {
        # Retrieve context using the normal question
        "normal_context": RunnableLambda(lambda x: x["question"]) | retriever,
        # Retrieve context using the step-back question
        "step_back_context": generate_queries_step_back | retriever,
        # Pass on the question
        "question": lambda x: x["question"],
    }

    return QueryTranslationResponse(chain_params=chain_params, system_prompt=system_prompt)

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