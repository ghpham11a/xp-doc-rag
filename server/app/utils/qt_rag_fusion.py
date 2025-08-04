from operator import itemgetter
from fastapi import Request

from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain.load import dumps, loads
from langchain_core.vectorstores.base import VectorStore


from models import ChatRequest, ChatResponse, QueryTranslationResponse

# RAG-Fusion: Related
async def run(chat_request: ChatRequest, request: Request, vector_store: VectorStore):

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

