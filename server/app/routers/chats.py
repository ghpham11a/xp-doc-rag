from typing import Any, List
from fastapi import APIRouter, File, UploadFile, HTTPException, Request

from langchain_core.messages import HumanMessage, AIMessage
from models import ChatRequest, ChatResponse, QueryConstructionResponse
from langchain.prompts import ChatPromptTemplate
from utils import query_translation, rt_logical, rt_semantic, qc_vector, qt_multi_query, qt_rag_fusion, qt_step_back, qt_hyde
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.vectorstores.base import VectorStore


router = APIRouter(prefix="/chats", tags=["chats"])
    
async def handle_query_translation(path: str, chat_request: ChatRequest, request: Request, vector_store: VectorStore):

    if path == "multi-query":
        return await qt_multi_query.run(chat_request, request, vector_store)
    
    if path == "rag-fusion":
        return await qt_rag_fusion.run(chat_request, request, vector_store)

    # decomposition is handled on its own
    
    if path == "step-back":
        return await qt_step_back.run(chat_request, request, vector_store)
    
    if path == "hyde":
        return await qt_hyde.run(chat_request, request, vector_store)
    
    return await run_default_question_answer(chat_request, request)
    
# should return the LLM instance
async def handle_query_construction(path: str, chat_request: ChatRequest, request: Request):

    if path == "vector":
        return await qc_vector.run(chat_request, request)

    return QueryConstructionResponse(llm=ChatOpenAI(temperature=0))

def build_chain(query_translation_options, query_construction_options):
     
    # Map the keys from query_translation output to match prompt template expectations
    def map_to_prompt_format(x):
        return {
            "context": x["context"],
            "input": x["question"],
            "question": x["question"],
            "chat_history": x.get("chat_history", [])
        }
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", query_translation_options.system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}")
    ])
    
    rag_chain = (
        query_translation_options.chain_params 
        | RunnableLambda(map_to_prompt_format)
        | prompt 
        | query_construction_options.llm 
        | StrOutputParser()
    )

    return rag_chain

@router.post("/send", response_model=ChatResponse)
async def chat(chat_request: ChatRequest, request: Request):

    path = chat_request.path

    rag_chain = None

    try:

        # construction of LLM vs structured LLM does not have any dependencies
        query_construction_options = await handle_query_construction(path[2], chat_request, request)

        if path[0] == "decomposition":
            return await query_translation.run_query_translation_decomposition(chat_request, request, query_construction_options)

        if path[1] == "logical":
            sub_one_options = await handle_query_translation(path[0], chat_request, request, request.app.state.vector_stores["subject_one"])
            sub_two_options = await handle_query_translation(path[0], chat_request, request, request.app.state.vector_stores["subject_two"])
            sub_one_options_multi = await handle_query_translation(path[0], chat_request, request, request.app.state.multi_vector_stores["subject_one"])
            sub_two_options_multi = await handle_query_translation(path[0], chat_request, request, request.app.state.multi_vector_stores["subject_two"])
            sub_one_options_raptor = await handle_query_translation(path[0], chat_request, request, request.app.state.raptor_vector_stores["subject_one"])
            sub_two_options_raptor = await handle_query_translation(path[0], chat_request, request, request.app.state.raptor_vector_stores["subject_two"])

            sub_one_chain = build_chain(sub_one_options, query_construction_options)
            sub_two_chain = build_chain(sub_two_options, query_construction_options)
            sub_one_multi_chain = build_chain(sub_one_options_multi, query_construction_options)
            sub_two_multi_chain = build_chain(sub_two_options_multi, query_construction_options)
            sub_one_raptor_chain = build_chain(sub_one_options_raptor, query_construction_options)
            sub_two_raptor_chain = build_chain(sub_two_options_raptor, query_construction_options)

            if path[3] == "multi-representation":
                rag_chain = await rt_logical.run(chat_request, request, {"subject_one": sub_one_multi_chain, "subject_two": sub_two_multi_chain})
            elif path[3] == "raptor":
                rag_chain = await rt_logical.run(chat_request, request, {"subject_one": sub_one_raptor_chain, "subject_two": sub_two_raptor_chain})
            else:
                rag_chain = await rt_logical.run(chat_request, request, {"subject_one": sub_one_chain, "subject_two": sub_two_chain})
        else:
            query_translation_options = await handle_query_translation(path[0], chat_request, request, request.app.state.vector_stores["subject_one"])
            rag_chain = build_chain(query_translation_options, query_construction_options)

        response = rag_chain.invoke({
            "question": chat_request.message
        })

        response = {"answer": response, "sources": []}

        request.app.state.chat_history.append(HumanMessage(content=chat_request.message))
        request.app.state.chat_history.append(AIMessage(content=response['answer']))
        
        # Keep only last 10 messages to prevent context overflow
        if len(request.app.state.chat_history) > 20:
            request.app.state.chat_history = request.app.state.chat_history[-20:]

        return ChatResponse(
            answer=response["answer"],
            sources=response["sources"]
        )
    except Exception as e:
        print(f"Error in chat: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

async def run_default_question_answer(chat_request: ChatRequest, request: Request):
    retrieval_chain = request.app.state.retrieval_chain
    
    try:
        # Invoke the chain with chat history
        response = retrieval_chain.invoke({
            "input": chat_request.message,
            "chat_history": request.app.state.chat_history
        })
        
        # Extract source documents and their metadata
        sources = []
        if "context" in response:
            for doc in response["context"]:
                if hasattr(doc, 'metadata') and 'source' in doc.metadata:
                    sources.append(doc.metadata['source'])
        
        # Remove duplicates while preserving order
        sources = list(dict.fromkeys(sources))
        
        # Update chat history
        request.app.state.chat_history.append(HumanMessage(content=chat_request.message))
        request.app.state.chat_history.append(AIMessage(content=response['answer']))
        
        # Keep only last 10 messages to prevent context overflow
        if len(request.app.state.chat_history) > 20:
            request.app.state.chat_history = request.app.state.chat_history[-20:]
        
        print(f"Answer: {response['answer']}")
        print(f"Sources: {sources}")

        return ChatResponse(
            answer=response['answer'],
            sources=sources
        )
    except Exception as e:
        print(f"Error in chat: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))