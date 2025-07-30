from fastapi import APIRouter, File, UploadFile, HTTPException, Request

from langchain_core.messages import HumanMessage, AIMessage
from models import ChatRequest, ChatResponse
from langchain.prompts import ChatPromptTemplate
from utils import rag_methods

router = APIRouter(prefix="/chats", tags=["chats"])

@router.post("/send", response_model=ChatResponse)
async def chat(chat_request: ChatRequest, request: Request):

    if chat_request.technique == "query-translation-(multi-query)":
        return await rag_methods.run_query_translation_multi_query(chat_request, request)
    
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