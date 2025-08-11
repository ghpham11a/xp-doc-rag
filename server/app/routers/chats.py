import json
from typing import Annotated, Dict, Literal
from typing_extensions import TypedDict

from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse
from langchain_core.messages import AIMessage

from models import ChatRequest
from graph.build import build_workflow

router = APIRouter(prefix="/chats", tags=["chats"])

@router.post("/send-stream")
async def chat_stream(chat_request: ChatRequest, request: Request):
    workflow = await build_workflow(chat_request, request)

    async def event_generator():
        try:
            initial_message_count = 1  # We start with 1 user message
            last_sent_content = None
            
            # Use values mode to get complete state updates
            async for event in workflow.astream({"messages": [{"role": "user", "content": chat_request.message}]}, stream_mode="values"):
                # Check if this event contains messages
                if isinstance(event, dict) and "messages" in event:
                    messages = event["messages"]
                    
                    # Check if we have more messages than we started with (i.e., AI responded)
                    if len(messages) > initial_message_count:
                        # Get the last message (should be AI response)
                        last_message = messages[-1]
                        
                        # Verify it's an AIMessage
                        if isinstance(last_message, AIMessage):
                            content = last_message.content
                            # Only send if we haven't sent this content already
                            if content and content != last_sent_content:
                                last_sent_content = content
                                yield f"data: {json.dumps({'type': 'content', 'content': content})}\n\n"
                                
            yield f"data: {json.dumps({'type': 'done'})}\n\n"
        except Exception as e:
            print(f"Streaming error: {str(e)}")
            yield f"data: {json.dumps({'type': 'error', 'error': str(e)})}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")