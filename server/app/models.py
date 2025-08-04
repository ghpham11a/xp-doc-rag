from typing import List, Literal, Any
from pydantic import BaseModel

from langchain_core.runnables import RunnablePassthrough

class ChatRequest(BaseModel):
    message: str
    path: List[str]

class ChatResponse(BaseModel):
    answer: str
    sources: List[str]

class QueryTranslationResponse(BaseModel):
    chain_params: Any
    system_prompt: str


class RoutingResponse(BaseModel):
    routed_vector: Any

class QueryConstructionResponse(BaseModel):
    llm: Any
