from typing import List, Literal
from pydantic import BaseModel

from langchain_core.pydantic_v1 import BaseModel, Field

class ChatRequest(BaseModel):
    message: str
    technique: str

class ChatResponse(BaseModel):
    answer: str
    sources: List[str]