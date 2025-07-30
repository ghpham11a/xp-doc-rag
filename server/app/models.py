from typing import List
from pydantic import BaseModel

class ChatRequest(BaseModel):
    message: str
    technique: str

class ChatResponse(BaseModel):
    answer: str
    sources: List[str]