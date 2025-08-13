from typing import Annotated, Dict, Any, List
from typing_extensions import TypedDict

from langgraph.graph.message import add_messages
from langchain.schema import Document
from langchain_core.vectorstores.base import VectorStore

class State(TypedDict):
    selectedRoutingType: str
    selectedQueryTranslation: str
    selectedQueryConstruction: str
    selectedIndexing: str
    selectedRetrieval: str
    selectedGeneration: str

    question: str
    generation: str = ""
    web_search: str = "No"
    documents: List[Document] = []
    q_a_pairs: str
    web_search_used: bool = False
    document_grades: List[Dict[str, Any]] = []
    messages: Annotated[list, add_messages]
    vector_store: VectorStore
    query_translation_type: str
    llm: Any

    routed_vector: str

    class Config:
        arbitrary_types_allowed = True