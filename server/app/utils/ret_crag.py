
from pydantic import BaseModel, Field
from typing import List, Dict, Any

# LangChain and LangGraph imports
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain.schema import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel as LangChainBaseModel, Field as LangChainField
from langchain_core.output_parsers import StrOutputParser
from langgraph.graph import StateGraph, START, END
from fastapi import Request

from models import ChatRequest, ChatResponse, QueryTranslationResponse

# Pydantic models for API
class QueryRequest(BaseModel):
    question: str = Field(..., description="The user's question")
    max_docs: int = Field(default=4, description="Maximum number of documents to retrieve")
    web_search_threshold: float = Field(default=0.7, description="Threshold for triggering web search")

class QueryResponse(BaseModel):
    answer: str = Field(..., description="The generated answer")
    sources: List[str] = Field(default=[], description="Sources used for the answer")
    web_search_used: bool = Field(default=False, description="Whether web search was triggered")
    document_grades: List[Dict[str, Any]] = Field(default=[], description="Document relevance grades")

# LangChain Pydantic models for structured outputs
class GradeDocuments(LangChainBaseModel):
    """Binary score for relevance check on retrieved documents."""
    binary_score: str = LangChainField(
        description="Documents are relevant to the question, 'yes' or 'no'"
    )

class GraphState(LangChainBaseModel):
    """Represents the state of our graph."""
    question: str
    generation: str = ""
    web_search: str = "No"
    documents: List[Document] = []
    web_search_used: bool = False
    document_grades: List[Dict[str, Any]] = []
    
    class Config:
        arbitrary_types_allowed = True

async def build_crag_workflow(chat_request: ChatRequest, request: Request, rag_chain: Any):
    """Initialize all CRAG components"""

    # Global variables for storing initialized components
    workflow_state = {
        "vectorstore": None,
        "retriever": None,
        "graph": None,
        "llm": None,
        "web_search": None
    }
    
    # Initialize LLM
    # llm = query_construction_options.llm
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
    
    # Get the retriever from the existing vector store
    # Using subject_one vector store as the default for CRAG
    vector_store = request.app.state.vector_stores.get("subject_one")
    if vector_store:
        retriever = vector_store.as_retriever(search_kwargs={"k": 4})
    else:
        # Fallback: create a dummy retriever that returns empty documents
        retriever = None
    
    # Initialize web search
    web_search = TavilySearchResults(k=3)
    
    # Build the CRAG graph
    workflow = StateGraph(GraphState)
    
    # Define the retrieval grader
    grade_prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a grader assessing relevance of a retrieved document to a user question.
         Give a binary score 'yes' or 'no' to indicate whether the document is relevant to the question."""),
        ("human", "Retrieved document: \\n\\n {document} \\n\\n User question: {question}")
    ])
    
    structured_llm_grader = llm.with_structured_output(GradeDocuments)
    retrieval_grader = grade_prompt | structured_llm_grader
    
    # Query transformation prompt
    rewrite_prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a question re-writer that converts an input question to a better version 
         that is optimized for web search. Look at the input and try to reason about the underlying semantic intent."""),
        ("human", "Here is the initial question: \\n\\n {question} \\n Formulate an improved question.")
    ])
    
    question_rewriter = rewrite_prompt | llm | StrOutputParser()
    
    def retrieve(state):
        """Retrieve documents"""
        print(f"---RETRIEVE {state.question}---")
        question = state.question
        if retriever:
            documents = retriever.invoke(question)
        else:
            # If no retriever available, return empty documents list
            documents = []
        return {"documents": documents, "question": question}
    
    def grade_documents(state):
        """Grade document relevance"""
        print("---CHECK DOCUMENT RELEVANCE TO QUESTION---")
        question = state.question
        documents = state.documents
        
        filtered_docs = []
        web_search = "No"
        grades = []

        print(f"---DOCUMENTS FOUND {len(documents)}---")
        
        for doc in documents:
            print(doc.page_content)
            score = retrieval_grader.invoke({
                "question": question, 
                "document": doc.page_content
            })
            grade = score.binary_score
            grades.append({"document": doc.page_content[:100] + "...", "grade": grade})
            
            if grade.lower() == "yes":
                print("---GRADE: DOCUMENT RELEVANT---")
                filtered_docs.append(doc)
            else:
                print("---GRADE: DOCUMENT NOT RELEVANT---")
                web_search = "Yes"
        
        return {
            "documents": filtered_docs,
            "question": question,
            "web_search": web_search,
            "document_grades": grades
        }
    
    def web_search_node(state):
        """Web search based on re-written question"""
        print("---WEB SEARCH---")
        question = state.question
        
        # Re-write question for web search
        better_question = question_rewriter.invoke({"question": question})
        
        # Web search
        docs = web_search.invoke({"query": better_question})
        web_results = [Document(page_content=d["content"], metadata={"source": d["url"]}) for d in docs]
        
        return {
            "documents": web_results,
            "question": question,
            "web_search_used": True
        }
    
    def generate(state):
        """Generate answer"""
        print("---GENERATE---")
        question = state.question
        documents = state.documents
        
        # Format documents
        context = "\\n\\n".join(doc.page_content for doc in documents)
        
        # RAG generation
        generation = rag_chain.invoke({"context": context, "question": question})
        
        # Extract sources
        sources = list(set([doc.metadata.get("source", "Unknown") for doc in documents]))
        
        return {
            "documents": documents,
            "question": question,
            "generation": generation,
            "sources": sources
        }
    
    def decide_to_generate(state):
        """Decide whether to generate an answer or do web search"""
        print("---ASSESS GRADED DOCUMENTS---")
        web_search = state.web_search
        
        if web_search == "Yes":
            print("---DECISION: ALL DOCUMENTS ARE NOT RELEVANT, DO WEB SEARCH---")
            return "web_search"
        else:
            print("---DECISION: GENERATE---")
            return "generate"
    
    # Add nodes to the graph
    workflow.add_node("retrieve", retrieve)
    workflow.add_node("grade_documents", grade_documents)
    workflow.add_node("generate", generate)
    workflow.add_node("web_search", web_search_node)
    
    # Build graph
    workflow.add_edge(START, "retrieve")
    workflow.add_edge("retrieve", "grade_documents")
    workflow.add_conditional_edges(
        "grade_documents",
        decide_to_generate,
        {
            "web_search": "web_search",
            "generate": "generate",
        },
    )
    workflow.add_edge("web_search", "generate")
    workflow.add_edge("generate", END)
    
    graph = workflow.compile()
    
    # Store in global state
    workflow_state["vectorstore"] = vector_store
    workflow_state["retriever"] = retriever
    workflow_state["graph"] = graph
    workflow_state["llm"] = llm
    workflow_state["web_search"] = web_search

    request.app.state.workflow_state = workflow_state

    return graph