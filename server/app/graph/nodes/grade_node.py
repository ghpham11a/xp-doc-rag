
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

from graph.state import State

# Multi Query: Different Perspectives
def grade_node(state: State):

    print("---USING CRAG GRADING NODE---")

    # Initialize LLM
    # llm = query_construction_options.llm
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
    
    # Initialize web search
    web_search_tool = TavilySearchResults(k=3)
    
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

    filtered_docs = []
    web_search_needed = "No"
    grades = []

    documents = state["documents"]
    question = state["question"]

    print(f"---DOCUMENTS FOUND {len(documents)}---")
    
    for doc in documents:
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
            web_search_needed = "Yes"

    if web_search_needed == "Yes":
        # Re-write question for web search
        better_question = question_rewriter.invoke({"question": question})
        # Web search
        docs = web_search_tool.invoke({"query": better_question})
        web_results = [Document(page_content=d["content"], metadata={"source": d["url"]}) for d in docs]
        return {"documents": web_results }
    else:
        print("---DECISION: GENERATE---")
        return {"documents": state["documents"] }
