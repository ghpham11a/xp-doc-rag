from typing import Literal, Any
from fastapi import Request

from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnableLambda
from langchain_openai import ChatOpenAI


from models import ChatRequest, ChatResponse, RoutingResponse

# Data model
class RouteQuery(BaseModel):
    """Route a user query to the most relevant datasource."""

    datasource: Literal["subject_one", "subject_two"] = Field(
        ...,
        description="Given a user question choose which datasource would be most relevant for answering their question",
    )

# Logical
async def run(chat_request: ChatRequest, request: Request, chains: Any):
    # LLM with function call 
    llm = ChatOpenAI(model="gpt-3.5-turbo-0125", temperature=0)
    structured_llm = llm.with_structured_output(RouteQuery)

    # Prompt 
    system = """You are an expert at routing a user question to the appropriate data source.

    Based on the subject number the question is referring to, route it to the relevant data source."""

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system),
            ("human", "{question}"),
        ]
    )

    # Define router 
    router = prompt | structured_llm

    def choose_route(result):
        
        # Then select and invoke the appropriate chain
        if "subject_one" in result.datasource.lower():
            selected_chain = chains["subject_one"]
        elif "subject_two" in result.datasource.lower():
            selected_chain = chains["subject_two"]
        else:
            selected_chain = chains["subject_one"]

    def route_and_invoke(inputs):
        # First, get the routing decision
        route_result = router.invoke({"question": inputs["question"]})
        
        # Then select and invoke the appropriate chain
        if "subject_one" in route_result.datasource.lower():
            selected_chain = chains["subject_one"]
        elif "subject_two" in route_result.datasource.lower():
            selected_chain = chains["subject_two"]
        else:
            selected_chain = chains["subject_one"]
        
        # Invoke the selected chain with the question
        return selected_chain.invoke({"question": inputs["question"]})

    return RunnableLambda(route_and_invoke)