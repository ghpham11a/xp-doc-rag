from typing import Literal

from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from graph.state import State

class RouteQuery(BaseModel):
    """Route a user query to the most relevant datasource."""

    datasource: Literal["subject_one", "subject_two"] = Field(
        ...,
        description="Given a user question choose which datasource would be most relevant for answering their question",
    )

def logical_routing_node(state: State):
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

    def route_and_invoke(inputs):
        # First, get the routing decision
        route_result = router.invoke({"question": inputs})

        if "subject_one" in route_result.datasource.lower():
            print("---USING SUBJECT ONE---")
            return "subject_one"
        elif "subject_two" in route_result.datasource.lower():
            print("---USING SUBJECT TWO---")
            return "subject_two"
        else:
            print("---USING SUBJECT ONE AS DEFAULT---")
            return "subject_one"

    return {"routed_vector": route_and_invoke(state["question"])}