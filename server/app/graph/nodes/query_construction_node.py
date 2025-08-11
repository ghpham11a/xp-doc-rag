from langchain_openai import ChatOpenAI

from graph.state import State

def query_construction_node(state: State):
    return {"llm": ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)}