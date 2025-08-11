from langchain_core.messages import AIMessage
from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser

from graph.state import State
from utils.build_context import build_context

def multi_query_generate_node(state: State):

    print("---USING MULTI QUERY GENERATOR---")

    system_prompt = """Answer the following question based on this context:

    {context}

    Question: {question}
    """

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{question}")
    ])

    context_text = build_context(state["documents"])

    prompt_inputs = {
        "context": context_text,
        "question": state["question"],
        "chat_history": state["messages"]
    }

    llm = ChatOpenAI(temperature=0)

    answer = (prompt | llm | StrOutputParser()).invoke(prompt_inputs)

    return {"messages": [AIMessage(content=answer)]}