from langchain_core.messages import AIMessage
from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser

from graph.state import State
from utils.build_context import build_context

def step_back_generate_node(state: State):

    print("---USING STEP BACK GENERATOR---")

    # Prompt
    system_prompt = """You are an expert of world knowledge. I am going to ask you a question. Your response should be comprehensive and not contradicted with the following context if they are relevant. Otherwise, ignore them if they are not relevant.

    # {normal_context}
    # {step_back_context}

    # Original Question: {question}
    # Answer:"""

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{question}")
    ])

    llm = ChatOpenAI(temperature=0)

    final_rag_chain = (
        prompt
        | llm
        | StrOutputParser()
    )

    prompt_inputs = {
        "normal_context": build_context(state["documents"]),
        "question": state["question"],
        "step_back_context": build_context(state["step_back_context"]),
        "chat_history": state["messages"],
    }

    answer = final_rag_chain.invoke(prompt_inputs)

    return {"messages": [AIMessage(content=answer)]}