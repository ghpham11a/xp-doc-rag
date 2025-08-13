from langchain_core.messages import AIMessage
from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser

from graph.state import State
from utils.build_context import build_context

def individual_decomposition_generate_node(state: State):

    print("---USING INDIVIDUAL DECOMPOSITION GENERATOR---")

    # Prompt
    system_prompt = """Here is a set of Q+A pairs:

    {context}

    Use these to synthesize an answer to the question: {question}
    """

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
        "context": state["q_a_pairs"],
        "question": state["question"],
        "chat_history": state["messages"],
    }

    answer = final_rag_chain.invoke(prompt_inputs)

    return {"messages": [AIMessage(content=answer)]}