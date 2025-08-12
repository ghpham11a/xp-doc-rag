from langchain_core.messages import AIMessage
from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser

from graph.state import State
from utils.build_context import build_context

def individual_decomposition_generate_node(state: State):

    print("---USING RECURSIVE DECOMPOSITION GENERATOR---")

    # Prompt
    template = """Here is a set of Q+A pairs:

    {context}

    Use these to synthesize an answer to the question: {question}
    """

    prompt = ChatPromptTemplate.from_template(template)

    llm = ChatOpenAI(temperature=0)

    final_rag_chain = (
        prompt
        | llm
        | StrOutputParser()
    )

    answer = final_rag_chain.invoke({"context":state[""],"question":state["question"]})

    return {"messages": [AIMessage(content=answer)]}