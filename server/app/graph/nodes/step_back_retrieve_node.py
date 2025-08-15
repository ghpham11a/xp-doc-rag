from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain.load import dumps, loads
from langchain_core.prompts import FewShotChatMessagePromptTemplate

from graph.state import State

# Multi Query: Different Perspectives
def step_back_retrieve_node(state: State):

    retriever = state["vector_store"].as_retriever()

    examples = [
        {
            "input": "Could the members of The Police perform lawful arrests?",
            "output": "what can the members of The Police do?",
        },
        {
            "input": "Jan Sindel’s was born in what country?",
            "output": "what is Jan Sindel’s personal history?",
        },
    ]
    # We now transform these to example messages
    example_prompt = ChatPromptTemplate.from_messages(
        [
            ("human", "{input}"),
            ("ai", "{output}"),
        ]
    )
    few_shot_prompt = FewShotChatMessagePromptTemplate(
        example_prompt=example_prompt,
        examples=examples,
    )
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """You are an expert at world knowledge. Your task is to step back and paraphrase a question to a more generic step-back question, which is easier to answer. Here are a few examples:""",
            ),
            # Few shot examples
            few_shot_prompt,
            # New question
            ("user", "{question}"),
        ]
    )

    generate_queries_step_back = prompt | ChatOpenAI(temperature=0) | StrOutputParser()

    docs = retriever.invoke(state["question"])

    step_back_retriever = generate_queries_step_back | retriever
    step_back_context = step_back_retriever.invoke({"question": state["question"]})

    print(f"RETRIEVED {len(docs)} documents using step back query translation")

    return {"documents": docs, "step_back_context": step_back_context}