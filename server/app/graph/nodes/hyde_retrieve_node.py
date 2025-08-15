from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain.load import dumps, loads

from graph.state import State

def hyde_retrieve_node(state: State):

    retriever = state["vector_store"].as_retriever()

    # HyDE document generation
    template = """Please write a scientific paper passage to answer the question
    Question: {question}
    Passage:"""
    prompt_hyde = ChatPromptTemplate.from_template(template)

    generate_docs_for_retrieval = (
        prompt_hyde | ChatOpenAI(temperature=0) | StrOutputParser() 
    )

    retrieval_chain = generate_docs_for_retrieval | retriever 
    docs = retrieval_chain.invoke({"question":state["question"]})

    print(f"RETRIEVED {len(docs)} documents using HyDE query translation")

    return {"documents": docs}