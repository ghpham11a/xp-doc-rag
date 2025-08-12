from operator import itemgetter

from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain.load import dumps, loads
from langchain.schema import Document

from graph.state import State

def individual_decomposition_retrieve_node(state: State):

    retriever = state["vector_store"].as_retriever()

    # llm
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)

    decomposition_template = """You are a helpful assistant that generates multiple sub-questions related to an input question. \n
    The goal is to break down the input into a set of sub-problems / sub-questions that can be answers in isolation. \n
    Generate multiple search queries related to: {question} \n
    Output (3 queries):"""
    prompt_decomposition = ChatPromptTemplate.from_template(decomposition_template)

    # LLM
    llm = ChatOpenAI(temperature=0)

    # Chain
    generate_queries_decomposition = (prompt_decomposition | llm | StrOutputParser() | (lambda x: x.split("\n")))

    # Run
    questions = generate_queries_decomposition.invoke({"question": state["question"]})

    rag_template = """You are an assistant for question-answering tasks. 
    Use the following pieces of retrieved context to answer the question. 
    If you don’t know the answer, just say that you don’t know. 
    Use three sentences maximum and keep the answer concise.

    Question: {question}
    Context: {context}
    Answer:"""

    prompt_rag = ChatPromptTemplate.from_template(rag_template)

    def retrieve_and_rag(question, prompt_rag,sub_question_generator_chain):
        """RAG on each sub-question"""
        # Use our decomposition / 
        sub_questions = sub_question_generator_chain.invoke({"question": state["question"]})
        
        # Initialize a list to hold RAG chain results
        rag_results = []
        
        for sub_question in sub_questions:
            
            # Retrieve documents for each sub-question
            retrieved_docs = retriever.get_relevant_documents(sub_question)
            
            # Use retrieved documents and sub-question in RAG chain
            answer = (prompt_rag | llm | StrOutputParser()).invoke({"context": retrieved_docs, 
                                                                    "question": sub_question})
            rag_results.append(answer)
        
        return rag_results, sub_question
    
    # Wrap the retrieval and RAG process in a RunnableLambda for integration into a chain
    answers, questions = retrieve_and_rag(state["question"], prompt_rag, generate_queries_decomposition)

    def format_qa_pairs(questions, answers):
        """Format Q and A pairs"""
        
        formatted_string = ""
        for i, (question, answer) in enumerate(zip(questions, answers), start=1):
            formatted_string += f"Question {i}: {question}\nAnswer {i}: {answer}\n\n"
        return formatted_string.strip()

    q_a_pairs = format_qa_pairs(questions, answers)

    retriever_chain = itemgetter("question") | retriever

    docs = retriever_chain.invoke({ "question": state["question"] })

    print(f"RETRIEVED {len(docs)} documents using individual decomposition query translation")

    return {"documents": docs, "q_a_pairs": q_a_pairs}