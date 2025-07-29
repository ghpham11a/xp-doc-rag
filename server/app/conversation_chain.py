from fastapi import FastAPI, HTTPException
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain import hub

def init_conversation_chain(app: FastAPI):
    if app.state.vector_store:
        llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo")
        
        # Create prompt template with chat history
        system_prompt = (
            "You are an assistant for question-answering tasks. "
            "Use the following pieces of retrieved context to answer "
            "the question. If you don't know the answer, say that you "
            "don't know. Use three sentences maximum and keep the "
            "answer concise."
            "\n\n"
            "{context}"
        )
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}")
        ])

        # prompt = hub.pull("rlm/rag-prompt")
        
        # Create the chains
        app.state.question_answer_chain = create_stuff_documents_chain(llm, prompt)
        app.state.retrieval_chain = create_retrieval_chain(
            app.state.vector_store.as_retriever(search_kwargs={"k": 3}),
            app.state.question_answer_chain
        )

def close_conversation_chain(app: FastAPI):
    pass