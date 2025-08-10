from typing import Any, List
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse
import asyncio
import json

from langchain_core.messages import HumanMessage, AIMessage
from models import ChatRequest
from langchain.prompts import ChatPromptTemplate
from utils import qt_multi_query, qt_rag_fusion, qt_step_back, qt_hyde
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda

from typing import Annotated, Dict, Literal
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain.schema import Document
from langchain_core.vectorstores.base import VectorStore
from pydantic import BaseModel, Field
from langchain.load import dumps, loads
from langchain.chains.combine_documents import create_stuff_documents_chain

router = APIRouter(prefix="/chats", tags=["chats"])


class State(TypedDict):
    selectedRoutingType: str
    selectedQueryTranslation: str
    selectedQueryConstruction: str
    selectedIndexing: str
    selectedRetrieval: str
    selectedGeneration: str

    question: str
    generation: str = ""
    web_search: str = "No"
    documents: List[Document] = []
    web_search_used: bool = False
    document_grades: List[Dict[str, Any]] = []
    messages: Annotated[list, add_messages]
    vector_store: VectorStore
    query_translation_type: str
    llm: Any

    routed_vector: str

    class Config:
        arbitrary_types_allowed = True


class RouteQuery(BaseModel):
    """Route a user query to the most relevant datasource."""

    datasource: Literal["subject_one", "subject_two"] = Field(
        ...,
        description="Given a user question choose which datasource would be most relevant for answering their question",
    )

def build_context(docs: List[Document], max_chars: int = 8000):
    # Simple concatenation with source tags
    parts = []
    total = 0
    for i, d in enumerate(docs, 1):
        snippet = d.page_content.strip()
        tag = d.metadata.get("source") or f"doc_{i}"
        block = f"[Source: {tag}]\n{snippet}"
        if total + len(block) > max_chars:
            break
        parts.append(block)
        total += len(block)
    return "\n\n".join(parts)

async def build_workflow(chat_request: ChatRequest, request: Request):
    workflow_builder = StateGraph(State)

    def init_node(state: State):
        print(f"---INIT GRAPH---")
        return {
            "selectedRoutingType": chat_request.path[1],
            "selectedQueryTranslation": chat_request.path[0],
            "selectedQueryConstruction": chat_request.path[2],
            "selectedIndexing": chat_request.path[3],
            "selectedRetrieval": chat_request.path[4],
            "selectedGeneration": chat_request.path[5],
        }

    def routing_conditional_edge(state):
        """Decide whether to generate an answer or do web search"""
        print(f"---ASSESS ROUTING EDGE [{state['selectedRoutingType']}]---")

        if state["selectedRoutingType"] == "logical":
            return "logical_routing_node"

        if state["selectedRoutingType"] == "semantic":
            return "semantic_routing_node"

        return ""

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

        return {"routed_vector": route_and_invoke(chat_request.message)}

    def semantic_routing_node(state: State):
        return {}

    def query_construction_node(state: State):
        return {"llm": ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)}

    def indexing_node(state: State):
        if state["selectedIndexing"] == "multi-representation":
            if state["routed_vector"] == "subject_one":
                return {"vector_store": request.app.state.multi_vector_stores["subject_one"]}
            elif state["routed_vector"] == "subject_two":
                return {"vector_store": request.app.state.multi_vector_stores["subject_two"]}
        elif state["selectedIndexing"] == "raptor":
            if state["routed_vector"] == "subject_one":
                return {"vector_store": request.app.state.raptor_vector_stores["subject_one"]}
            elif state["routed_vector"] == "subject_two":
                return {"vector_store": request.app.state.raptor_vector_stores["subject_two"]}
        else:
            if state["routed_vector"] == "subject_one":
                return {"vector_store": request.app.state.vector_stores["subject_one"]}
            elif state["routed_vector"] == "subject_two":
                return {"vector_store": request.app.state.vector_stores["subject_two"]}
            
    def retrieve_conditional_edge(state):
        print(f"---ASSESS RETRIEVE EDGE [{state['selectedQueryTranslation']}]---")
        """Decide whether to generate an answer or do web search"""

        if state["selectedQueryTranslation"] == "multi-query":
            return "multi_query_retrieve_node"

        if state["selectedQueryTranslation"] == "rag-fusion":
            return "rag_fusion_retrieve_node"

        return ""

    
    def multi_query_retrieve_node(state: State):

        retriever = state["vector_store"].as_retriever()

        template = """You are an AI language model assistant. Your task is to generate five 
        different versions of the given user question to retrieve relevant documents from a vector 
        database. By generating multiple perspectives on the user question, your goal is to help
        the user overcome some of the limitations of the distance-based similarity search. 
        Provide these alternative questions separated by newlines. Original question: {question}"""
        prompt_perspectives = ChatPromptTemplate.from_template(template)

        generate_queries = (
            prompt_perspectives 
            | ChatOpenAI(temperature=0) 
            | StrOutputParser() 
            | (lambda x: x.split("\n"))
        )

        def get_unique_union(documents: list[list]):
            """ Unique union of retrieved docs """
            # Flatten list of lists, and convert each Document to string
            flattened_docs = [dumps(doc) for sublist in documents for doc in sublist]
            # Get unique documents
            unique_docs = list(set(flattened_docs))
            # Return
            return [loads(doc) for doc in unique_docs]

        # Retrieve
        retrieval_chain = generate_queries | retriever.map() | get_unique_union

        docs = retrieval_chain.invoke({"question": chat_request.message})

        print(f"RETRIEVED {len(docs)} documents using multi-query query translation")

        return {"documents": docs}
    
    def rag_fusion_retrieve_node(state: State):
        print("INSIDE rag_fusion_retrieve_node")
        
        return {}
    
    def post_process_retrieve_node(state: State):
        return {}
    
    def generate_conditional_edge(state: State):
        print(f"---ASSESS GENERATE EDGE [{state['selectedQueryTranslation']}]---")
        """Decide whether to generate an answer or do web search"""

        if state["selectedQueryTranslation"] == "multi-query":
            return "multi_query_generate_node"

        if state["selectedQueryTranslation"] == "rag-fusion":
            return "rag_fusion_generate_node"

        return ""
    
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
            "question": chat_request.message,
            "chat_history": state["messages"]
        }

        llm = ChatOpenAI(temperature=0)

        answer = (prompt | llm | StrOutputParser()).invoke(prompt_inputs)

        return {"messages": [llm.invoke(AIMessage(content=answer))]}
    
    def rag_fusion_generate_node(state: State):
        return {}

    def generate(state: State):
        # Extract last user message from state
        # last_message = state["messages"][-1].content
        # response = chain.invoke({"question": last_message})
        # Just pass through final state - execution happens in streaming
        return {}
    
    def post_process_generate_node(state: State):
        return {}
    
    # add nodes
    workflow_builder.add_node(init_node)
    workflow_builder.add_node(logical_routing_node)
    workflow_builder.add_node(semantic_routing_node)
    workflow_builder.add_node(query_construction_node)
    workflow_builder.add_node(indexing_node)
    workflow_builder.add_node(multi_query_retrieve_node)
    workflow_builder.add_node(rag_fusion_retrieve_node)
    workflow_builder.add_node(post_process_retrieve_node)
    workflow_builder.add_node(multi_query_generate_node)
    workflow_builder.add_node(rag_fusion_generate_node)
    workflow_builder.add_node(post_process_generate_node)

    # add edges
    workflow_builder.add_edge(START, "init_node")
    workflow_builder.add_conditional_edges(
        "init_node",
        routing_conditional_edge,
        {
            "logical_routing_node": "logical_routing_node",
            "semantic_routing_node": "semantic_routing_node",
        },
    )
    workflow_builder.add_edge("logical_routing_node", "query_construction_node")
    workflow_builder.add_edge("semantic_routing_node", "query_construction_node")
    workflow_builder.add_edge("query_construction_node", "indexing_node")
    workflow_builder.add_conditional_edges(
        "indexing_node",
        retrieve_conditional_edge,
        {
            "multi_query_retrieve_node": "multi_query_retrieve_node",
            "rag_fusion_retrieve_node": "rag_fusion_retrieve_node",
        },
    )
    workflow_builder.add_edge("multi_query_retrieve_node", "post_process_retrieve_node")
    workflow_builder.add_edge("rag_fusion_retrieve_node", "post_process_retrieve_node")
    workflow_builder.add_conditional_edges(
        "post_process_retrieve_node",
        generate_conditional_edge,
        {
            "multi_query_generate_node": "multi_query_generate_node",
        },
    )
    workflow_builder.add_edge("multi_query_generate_node", "post_process_generate_node")
    workflow_builder.add_edge("rag_fusion_generate_node", "post_process_generate_node")
    workflow_builder.add_edge("post_process_generate_node", END)

    return workflow_builder.compile()


@router.post("/send-stream")
async def chat_stream(chat_request: ChatRequest, request: Request):
    workflow = await build_workflow(chat_request, request)

    async def event_generator():
        try:
            async for chunk in workflow.astream({"messages": [{"role": "user", "content": chat_request.message}]}, stream_mode="messages"):
                # Handle different chunk formats
                if isinstance(chunk, tuple) and len(chunk) == 2:
                    # This is the format: (AIMessageChunk, metadata_dict)
                    msg_chunk, metadata = chunk
                    content = getattr(msg_chunk, "content", "")
                elif isinstance(chunk, dict) and "chatbot" in chunk:
                    # Original format
                    msg = chunk["chatbot"]["messages"][-1]
                    content = getattr(msg, "content", "")
                else:
                    # Direct message chunk
                    content = getattr(chunk, "content", "") if hasattr(chunk, "content") else ""

                if content:
                    yield f"data: {content}"
        except Exception as e:
            print(str(e))
            yield f"data: {json.dumps({'type': 'done'})}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")


    # async def stream_graph_updates(user_input: str):
    #     # Collect the final state from the workflow
    #     final_state = {}
        
    #     # Run workflow and collect state
    #     async for event in workflow.astream({"question": user_input}):
    #         for node_name, value in event.items():
    #             # Send progress updates
    #             if node_name != "__end__":
    #                 yield f"data: {json.dumps({'type': 'progress', 'node': node_name})}\n\n"
                
    #             # Accumulate state
    #             if isinstance(value, dict):
    #                 final_state.update(value)
        
    #     # After workflow completes, build and execute the chain with streaming
    #     vector_store = final_state.get("vector_store")
    #     llm = final_state.get("llm")
    #     query_translation_type = final_state.get("query_translation_type")
        
    #     if vector_store and llm and query_translation_type:
    #         # Build the query translation based on type
    #         if query_translation_type == "multi-query":
    #             query_translation_options = qt_multi_query.run(chat_request, request, vector_store)
    #         elif query_translation_type == "rag-fusion":
    #             query_translation_options = await qt_rag_fusion.run(chat_request, request, vector_store)
    #         elif query_translation_type == "step-back":
    #             query_translation_options = await qt_step_back.run(chat_request, request, vector_store)
    #         elif query_translation_type == "hyde":
    #             query_translation_options = await qt_hyde.run(chat_request, request, vector_store)
    #         else:
    #             # Default fallback
    #             query_translation_options = qt_multi_query.run(chat_request, request, vector_store)
            
    #         # Build the prompt
    #         prompt = ChatPromptTemplate.from_messages([
    #             ("system", query_translation_options.system_prompt),
    #             MessagesPlaceholder("chat_history"),
    #             ("human", "{input}")
    #         ])
            
    #         # Map function for the chain
    #         def map_to_prompt_format(x):
    #             return {
    #                 "context": x["context"],
    #                 "input": x["question"],
    #                 "question": x["question"],
    #                 "chat_history": x.get("chat_history", [])
    #             }
            
    #         # Build the complete chain
    #         rag_chain = (
    #             query_translation_options.chain_params
    #             | RunnableLambda(map_to_prompt_format)
    #             | prompt
    #             | llm
    #             | StrOutputParser()
    #         )
            
    #         # Stream the response
    #         async for chunk in rag_chain.astream({"question": user_input}):
    #             if chunk:
    #                 yield f"data: {json.dumps({'type': 'content', 'content': chunk})}\n\n"
    #                 await asyncio.sleep(0.005)
        
    #     # Send completion signal
    #     yield f"data: {json.dumps({'type': 'done'})}\n\n"

    # return StreamingResponse(stream_graph_updates(chat_request.message), media_type="text/event-stream")

    return ""