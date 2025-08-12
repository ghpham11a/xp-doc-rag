from fastapi import Request
from langgraph.graph import StateGraph, START, END

from models import ChatRequest
from graph.state import State
from graph.nodes.logical_routing_node import logical_routing_node
from graph.nodes.semantic_routing_node import semantic_routing_node
from graph.nodes.query_construction_node import query_construction_node
from graph.nodes.multi_query_retrieve_node import multi_query_retrieve_node
from graph.nodes.rag_fusion_retrieve_node import rag_fusion_retrieve_node
from graph.nodes.recursive_decomposition_retrieve_node import recursive_decomposition_retrieve_node
from graph.nodes.individual_decomposition_retrieve_node import individual_decomposition_retrieve_node
from graph.nodes.post_process_retrieve_node import post_process_retrieve_node
from graph.nodes.multi_query_generate_node import multi_query_generate_node
from graph.nodes.rag_fusion_generate_node import rag_fusion_generate_node
from graph.nodes.recursive_decomposition_generate_node import recursive_decomposition_generate_node
from graph.nodes.individual_decomposition_generate_node import individual_decomposition_generate_node
from graph.nodes.post_process_generate_node import post_process_generate_node
from graph.routers.routing_conditional_edge import routing_conditional_edge
from graph.routers.retrieve_conditional_edge import retrieve_conditional_edge
from graph.routers.generate_conditional_edge import generate_conditional_edge


async def build_workflow(chat_request: ChatRequest, request: Request):
    workflow_builder = StateGraph(State)

    def init_node(state: State):
        print(f"---INIT GRAPH---")
        return {
            "question": chat_request.message,
            "selectedRoutingType": chat_request.path[1],
            "selectedQueryTranslation": chat_request.path[0],
            "selectedQueryConstruction": chat_request.path[2],
            "selectedIndexing": chat_request.path[3],
            "selectedRetrieval": chat_request.path[4],
            "selectedGeneration": chat_request.path[5],
        }

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
    
    # add nodes
    workflow_builder.add_node(init_node)
    workflow_builder.add_node(logical_routing_node)
    workflow_builder.add_node(semantic_routing_node)
    workflow_builder.add_node(query_construction_node)
    workflow_builder.add_node(indexing_node)
    workflow_builder.add_node(multi_query_retrieve_node)
    workflow_builder.add_node(rag_fusion_retrieve_node)
    workflow_builder.add_node(recursive_decomposition_retrieve_node)
    workflow_builder.add_node(individual_decomposition_retrieve_node)
    workflow_builder.add_node(post_process_retrieve_node)
    workflow_builder.add_node(multi_query_generate_node)
    workflow_builder.add_node(rag_fusion_generate_node)
    workflow_builder.add_node(recursive_decomposition_generate_node)
    workflow_builder.add_node(individual_decomposition_generate_node)
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
            "recursive_decomposition_retrieve_node": "recursive_decomposition_retrieve_node",
            "individual_decomposition_retrieve_node": "individual_decomposition_retrieve_node"
        },
    )
    workflow_builder.add_edge("multi_query_retrieve_node", "post_process_retrieve_node")
    workflow_builder.add_edge("rag_fusion_retrieve_node", "post_process_retrieve_node")
    workflow_builder.add_edge("recursive_decomposition_retrieve_node", "post_process_retrieve_node")
    workflow_builder.add_edge("individual_decomposition_retrieve_node", "post_process_retrieve_node")
    workflow_builder.add_conditional_edges(
        "post_process_retrieve_node",
        generate_conditional_edge,
        {
            "multi_query_generate_node": "multi_query_generate_node",
            "rag_fusion_generate_node": "rag_fusion_generate_node",
            "recursive_decomposition_generate_node": "recursive_decomposition_generate_node",
            "individual_decomposition_generate_node": "individual_decomposition_generate_node"
        },
    )
    workflow_builder.add_edge("multi_query_generate_node", "post_process_generate_node")
    workflow_builder.add_edge("rag_fusion_generate_node", "post_process_generate_node")
    workflow_builder.add_edge("recursive_decomposition_generate_node", "post_process_generate_node")
    workflow_builder.add_edge("individual_decomposition_generate_node", "post_process_generate_node")
    workflow_builder.add_edge("post_process_generate_node", END)

    return workflow_builder.compile()