from graph.state import State

def retrieve_conditional_edge(state: State):
    print(f"---ASSESS RETRIEVE EDGE [{state['selectedQueryTranslation']}]---")
    """Decide whether to generate an answer or do web search"""

    if state["selectedQueryTranslation"] == "multi-query":
        return "multi_query_retrieve_node"

    if state["selectedQueryTranslation"] == "rag-fusion":
        return "rag_fusion_retrieve_node"

    return ""