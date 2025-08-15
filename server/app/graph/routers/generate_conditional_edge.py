from graph.state import State

def generate_conditional_edge(state: State):
    print(f"---ASSESS GENERATE EDGE [{state['selectedQueryTranslation']}]---")
    """Decide whether to generate an answer or do web search"""

    if state["selectedQueryTranslation"] == "multi-query":
        return "multi_query_generate_node"

    if state["selectedQueryTranslation"] == "rag-fusion":
        return "rag_fusion_generate_node"
    
    if state["selectedQueryTranslation"] == "recursive-decomposition":
        return "recursive_decomposition_generate_node"
    
    if state["selectedQueryTranslation"] == "individual-decomposition":
        return "individual_decomposition_generate_node"
    
    if state["selectedQueryTranslation"] == "step-back":
        return "step_back_generate_node"

    return ""