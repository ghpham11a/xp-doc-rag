from graph.state import State

def routing_conditional_edge(state: State):
    """Decide whether to generate an answer or do web search"""
    print(f"---ASSESS ROUTING EDGE [{state['selectedRoutingType']}]---")

    if state["selectedRoutingType"] == "logical":
        return "logical_routing_node"

    if state["selectedRoutingType"] == "semantic":
        return "semantic_routing_node"

    return ""