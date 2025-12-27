from agent.state import AgentState

def planner_node(state: AgentState) -> AgentState:
    """
    Deterministic planner for Tavily-based search.
    No browser. No loops. No guessing.
    """

    action = {
        "action": "search_web",
        "args": {
            "query": state["user_task"]
        }
    }

    state["current_action"] = action
    state["history"].append(f"PLAN: {action}")
    return state
