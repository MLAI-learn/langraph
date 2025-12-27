from agent.state import AgentState
from tools.search_tools import search_web

def executor_node(state: AgentState) -> AgentState:
    action = state.get("current_action")
    if not action:
        return state

    try:
        if action["action"] == "search_web":
            result = search_web(action["args"]["query"])
            state["extracted_text"] = result
            state["history"].append("OBS: Web search completed")
    except Exception as e:
        state["extracted_text"] = None
        state["history"].append(f"OBS: Search error - {str(e)}")

    state["done"] = True
    return state
