from langgraph.graph import StateGraph, END
from agent.state import AgentState
from agent.planner import planner_node
from agent.executor import executor_node
from agent.summarizer import summarizer_node

def build_graph():
    graph = StateGraph(AgentState)

    graph.add_node("planner", planner_node)
    graph.add_node("executor", executor_node)
    graph.add_node("summarizer", summarizer_node)

    graph.set_entry_point("planner")

    graph.add_edge("planner", "executor")
    graph.add_edge("executor", "summarizer")
    graph.add_edge("summarizer", END)

    return graph.compile()
