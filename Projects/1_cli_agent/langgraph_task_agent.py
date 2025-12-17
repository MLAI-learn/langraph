# langgraph_task_agent.py
import os
import json
import textwrap
from dotenv import load_dotenv
from rich.table import Table
from rich import print as rprint

# LangChain / LangGraph imports (LCEL style)
from typing_extensions import TypedDict, Annotated
import operator
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage, AnyMessage
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI

from langgraph.graph import StateGraph, START, END

# local DB helpers
from db_tools import (
    connect_db,
    add_task as db_add_task,
    list_tasks as db_list_tasks,
    complete_task as db_complete_task,
    delete_task as db_delete_task,
    search_tasks as db_search_tasks,
)

load_dotenv()

# ----------------------------
# Model setup (Gemini via LangChain provider package)
# ----------------------------
MODEL_NAME = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")  # adjust to your available model
model = ChatGoogleGenerativeAI(model=MODEL_NAME, temperature=0.0, max_output_tokens=512)

# ----------------------------
# Define tools (these are what the model can call)
# Each tool MUST have a docstring (or provide description= in @tool) for StructuredTool creation.
# ----------------------------
@tool
def add_task(title: str, description: str = "", category: str = "general", priority: str = "medium", due_date: str = None) -> str:
    """
    Insert a new task into the SQLite DB. Returns a JSON string with message and task_id.
    Args:
      title: short title of the task
      description: optional longer description
      category: task category like personal/work
      priority: low|medium|high
      due_date: optional ISO date string (YYYY-MM-DD)
    """
    conn = connect_db()
    tid = db_add_task(conn, title=title, description=description, category=category, priority=priority, due_date=due_date)
    return json.dumps({"msg": f"Added task {tid}", "task_id": tid})

@tool
def list_tasks(include_completed: bool = False) -> str:
    """
    Return a JSON string containing a list of tasks.
    Args:
      include_completed: if True, include completed tasks as well.
    """
    conn = connect_db()
    rows = db_list_tasks(conn, include_completed=include_completed)
    items = [{
        "id": r[0],
        "title": r[1],
        "description": r[2],
        "category": r[3],
        "priority": r[4],
        "due_date": r[5],
        "completed": bool(r[6])
    } for r in rows]
    return json.dumps({"tasks": items})

@tool
def complete_task(task_id: int) -> str:
    """
    Mark a task completed by id. Returns JSON with result message and updated flag.
    Args:
      task_id: integer id of the task to complete
    """
    conn = connect_db()
    updated = db_complete_task(conn, int(task_id))
    return json.dumps({"msg": f"Marked task {task_id} completed." if updated else f"Task {task_id} not found.", "updated": bool(updated)})

@tool
def delete_task(task_id: int) -> str:
    """
    Delete a task by id. Returns JSON with result message and deleted flag.
    Args:
      task_id: integer id of the task to delete
    """
    conn = connect_db()
    deleted = db_delete_task(conn, int(task_id))
    return json.dumps({"msg": f"Deleted task {task_id}." if deleted else f"Task {task_id} not found.", "deleted": bool(deleted)})

@tool
def search_tasks(query: str) -> str:
    """
    Search tasks by query (matches title or description). Returns JSON results list.
    Args:
      query: search substring
    """
    conn = connect_db()
    rows = db_search_tasks(conn, query)
    items = [{
        "id": r[0],
        "title": r[1],
        "description": r[2],
        "category": r[3],
        "priority": r[4],
        "due_date": r[5],
        "completed": bool(r[6])
    } for r in rows]
    return json.dumps({"results": items})

# Prepare tools
TOOLS = [add_task, list_tasks, complete_task, delete_task, search_tasks]
TOOLS_BY_NAME = {t.name: t for t in TOOLS}

# Bind tools to the model (so the model can emit tool_calls)
try:
    model_with_tools = model.bind_tools(TOOLS)
except Exception:
    # Some versions prefer `model.bind_tools` while others require a different mechanism.
    # Keep a fallback to the raw model to avoid breaking; if binding fails, the model will still reply (without tool calling).
    model_with_tools = model

# ----------------------------
# Define state type
# ----------------------------
class MessagesState(TypedDict):
    messages: Annotated[list[AnyMessage], operator.add]
    llm_calls: int

# ----------------------------
# Model node: call LLM (may produce tool_calls)
# ----------------------------
def llm_call(state: MessagesState):
    """Call the model with system prompt + conversation messages. Returns updated messages state."""
    system = SystemMessage(content=textwrap.dedent("""
    You are a task-agent assistant. Use the available tools to manage tasks when appropriate:
    add_task, list_tasks, complete_task, delete_task, search_tasks.
    If you want to perform an action, call a tool with JSON-serializable args.
    Otherwise return a normal assistant reply.
    """))
    # Combine system prompt + user messages
    incoming = state.get("messages", [])
    try:
        new_msg = model_with_tools.invoke([system] + incoming)
    except Exception:
        # Fallback: call raw model if binding paths differ
        new_msg = model.invoke([system] + incoming)
    return {
        "messages": [new_msg],
        "llm_calls": state.get("llm_calls", 0) + 1
    }

# ----------------------------
# Tool node: execute tool calls present in last message
# ----------------------------
def tool_node(state: MessagesState):
    """Execute any tool_calls the model produced and return ToolMessage(s) containing observations."""
    results = []
    last_msg = state["messages"][-1]
    for tool_call in getattr(last_msg, "tool_calls", []):
        name = tool_call.get("name")
        args = tool_call.get("args", {}) or {}
        tool_callable = TOOLS_BY_NAME.get(name)
        if not tool_callable:
            obs = json.dumps({"error": f"Unknown tool {name}"})
        else:
            # Use the tool's invoke if available, else call normally
            try:
                obs = tool_callable.invoke(args)
            except Exception:
                # fallback to direct call (positional/keyword handling)
                obs = tool_callable(**args)
        # Create a ToolMessage-like object. The runtime expects message-like return values;
        # here we return a dict with content and tool_call_id which will be wrapped by the runtime.
        results.append(ToolMessage(content=obs, tool_call_id=tool_call.get("id")))
    return {"messages": results}

# ----------------------------
# Conditional edge function
# ----------------------------
def should_continue(state: MessagesState) -> Any:
    """
    Decide next step after the LLM node.
    Returns either the string name of the next node ("tool_node") or the special END sentinel.
    (Using `Any` avoids Pylance complaining about non-literal values in Literal[].)
    """
    msgs = state.get("messages", [])
    last = msgs[-1] if msgs else None
    if last and getattr(last, "tool_calls", None):
        return "tool_node"
    return END


# ----------------------------
# Build and compile the StateGraph
# ----------------------------
agent_builder = StateGraph(MessagesState)

agent_builder.add_node("llm_call", llm_call)
agent_builder.add_node("tool_node", tool_node)

agent_builder.add_edge(START, "llm_call")
agent_builder.add_conditional_edges("llm_call", should_continue, ["tool_node", END])
agent_builder.add_edge("tool_node", "llm_call")

agent = agent_builder.compile()

# ----------------------------
# REPL + pretty printers
# ----------------------------
def pretty_print_tasks_list(json_str: str):
    try:
        data = json.loads(json_str)
        tasks = data.get("tasks") or data.get("results") or []
    except Exception:
        rprint(json_str)
        return
    if not tasks:
        rprint("[italic]No tasks returned.[/italic]")
        return
    table = Table(title="Tasks")
    table.add_column("ID", justify="right")
    table.add_column("Title")
    table.add_column("Category")
    table.add_column("Priority")
    table.add_column("Due")
    table.add_column("Done")
    for t in tasks:
        table.add_row(str(t.get("id")), t.get("title","")[:40], t.get("category",""), t.get("priority",""), t.get("due_date","-"), "✅" if t.get("completed") else "❌")
    rprint(table)

def run_repl():
    rprint("[bold]LangGraph Task Agent (Gemini-powered) — REPL[/bold]")
    while True:
        try:
            user = input(">>> ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nExiting.")
            break
        if not user:
            continue
        if user.lower() in ("exit","quit","q"):
            break
        messages = [HumanMessage(content=user)]
        state_in = {"messages": messages}
        result_state = agent.invoke(state_in)
        out_msgs = result_state.get("messages", [])
        for m in out_msgs:
            if isinstance(m, ToolMessage):
                try:
                    payload = json.loads(m.content)
                    if "tasks" in payload or "results" in payload:
                        pretty_print_tasks_list(m.content)
                    else:
                        rprint(payload)
                except Exception:
                    rprint(m.content)
            else:
                try:
                    rprint(m.content)
                except Exception:
                    rprint(str(m))

if __name__ == "__main__":
    run_repl()
