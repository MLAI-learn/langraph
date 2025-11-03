# ====================== IMPORTS ==========================
from typing import Annotated, Sequence, TypedDict          # For type annotations and structured state typing
from dotenv import load_dotenv                             # Loads environment variables from a .env file (API keys, etc.)
from langchain_core.messages import (                      # LangChain message primitives to represent conversation turns
    BaseMessage,
    HumanMessage,
    AIMessage,
    ToolMessage,
    SystemMessage
)
from langchain_openai import ChatOpenAI                    # LangChain wrapper for OpenAI models (e.g., GPT-4o)
from langchain_core.tools import tool                      # Decorator to register Python functions as model-callable tools
from langgraph.graph.message import add_messages           # Helper to handle message merging inside AgentState
from langgraph.graph import StateGraph, END                # StateGraph allows defining node-based workflows; END marks termination
from langgraph.prebuilt import ToolNode                    # Prebuilt node type to automatically execute tool calls

# ==================== INITIAL SETUP ======================

load_dotenv()                                              # Load environment variables (e.g., OPENAI_API_KEY)

# Global variable to hold the document text throughout the session
document_content = ""

# Define what the agent's "state" looks like at each step in the graph
class AgentState(TypedDict):
    # messages: holds conversation history (user, AI, tool)
    # Annotated with add_messages → LangGraph knows how to merge messages across nodes
    messages: Annotated[Sequence[BaseMessage], add_messages]


# ====================== TOOL DEFINITIONS =================

@tool
def update(content: str) -> str:
    """
    Tool that updates the global document content.
    Args:
        content: new text to replace the current document.
    Returns:
        Confirmation message with new content.
    """
    global document_content                                # Access global variable
    document_content = content                             # Overwrite with new content
    # Return confirmation message (becomes a ToolMessage later)
    return f"Document has been updated successfully! The current content is:\n{document_content}"


@tool
def save(filename: str) -> str:
    """
    Tool that saves the current document to a .txt file.
    Args:
        filename: desired name for the file.
    Returns:
        Success or error message.
    """
    global document_content                                # Access global variable

    # Ensure the file name ends with ".txt"
    if not filename.endswith('.txt'):
        filename = f"{filename}.txt"

    try:
        # Open file in write mode and save current content
        with open(filename, 'w') as file:
            file.write(document_content)

        # Print a local confirmation (for console)
        print(f"\n💾 Document has been saved to: {filename}")

        # Return confirmation message for AI/user context
        return f"Document has been saved successfully to '{filename}'."

    # If writing fails (permissions, etc.), catch and return error
    except Exception as e:
        return f"Error saving document: {str(e)}"


# ====================== MODEL SETUP =======================

# Register both tools so the model can call them
tools = [update, save]

# Instantiate GPT-4o model and bind it with the tools (so it can invoke them)
model = ChatOpenAI(model="gpt-4o").bind_tools(tools)


# ====================== AGENT NODE ========================

def our_agent(state: AgentState) -> AgentState:
    """
    The main agent function — represents one conversational 'turn'.
    It handles:
      - Constructing the system prompt
      - Getting user input (if needed)
      - Invoking the model
      - Returning updated messages for the graph state
    """

    # Create system prompt that guides the LLM behavior
    system_prompt = SystemMessage(content=f"""
    You are Drafter, a helpful writing assistant. You are going to help the user update and modify documents.
    
    - If the user wants to update or modify content, use the 'update' tool with the complete updated content.
    - If the user wants to save and finish, you need to use the 'save' tool.
    - Make sure to always show the current document state after modifications.
    
    The current document content is:{document_content}
    """)

    # Handle first-turn initialization vs. subsequent turns
    if not state["messages"]:
        # If no prior messages → initial startup prompt
        user_input = "I'm ready to help you update a document. What would you like to create?"
        user_message = HumanMessage(content=user_input)
    else:
        # For ongoing interactions, ask user for input from the console
        user_input = input("\nWhat would you like to do with the document? ")
        print(f"\n👤 USER: {user_input}")
        user_message = HumanMessage(content=user_input)

    # Build the full message list sent to the LLM:
    #  system instructions + chat history + new user input
    all_messages = [system_prompt] + list(state["messages"]) + [user_message]

    # Invoke the OpenAI model (GPT-4o) with current conversation
    response = model.invoke(all_messages)

    # Print AI's natural-language output
    print(f"\n🤖 AI: {response.content}")

    # If the model decided to call a tool, list the tool names
    if hasattr(response, "tool_calls") and response.tool_calls:
        print(f"🔧 USING TOOLS: {[tc['name'] for tc in response.tool_calls]}")

    # Return updated state: include both the user's new message and the model response
    return {"messages": list(state["messages"]) + [user_message, response]}


# ====================== CONTINUATION LOGIC =================

def should_continue(state: AgentState) -> str:
    """
    Determines whether the graph should continue looping or end.
    Looks for a ToolMessage indicating that the document has been saved.
    """

    messages = state["messages"]

    if not messages:
        # No messages yet → keep going
        return "continue"

    # Scan the messages in reverse (latest first)
    for message in reversed(messages):
        # Check if the last tool output message confirms a save
        if (isinstance(message, ToolMessage) and 
            "saved" in message.content.lower() and
            "document" in message.content.lower()):
            # Found a 'save' confirmation → end the workflow
            return "end"

    # Otherwise, continue the loop
    return "continue"


# ====================== MESSAGE PRINTING HELPER =============

def print_messages(messages):
    """
    Helper to display only the recent tool outputs (ToolMessages)
    in a more readable way.
    """
    if not messages:
        return

    # Iterate over the last few messages
    for message in messages[-3:]:
        # Display only tool results
        if isinstance(message, ToolMessage):
            print(f"\n🛠️ TOOL RESULT: {message.content}")


# ====================== GRAPH DEFINITION ====================

# Initialize a new StateGraph using AgentState as schema
graph = StateGraph(AgentState)

# Add the main agent node that handles reasoning and conversation
graph.add_node("agent", our_agent)

# Add a prebuilt tool node that executes LLM tool calls
graph.add_node("tools", ToolNode(tools))

# Define entry point — the workflow always starts with the agent
graph.set_entry_point("agent")

# After agent finishes one reasoning step → go to tools node (execute any tool calls)
graph.add_edge("agent", "tools")

# After tools node → conditionally loop back or end, based on should_continue()
graph.add_conditional_edges(
    "tools",                      # From tools node
    should_continue,              # Function deciding the next step
    {
        "continue": "agent",      # Go back to agent for next turn
        "end": END,               # Stop workflow if save was confirmed
    },
)

# Compile the defined graph into an executable object
app = graph.compile()


# ====================== MAIN EXECUTION LOOP =================

def run_document_agent():
    """
    Entry point to run the Drafter agent interactively.
    Loops through steps in the graph and prints updates.
    """
    print("\n ===== DRAFTER =====")

    # Initial empty state (no conversation history yet)
    state = {"messages": []}

    # Stream through graph execution steps one by one
    # Each 'step' yields updated state values
    for step in app.stream(state, stream_mode="values"):
        if "messages" in step:
            # Print recent tool results (if any)
            print_messages(step["messages"])

    print("\n ===== DRAFTER FINISHED =====")


# Run the document agent if this script is executed directly
if __name__ == "__main__":
    run_document_agent()
