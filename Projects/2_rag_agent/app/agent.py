from typing import TypedDict, List, Optional

import chromadb
from chromadb.config import Settings

from langgraph.graph import StateGraph, START, END
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_google_genai import (
    ChatGoogleGenerativeAI,
    GoogleGenerativeAIEmbeddings
)

from .config import (
    CHROMA_DB_DIR,
    EMBEDDING_MODEL,
    GENERATION_MODEL,
    TOP_K
)

# ==================================================
# 1. Agent State
# ==================================================

class AgentState(TypedDict):
    query: str
    retrieved_docs: Optional[List[Document]]
    answer: Optional[str]


# ==================================================
# 2. Persistent Chroma Vector Store
# ==================================================

chroma_client = chromadb.Client(
    Settings(
        persist_directory=CHROMA_DB_DIR,
        is_persistent=True,
        anonymized_telemetry=False
    )
)

embeddings = GoogleGenerativeAIEmbeddings(
    model=EMBEDDING_MODEL
)

vectorstore = Chroma(
    client=chroma_client,
    collection_name="docs",
    embedding_function=embeddings
)

# ==================================================
# 3. Gemini LLM (Generation)
# ==================================================

llm = ChatGoogleGenerativeAI(
    model=GENERATION_MODEL,
    temperature=0.0  # 🔒 deterministic, no creativity
)

# ==================================================
# 4. Graph Nodes
# ==================================================

def retrieve_docs(state: AgentState) -> AgentState:
    """
    Retrieve relevant documents from vector DB.
    """
    docs = vectorstore.similarity_search(
        state["query"],
        k=TOP_K
    )

    state["retrieved_docs"] = docs
    return state


def generate_answer(state: AgentState) -> AgentState:
    """
    Generate answer STRICTLY grounded in retrieved context.
    """
    docs = state.get("retrieved_docs", [])

    # 🚫 HARD STOP: no docs → no answer
    if not docs:
        state["answer"] = (
            "I don’t know based on the provided documents."
        )
        return state

    # Build context
    context = "\n\n".join(
        f"[{i}] {doc.page_content}"
        for i, doc in enumerate(docs)
    )

    # 🔒 Grounding-enforced prompt
    prompt = f"""
You are a factual AI assistant.

RULES (must follow strictly):
- Use ONLY the information from the context below
- Do NOT use any outside knowledge
- If the answer is not explicitly present, say:
  "I don’t know based on the provided documents."

Context:
{context}

Question:
{state["query"]}

Answer (cite sources like [0], [1]):
"""

    response = llm.invoke(prompt)
    state["answer"] = response.content.strip()

    return state


# ==================================================
# 5. LangGraph Definition
# ==================================================

def build_agent():
    graph = StateGraph(AgentState)

    graph.add_node("retrieve", retrieve_docs)
    graph.add_node("generate", generate_answer)

    graph.add_edge(START, "retrieve")
    graph.add_edge("retrieve", "generate")
    graph.add_edge("generate", END)

    return graph.compile()


# ==================================================
# 6. Public Helper
# ==================================================

def answer_query(graph, query: str) -> str:
    """
    External entry point for API / CLI.
    """
    result = graph.invoke({"query": query})
    return result["answer"]
