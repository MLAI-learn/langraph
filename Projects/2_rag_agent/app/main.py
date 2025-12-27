from fastapi import FastAPI, Query
from .agent import build_agent, answer_query


app = FastAPI(
    title="RAG FAQ Agent",
    description="LangGraph + Gemini powered RAG agent",
    version="1.0.0"
)

# --------------------------------------------------
# Build LangGraph once at startup
# --------------------------------------------------

rag_graph = build_agent()


# --------------------------------------------------
# API endpoint
# --------------------------------------------------

@app.get("/query")
def query(q: str = Query(..., min_length=3)):
    """
    Ask a question against the RAG agent.
    """
    answer = answer_query(rag_graph, q)
    return {
        "query": q,
        "answer": answer
    }
