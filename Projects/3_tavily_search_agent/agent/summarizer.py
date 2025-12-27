from langchain_google_genai import ChatGoogleGenerativeAI
from agent.state import AgentState
from config import GEMINI_MODEL

llm = ChatGoogleGenerativeAI(
    model=GEMINI_MODEL,
    temperature=0.2
)

PROMPT = """
You are an intelligent research assistant and search agent.

Your task is to analyze web search results and produce a clear,
fact-based response to the user's original query.

GUIDELINES:
- Focus on answering the user's question directly.
- Use only the information present in the search results.
- Do not invent facts or speculate.
- If information is incomplete or unclear, state that explicitly.
- Prefer concise, structured explanations.

USER QUERY:
{query}

WEB SEARCH RESULTS:
{text}

EXPECTED OUTPUT:
- A concise, well-structured answer to the user's query.
- Bullet points where helpful.
- Clear distinctions between confirmed facts and uncertain information.
"""


def summarizer_node(state: AgentState) -> AgentState:
    text = state.get("extracted_text")
    query = state.get("user_task")

    if not text or len(text.strip()) < 20:
        state["summary"] = (
            "The search results did not contain enough reliable "
            "information to answer the query."
        )
        return state

    response = llm.invoke(
        PROMPT.format(
            query=query,
            text=text[:6000]
        )
    )

    state["summary"] = response.content
    return state

