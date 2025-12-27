import os
from dotenv import load_dotenv
from tavily import TavilyClient

# Ensure environment variables are loaded
load_dotenv()

api_key = os.getenv("TAVILY_API_KEY")

if not api_key:
    raise RuntimeError(
        "TAVILY_API_KEY is not set. "
        "Please add it to your .env file or environment variables."
    )

client = TavilyClient(api_key=api_key)

def search_web(query: str) -> str:
    response = client.search(
        query=query,
        max_results=5,
        include_answer=True,
        include_raw_content=False
    )

    return response.get("answer", "")
