import chromadb
from chromadb.config import Settings

from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_core.documents import Document

from app.config import CHROMA_DB_DIR, EMBEDDING_MODEL, TOP_K


# --------------------------------------------------
# Create persistent Chroma client
# --------------------------------------------------

client = chromadb.Client(
    Settings(
        persist_directory=CHROMA_DB_DIR,
        is_persistent=True,
        anonymized_telemetry=False
    )
)

# --------------------------------------------------
# Embeddings (MUST match ingestion)
# --------------------------------------------------

embeddings = GoogleGenerativeAIEmbeddings(
    model=EMBEDDING_MODEL
)

# --------------------------------------------------
# Vector store (same collection as ingestion)
# --------------------------------------------------

db = Chroma(
    client=client,
    collection_name="docs",
    embedding_function=embeddings
)


# --------------------------------------------------
# Debug function
# --------------------------------------------------

def debug_retrieval(query: str, k: int = TOP_K):
    print("\n" + "=" * 100)
    print(f"🔍 QUERY: {query}")
    print("=" * 100)

    # Retrieve with similarity scores (distance)
    results = db.similarity_search_with_score(query, k=k)

    if not results:
        print("❌ No documents retrieved.")
        return

    for i, (doc, score) in enumerate(results):
        print(f"\n📄 RESULT #{i + 1}")
        print("-" * 100)
        print(f"📁 Source      : {doc.metadata.get('source', 'unknown')}")
        print(f"📏 Chunk length: {len(doc.page_content)}")
        print(f"📉 Distance    : {score:.4f}")
        print("\n📌 Chunk content:\n")
        print(doc.page_content)
        print("\n" + "-" * 100)


# --------------------------------------------------
# CLI entry point
# --------------------------------------------------

if __name__ == "__main__":
    query = input("\nEnter a query to debug retrieval:\n> ").strip()

    if not query:
        print("❌ Query cannot be empty.")
    else:
        debug_retrieval(query)
