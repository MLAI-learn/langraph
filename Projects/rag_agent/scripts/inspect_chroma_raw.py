import chromadb
from chromadb.config import Settings

client = chromadb.Client(
    Settings(
        persist_directory="./chroma_db",
        is_persistent=True,          # 🔑 REQUIRED
        anonymized_telemetry=False
    )
)

collections = client.list_collections()
print("📚 Collections found:", [c.name for c in collections])

for c in collections:
    print(f"➡ Collection '{c.name}' has {c.count()} documents")
