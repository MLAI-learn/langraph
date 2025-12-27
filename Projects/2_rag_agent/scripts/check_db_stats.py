import chromadb
from chromadb.config import Settings

client = chromadb.Client(
    Settings(
        persist_directory="./chroma_db",
        anonymized_telemetry=False
    )
)

collections = client.list_collections()
print("📚 Collections:", [c.name for c in collections])

for c in collections:
    print(f"➡ {c.name}: {c.count()} documents")
