import chromadb
from chromadb.config import Settings

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings

from .config import CHROMA_DB_DIR, EMBEDDING_MODEL

# --------------------------------------------------
# Explicit PERSISTENT Chroma client (this is the key)
# --------------------------------------------------

chroma_client = chromadb.Client(
    Settings(
        persist_directory=CHROMA_DB_DIR,
        is_persistent=True,          # 🔑 REQUIRED
        anonymized_telemetry=False
    )
)

embeddings = GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL)

vectorstore = Chroma(
    client=chroma_client,
    collection_name="docs",
    embedding_function=embeddings
)


def index_chunks(chunks, metadata):
    documents = [
        Document(page_content=chunk, metadata=metadata)
        for chunk in chunks
    ]

    vectorstore.add_documents(documents)
    return len(documents)
