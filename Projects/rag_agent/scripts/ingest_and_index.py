import os
from app.ingest import ingest_file
from app.index import index_chunks

DATA_DIR = "./data"

def main():
    for filename in os.listdir(DATA_DIR):
        path = os.path.join(DATA_DIR, filename)

        if not os.path.isfile(path):
            continue

        print(f"\n📄 Processing: {filename}")

        chunks = ingest_file(path)

        metadata = {
            "source": filename
        }

        index_chunks(chunks, metadata)

        print(f"✅ Indexed {len(chunks)} chunks from {filename}")

    print("\n🎉 Ingestion & indexing complete!")

if __name__ == "__main__":
    main()
