from pathlib import Path
import chromadb
from chromadb.utils import embedding_functions
from pypdf import PdfReader

DOCS_DIR = Path("docs")
DB_DIR = "chroma_db"
CHUNK_SIZE = 800
CHUNK_OVERLAP = 100


def load_text(path: Path) -> str:
    if path.suffix.lower() == ".pdf":
        reader = PdfReader(str(path))
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    return path.read_text(encoding="utf-8", errors="ignore")


def chunk_text(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    chunks = []
    start = 0
    while start < len(text):
        chunks.append(text[start:start + size])
        start += size - overlap
    return [c.strip() for c in chunks if c.strip()]


def main():
    files = [p for p in DOCS_DIR.glob("*") if p.suffix.lower() in (".pdf", ".txt", ".md")]
    if not files:
        print(f"Pon archivos .pdf, .txt o .md en la carpeta '{DOCS_DIR}/' y vuelve a correr este script.")
        return

    client = chromadb.PersistentClient(path=DB_DIR)
    embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
    collection = client.get_or_create_collection("documentos", embedding_function=embed_fn)

    ids, docs, metadatas = [], [], []
    for path in files:
        text = load_text(path)
        for i, chunk in enumerate(chunk_text(text)):
            ids.append(f"{path.name}-{i}")
            docs.append(chunk)
            metadatas.append({"source": path.name})

    collection.upsert(ids=ids, documents=docs, metadatas=metadatas)
    print(f"Listo: {len(docs)} fragmentos indexados de {len(files)} archivo(s).")


if __name__ == "__main__":
    main()
