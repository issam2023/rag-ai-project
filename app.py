from fastapi import FastAPI
import chromadb

app = FastAPI(title="Mini RAG with ChromaDB")

# Create an in-memory Chroma database
client = chromadb.Client()

collection = client.get_or_create_collection(
    name="demo_documents"
)

documents = [
    "Machine learning allows computers to learn patterns from data.",
    "RAG retrieves relevant documents before generating an answer.",
    "Render is a cloud platform used to deploy web applications.",
    "Gemini is a large language model developed by Google.",
    "Langfuse provides observability and tracing for LLM applications."
]

ids = ["doc1", "doc2", "doc3", "doc4", "doc5"]

# Chroma automatically creates embeddings
collection.add(
    documents=documents,
    ids=ids
)

@app.get("/")
def home():
    return {
        "message": "Mini RAG with ChromaDB is running",
        "documents": len(documents)
    }

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/query")
def query(q: str):

    results = collection.query(
        query_texts=[q],
        n_results=1
    )

    retrieved = results["documents"][0][0]
    distance = results["distances"][0][0]

    return {
        "question": q,
        "retrieved_document": retrieved,
        "distance": round(float(distance), 3)
    }
