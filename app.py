from fastapi import FastAPI
import chromadb
import os
from google import genai

app = FastAPI(title="Mini RAG with Gemini")

# Gemini
gemini = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

# ChromaDB
chroma = chromadb.Client()
collection = chroma.get_or_create_collection(name="demo_documents")

documents = [
    "Machine learning allows computers to learn patterns from data.",
    "RAG retrieves relevant documents before generating an answer.",
    "Render is a cloud platform used to deploy web applications.",
    "Gemini is a large language model developed by Google.",
    "Langfuse provides observability and tracing for LLM applications."
]

ids = ["doc1", "doc2", "doc3", "doc4", "doc5"]

collection.add(
    documents=documents,
    ids=ids
)

@app.get("/")
def home():
    return {
        "message": "Mini RAG with ChromaDB + Gemini is running",
        "try": "/query?q=What is Gemini?"
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

    context = results["documents"][0][0]

    prompt = f"""
Answer the question using ONLY the context below.

Context:
{context}

Question:
{q}

If the context does not contain the answer, say:
I don't know based on the provided context.
"""

    response = gemini.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt
    )

    return {
        "question": q,
        "retrieved_context": context,
        "gemini_answer": response.text
    }
