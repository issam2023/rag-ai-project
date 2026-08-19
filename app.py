from fastapi import FastAPI
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = FastAPI(title="Mini RAG Demo")

documents = [
    "Machine learning allows computers to learn patterns from data.",
    "RAG retrieves relevant documents before generating an answer.",
    "Render is a cloud platform used to deploy web applications."
]

vectorizer = TfidfVectorizer()
doc_vectors = vectorizer.fit_transform(documents)

@app.get("/")
def home():
    return {
        "message": "Mini RAG is running",
        "try": "/query?q=What is RAG?"
    }

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/query")
def query(q: str):
    query_vector = vectorizer.transform([q])
    scores = cosine_similarity(query_vector, doc_vectors)[0]

    best_index = scores.argmax()

    return {
        "question": q,
        "retrieved_document": documents[best_index],
        "similarity_score": round(float(scores[best_index]), 3)
    }
