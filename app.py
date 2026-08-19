from fastapi import FastAPI
import chromadb
import os
from pathlib import Path
from google import genai

app = FastAPI(title="Mini RAG with Gemini")

# Gemini
gemini = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

# ChromaDB
chroma = chromadb.Client()
collection = chroma.get_or_create_collection(name="demo_documents")

# Load real text files from documents/
documents = []
ids = []

for file in Path("documents").glob("*.txt"):
    content = file.read_text(encoding="utf-8")

    # Simple chunking
    chunks = [content[i:i+500] for i in range(0, len(content), 500)]

    for number, chunk in enumerate(chunks):
        if chunk.strip():
            documents.append(chunk)
            ids.append(f"{file.stem}_{number}")

collection.add(
    documents=documents,
    ids=ids
)

@app.get("/")
def home():
    return {
        "message": "Mini RAG with ChromaDB + Gemini is running",
        "project": "Mini RAG AI Project",
        "developed_by": "A.Masmi",
        "location": "Montreal, Canada",
        "date": "August 19, 2026",
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
        model="gemini-3.6-flash",
        contents=prompt
    )

    return {
        "question": q,
        "retrieved_context": context,
        "gemini_answer": response.text
    }

from fastapi.responses import HTMLResponse

@app.get("/app", response_class=HTMLResponse)
def web_app():
    return """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">

<title>Mini RAG AI Assistant - A.Masmi</title>

<style>
    * {
        box-sizing: border-box;
    }

    body {
        margin: 0;
        font-family: Arial, Helvetica, sans-serif;
        background: #f4f7fb;
        color: #1f2937;
    }

    .hero {
        background: linear-gradient(135deg, #111827, #1e3a5f);
        color: white;
        padding: 34px 20px 42px;
    }

    .hero-inner {
        max-width: 1000px;
        margin: auto;
    }

    .credit {
        font-size: 14px;
        opacity: 0.9;
        margin-bottom: 18px;
        letter-spacing: 0.3px;
    }

    .hero h1 {
        margin: 0;
        font-size: 38px;
    }

    .hero p {
        margin-top: 10px;
        font-size: 17px;
        color: #dbeafe;
    }

    .badges {
        margin-top: 20px;
        display: flex;
        flex-wrap: wrap;
        gap: 8px;
    }

    .badge {
        background: rgba(255,255,255,0.12);
        padding: 7px 12px;
        border-radius: 20px;
        font-size: 13px;
    }

    .page {
        max-width: 1000px;
        margin: -24px auto 50px;
        padding: 0 20px;
    }

    .card {
        background: white;
        border-radius: 16px;
        padding: 30px;
        box-shadow: 0 8px 28px rgba(0,0,0,0.08);
        margin-bottom: 20px;
    }

    .card h2 {
        margin-top: 0;
        font-size: 25px;
    }

    .description {
        color: #64748b;
        margin-bottom: 22px;
    }

    .input-row {
        display: flex;
        gap: 10px;
    }

    input {
        flex: 1;
        padding: 15px 16px;
        font-size: 16px;
        border: 1px solid #cbd5e1;
        border-radius: 10px;
        outline: none;
    }

    input:focus {
        border-color: #2563eb;
    }

    button {
        border: 0;
        border-radius: 10px;
        padding: 14px 24px;
        font-size: 16px;
        cursor: pointer;
        background: #1e3a5f;
        color: white;
        transition: 0.2s;
    }

    button:hover {
        opacity: 0.88;
    }

    .examples {
        margin-top: 18px;
    }

    .examples-title {
        font-size: 14px;
        color: #64748b;
        margin-bottom: 8px;
    }

    .example-btn {
        background: #eef2f7;
        color: #334155;
        padding: 8px 12px;
        margin: 4px 4px 4px 0;
        font-size: 13px;
        border-radius: 18px;
    }

    .loading {
        display: none;
        margin-top: 20px;
        padding: 12px;
        color: #475569;
    }

    .answer-box {
        display: none;
        background: #eef9f1;
        border-left: 5px solid #16a34a;
    }

    .answer-label {
        font-size: 13px;
        text-transform: uppercase;
        letter-spacing: 0.7px;
        color: #15803d;
        font-weight: bold;
    }

    .answer-text {
        margin-top: 12px;
        font-size: 18px;
        line-height: 1.55;
    }

    .context-box {
        display: none;
    }

    details {
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 16px;
        background: #f8fafc;
    }

    summary {
        cursor: pointer;
        font-weight: bold;
        color: #334155;
    }

    .context-text {
        margin-top: 14px;
        line-height: 1.55;
        color: #475569;
    }

    .footer {
        text-align: center;
        color: #64748b;
        font-size: 14px;
        padding: 18px;
    }

    .footer strong {
        color: #334155;
    }

    @media (max-width: 700px) {
        .hero h1 {
            font-size: 30px;
        }

        .input-row {
            flex-direction: column;
        }

        button {
            width: 100%;
        }
    }
</style>
</head>

<body>

<section class="hero">
    <div class="hero-inner">

        <div class="credit">
            Developed by <strong>A.Masmi</strong> · Montreal, Canada · August 19, 2026
        </div>

        <h1>Mini RAG AI Assistant</h1>

        <p>
            A lightweight Retrieval-Augmented Generation application
            for learning and experimenting with AI concepts.
        </p>

        <div class="badges">
            <span class="badge">ChromaDB</span>
            <span class="badge">Gemini</span>
            <span class="badge">FastAPI</span>
            <span class="badge">Render</span>
            <span class="badge">RAG</span>
        </div>

    </div>
</section>

<main class="page">

    <section class="card">

        <h2>Ask the Knowledge Base</h2>

        <p class="description">
            Ask a question about RAG, embeddings, vector databases,
            Gemini, Langfuse, Docker, FastAPI, or model monitoring.
        </p>

        <div class="input-row">

            <input
                id="question"
                type="text"
                placeholder="Example: Why are embeddings used in RAG?"
                onkeydown="if(event.key === 'Enter') askQuestion()"
            >

            <button onclick="askQuestion()">
                Ask AI
            </button>

        </div>

        <div class="examples">

            <div class="examples-title">
                Try an example:
            </div>

            <button class="example-btn"
                onclick="setQuestion('What is model drift?')">
                Model drift
            </button>

            <button class="example-btn"
                onclick="setQuestion('Why are embeddings used in RAG?')">
                Embeddings
            </button>

            <button class="example-btn"
                onclick="setQuestion('What is ChromaDB used for?')">
                ChromaDB
            </button>

            <button class="example-btn"
                onclick="setQuestion('What does Langfuse monitor?')">
                Langfuse
            </button>

        </div>

        <div id="loading" class="loading">
            Searching the knowledge base and asking Gemini...
        </div>

    </section>

    <section id="answerBox" class="card answer-box">

        <div class="answer-label">
            Grounded AI Answer
        </div>

        <div id="answerText" class="answer-text"></div>

    </section>

    <section id="contextBox" class="card context-box">

        <details>

            <summary>
                View Retrieved Context
            </summary>

            <div id="contextText" class="context-text"></div>

        </details>

    </section>

    <div class="footer">

        <strong>Mini RAG AI Project</strong><br>

        Developed by A.Masmi · Montreal, Canada<br>

        ChromaDB + Gemini + FastAPI + Render

    </div>

</main>

<script>

function setQuestion(text) {
    document.getElementById("question").value = text;
    document.getElementById("question").focus();
}

async function askQuestion() {

    const q = document.getElementById("question").value.trim();

    if (!q) {
        alert("Please enter a question.");
        return;
    }

    const loading = document.getElementById("loading");
    const answerBox = document.getElementById("answerBox");
    const contextBox = document.getElementById("contextBox");

    loading.style.display = "block";
    answerBox.style.display = "none";
    contextBox.style.display = "none";

    try {

        const response =
            await fetch("/query?q=" + encodeURIComponent(q));

        if (!response.ok) {
            throw new Error("API request failed");
        }

        const data = await response.json();

        document.getElementById("answerText").textContent =
            data.gemini_answer;

        document.getElementById("contextText").textContent =
            data.retrieved_context;

        answerBox.style.display = "block";
        contextBox.style.display = "block";

    } catch (error) {

        document.getElementById("answerText").textContent =
            "The AI service could not answer the question. Please try again.";

        answerBox.style.display = "block";
    }

    loading.style.display = "none";
}

</script>

</body>
</html>
"""
