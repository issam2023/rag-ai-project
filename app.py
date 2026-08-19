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
<html>
<head>
    <title>Mini RAG AI - A.Masmi</title>
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <style>
        body {
            font-family: Arial, sans-serif;
            background: #f4f6f8;
            margin: 0;
            color: #222;
        }
        .header {
            background: #172033;
            color: white;
            padding: 25px;
            text-align: center;
        }
        .container {
            max-width: 850px;
            margin: 40px auto;
            background: white;
            padding: 35px;
            border-radius: 12px;
            box-shadow: 0 4px 15px rgba(0,0,0,.10);
        }
        input {
            width: 75%;
            padding: 14px;
            font-size: 16px;
            border: 1px solid #bbb;
            border-radius: 7px;
        }
        button {
            padding: 14px 25px;
            font-size: 16px;
            background: #172033;
            color: white;
            border: 0;
            border-radius: 7px;
            cursor: pointer;
        }
        button:hover {
            opacity: .85;
        }
        .answer {
            margin-top: 30px;
            padding: 20px;
            background: #eef7ee;
            border-radius: 8px;
            display: none;
        }
        .context {
            margin-top: 15px;
            padding: 20px;
            background: #f3f3f3;
            border-radius: 8px;
            display: none;
        }
        .footer {
            text-align: center;
            margin-top: 35px;
            color: #666;
            font-size: 14px;
        }
        #loading {
            display: none;
            margin-top: 20px;
        }
    </style>
</head>

<body>

<div class="header">
    <h1>Mini RAG AI Assistant</h1>
    <p>ChromaDB + Gemini + FastAPI</p>
</div>

<div class="container">

    <h2>Ask the AI</h2>

    <p>Ask a question about the knowledge stored in the RAG system.</p>

    <input id="question"
           placeholder="Example: What is Gemini?"
           onkeydown="if(event.key==='Enter') askQuestion()">

    <button onclick="askQuestion()">Ask</button>

    <div id="loading">Thinking...</div>

    <div id="answer" class="answer">
        <h3>AI Answer</h3>
        <p id="answerText"></p>
    </div>

    <div id="context" class="context">
        <h3>Retrieved Context</h3>
        <p id="contextText"></p>
    </div>

    <div class="footer">
        <strong>Developed by A.Masmi</strong><br>
        Montreal, Canada<br>
        August 19, 2026
    </div>

</div>

<script>
async function askQuestion() {

    const q = document.getElementById("question").value.trim();

    if (!q) {
        alert("Please enter a question.");
        return;
    }

    document.getElementById("loading").style.display = "block";
    document.getElementById("answer").style.display = "none";
    document.getElementById("context").style.display = "none";

    try {
        const response = await fetch("/query?q=" + encodeURIComponent(q));

        if (!response.ok) {
            throw new Error("Request failed");
        }

        const data = await response.json();

        document.getElementById("answerText").textContent =
            data.gemini_answer;

        document.getElementById("contextText").textContent =
            data.retrieved_context;

        document.getElementById("answer").style.display = "block";
        document.getElementById("context").style.display = "block";

    } catch (error) {

        document.getElementById("answerText").textContent =
            "Sorry, an error occurred.";

        document.getElementById("answer").style.display = "block";
    }

    document.getElementById("loading").style.display = "none";
}
</script>

</body>
</html>
"""
