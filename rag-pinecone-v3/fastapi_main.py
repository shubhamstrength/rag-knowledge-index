# fastapi_main.py
# RAG API with:
# - Local Mistral (llama-cpp-python) for generation
# - OpenAI embeddings for vectors
# - Pinecone for vector search
# - TXT/PDF/MD ingestion + web page ingestion

import os
import hashlib
from pathlib import Path
from typing import List, Dict, Optional

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import requests
from bs4 import BeautifulSoup

# PDF loaders (prefer pypdf; fallback to PyMuPDF if installed)
try:
    from pypdf import PdfReader
    HAVE_PYPDF = True
except Exception:
    HAVE_PYPDF = False
    try:
        import fitz  # PyMuPDF
        HAVE_FITZ = True
    except Exception:
        HAVE_FITZ = False

from openai import OpenAI
from pinecone import Pinecone
from llama_cpp import Llama


# ----------------------------
# Environment / Clients
# ----------------------------
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX = os.getenv("PINECONE_INDEX")
LLM_MODEL_PATH = os.getenv("LLM_MODEL_PATH")  # e.g. ../models/mistral.gguf

if not (OPENAI_API_KEY and PINECONE_API_KEY and PINECONE_INDEX and LLM_MODEL_PATH):
    raise ValueError("Missing .env config! Please set OPENAI_API_KEY, PINECONE_API_KEY, PINECONE_INDEX, LLM_MODEL_PATH")

# OpenAI for embeddings
oa = OpenAI(api_key=OPENAI_API_KEY)

# Pinecone
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(PINECONE_INDEX)

# Local LLM (Mistral gguf via llama.cpp)
if not Path(LLM_MODEL_PATH).exists():
    raise ValueError(f"LLM model file not found at: {LLM_MODEL_PATH}")
llm = Llama(model_path=LLM_MODEL_PATH, n_ctx=4096)

# ----------------------------
# FastAPI setup
# ----------------------------
app = FastAPI(title="RAG API (Mistral + Pinecone)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten for prod
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ----------------------------
# Models
# ----------------------------
class Question(BaseModel):
    question: str

class IngestURL(BaseModel):
    url: str

# ----------------------------
# Utils: Chunking / IDs / Embedding / Upsert
# ----------------------------
def chunk_text(text: str, chunk_size: int = 800, overlap: int = 120) -> List[str]:
    text = text.strip()
    if not text:
        return []
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunks.append(text[start:end])
        start += max(chunk_size - overlap, 1)
    return chunks

def make_id(source: str, idx: int, text: str) -> str:
    h = hashlib.md5(text.encode("utf-8")).hexdigest()[:12]
    return f"{source}:{idx}:{h}"

def embed_batch(texts: List[str]) -> List[List[float]]:
    if not texts:
        return []
    resp = oa.embeddings.create(
        model="text-embedding-3-small",  # 1536-dim
        input=texts
    )
    return [d.embedding for d in resp.data]

def upsert_chunks(chunks: List[Dict[str, str]], source: str) -> int:
    # chunks: [{"text": ..., "source": source}]
    if not chunks:
        return 0

    # Embed in batches
    BATCH = 100
    total = 0
    for i in range(0, len(chunks), BATCH):
        batch = chunks[i:i+BATCH]
        texts = [c["text"] for c in batch]
        vectors = embed_batch(texts)

        # Pinecone upsert payload (new SDK accepts dicts)
        payload = []
        for j, (vec, item) in enumerate(zip(vectors, batch)):
            _id = make_id(source, i + j, item["text"])
            payload.append({"id": _id, "values": vec, "metadata": {"text": item["text"], "source": source}})

        index.upsert(vectors=payload)
        total += len(payload)
    return total

# ----------------------------
# Loaders: TXT / PDF / MD
# ----------------------------
def load_txt(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")

def load_md(path: Path) -> str:
    # Markdown to text (simple strip; you can improve by removing fenced code, etc.)
    return path.read_text(encoding="utf-8", errors="ignore")

def load_pdf(path: Path) -> str:
    if HAVE_PYPDF:
        text = []
        reader = PdfReader(str(path))
        for page in reader.pages:
            try:
                text.append(page.extract_text() or "")
            except Exception:
                pass
        return "\n".join(text)
    elif HAVE_FITZ:
        t = []
        doc = fitz.open(str(path))
        for p in doc:
            t.append(p.get_text())
        return "\n".join(t)
    else:
        raise RuntimeError("No PDF parser available. Install 'pypdf' or 'PyMuPDF'.")

def load_file(path: Path) -> str:
    ext = path.suffix.lower()
    if ext == ".txt":
        return load_txt(path)
    if ext == ".md":
        return load_md(path)
    if ext == ".pdf":
        return load_pdf(path)
    return ""

def ingest_knowledge_folder(folder: Path = Path("knowledge")) -> Dict[str, int]:
    if not folder.exists():
        return {"files": 0, "chunks": 0, "upserted": 0}

    total_files = 0
    total_chunks = 0
    total_upserted = 0

    for path in folder.glob("*"):
        if not path.is_file():
            continue
        text = load_file(path)
        if not text.strip():
            continue

        chunks = [{"text": t, "source": path.name} for t in chunk_text(text)]
        upserted = upsert_chunks(chunks, source=path.name)

        total_files += 1
        total_chunks += len(chunks)
        total_upserted += upserted

    return {"files": total_files, "chunks": total_chunks, "upserted": total_upserted}

# ----------------------------
# Web Ingestion
# ----------------------------
def fetch_webpage(url: str) -> str:
    try:
        r = requests.get(url, timeout=20)
        r.raise_for_status()
        soup = BeautifulSoup(r.text, "html.parser")
        for s in soup(["script", "style", "noscript"]):
            s.extract()
        text = soup.get_text(separator="\n")
        # normalize whitespace
        lines = [ln.strip() for ln in text.splitlines()]
        text = "\n".join([ln for ln in lines if ln])
        return text
    except Exception as e:
        raise RuntimeError(f"Failed to fetch/parse URL: {e}")

def ingest_url(url: str) -> Dict[str, int]:
    text = fetch_webpage(url)
    if not text.strip():
        return {"chunks": 0, "upserted": 0}
    chunks = [{"text": t, "source": url} for t in chunk_text(text)]
    upserted = upsert_chunks(chunks, source=url)
    return {"chunks": len(chunks), "upserted": upserted}

# ----------------------------
# QA with local Mistral
# ----------------------------
def generate_answer_with_llm(contexts: List[str], question: str) -> str:
    context_block = "\n\n".join(contexts)
    prompt = f"""
### Instruction:
You are a helpful assistant. Answer the question using ONLY the provided context. If the answer is not in the context, say you don't know.

### Context:
{context_block}

### Question:
{question}

### Answer:
""".strip()

    out = llm(prompt, max_tokens=512, stop=["###"])
    return out["choices"][0]["text"].strip()

# ----------------------------
# Startup: (optional) auto-ingest
# ----------------------------
@app.on_event("startup")
def on_startup():
    # Auto-ingest knowledge folder on startup
    stats = ingest_knowledge_folder(Path("knowledge"))
    print(f"🔍 Startup ingest -> files: {stats['files']} | chunks: {stats['chunks']} | upserted: {stats['upserted']}")

# ----------------------------
# Endpoints
# ----------------------------
@app.get("/healthz")
def healthz():
    return {"status": "ok"}

@app.post("/reindex")
def reindex():
    stats = ingest_knowledge_folder(Path("knowledge"))
    return {"message": "reindexed", **stats}

@app.post("/ingest_url")
def ingest_url_endpoint(body: IngestURL):
    if not body.url:
        raise HTTPException(status_code=400, detail="url is required")
    try:
        stats = ingest_url(body.url)
        return {"message": "ingested", "url": body.url, **stats}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/ask")
def ask(body: Question):
    if not body.question or not body.question.strip():
        raise HTTPException(status_code=400, detail="question is required")

    # Embed the question
    q_vec = embed_batch([body.question])[0]

    # Query Pinecone
    res = index.query(vector=q_vec, top_k=5, include_metadata=True)

    matches = res.get("matches", []) if isinstance(res, dict) else getattr(res, "matches", [])
    if not matches:
        return {"answer": "I don't know based on the current knowledge."}

    # Build contexts: "text (from source)"
    contexts = []
    for m in matches:
        # works for dict or object style
        md = m.get("metadata", {}) if isinstance(m, dict) else getattr(m, "metadata", {})
        txt = md.get("text", "")
        src = md.get("source", "unknown")
        if txt:
            contexts.append(f"{txt}\n(from {src})")

    if not contexts:
        return {"answer": "I don't know based on the current knowledge."}

    # Generate locally with Mistral
    answer = generate_answer_with_llm(contexts, body.question)
    return {"answer": answer, "sources": list({ctx.split('(from')[-1].strip(') ').strip() for ctx in contexts})}
