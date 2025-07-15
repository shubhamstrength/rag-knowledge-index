from fastapi import FastAPI
from pydantic import BaseModel
import os
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI
from pinecone import Pinecone
from fastapi.middleware.cors import CORSMiddleware


# Load env
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
pinecone_api_key = os.getenv("PINECONE_API_KEY")
pinecone_index_name = os.getenv("PINECONE_INDEX")

if not (openai_api_key and pinecone_api_key and pinecone_index_name):
    raise ValueError("Missing .env config!")

# Clients
client = OpenAI(api_key=openai_api_key)
pc = Pinecone(api_key=pinecone_api_key)
index = pc.Index(pinecone_index_name)

# FastAPI app
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 🔥 allows ALL origins. For production use your domain.
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Chunk function
def chunk_text(text, chunk_size=500, overlap=50):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start += chunk_size - overlap
    return chunks

# Load knowledge
knowledge_folder = Path("knowledge")
all_chunks = []
for file_path in knowledge_folder.glob("*.txt"):
    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()
    chunks = chunk_text(text)
    for chunk in chunks:
        all_chunks.append({
            "text": chunk,
            "source": file_path.name
        })

print(f"✅ Loaded {len(all_chunks)} chunks from {len(list(knowledge_folder.glob('*.txt')))} files.")

# Embed + upsert into Pinecone
for i, item in enumerate(all_chunks):
    embedding = client.embeddings.create(
        model="text-embedding-3-small",
        input=item["text"]
    ).data[0].embedding
    index.upsert([
        (f"chunk-{i}", embedding, {"text": item["text"], "source": item["source"]})
    ])

print(f"✅ Upserted {len(all_chunks)} chunks into Pinecone index '{pinecone_index_name}'.")

# Schema for request
class Question(BaseModel):
    question: str

@app.post("/ask")
async def ask_question(payload: Question):
    query_embedding = client.embeddings.create(
        model="text-embedding-3-small",
        input=payload.question
    ).data[0].embedding

    res = index.query(vector=query_embedding, top_k=3, include_metadata=True)

    # Defensive metadata extraction
    contexts = []
    for match in res["matches"]:
        meta = match.get("metadata", {})
        text = meta.get("text", "")
        source = meta.get("source", "unknown file")
        contexts.append(f"{text} (from {source})")

    # Compose final GPT answer
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{
            "role": "user",
            "content": f"Use the following context to answer the question:\n\n{contexts}\n\nQuestion: {payload.question}"
        }]
    )
    return {"answer": response.choices[0].message.content}
