# 🚀 Personal RAG Knowledge Assistant with Pinecone & OpenAI

This is a complete Retrieval-Augmented Generation (RAG) project that turns your personal data (like your CV) into a question-answering assistant using:

- 🧠 **OpenAI** for generating embeddings & smart answers.
- 🗃 **Pinecone** as a vector database for semantic search.
- 🐍 **Python + FastAPI** for the backend.
- ⚛️ **React + MUI** for the frontend.

---

## 💡 What does it do?

✅ Loads your personal data from multiple `.txt` files inside a `knowledge/` folder.  
✅ Chunks and embeds using OpenAI, then uploads them to Pinecone.  
✅ When you ask a question, it finds the most relevant chunks and generates a smart GPT-based answer.

---

## 🏗 Full project structure

rag-knowledge-assistant/
├── env/ # Python virtual environment
├── .env # Your API keys (never commit this)
├── .gitignore # Ignores secrets & venv
├── requirements.txt # Python packages
├── fastapi_main.py # FastAPI server (answers questions over HTTP)
├── knowledge/ # Folder with multiple .txt files
│ ├── cv.txt
│ ├── projects.txt
│ └── skills.txt
└── ask-gpt-app/ # React frontend (built in week 1)

---

## 🚀 How to run the project

### ✅ 1️⃣ Clone the repo & setup Python

```bash
git clone https://github.com/yourusername/rag-knowledge-assistant.git
cd rag-knowledge-assistant

python3 -m venv env
source env/bin/activate

```

Install backend requirements
```bash
pip install -r requirements.txt
```

Your requirements.txt should look like:

fastapi
uvicorn
openai
pinecone
python-dotenv

✅ 3️⃣ Setup your .env file
Create a .env file in your backend root with:


OPENAI_API_KEY=sk-...
PINECONE_API_KEY=your-pinecone-key
PINECONE_INDEX=rag-knowledge-index-openai
✅ This keeps your secrets out of git.
(Your .gitignore includes .env and env/.)

✅ 4️⃣ Prepare your knowledge files
Create a knowledge/ folder and put any .txt files in there, e.g.:

knowledge/
├── cv.txt
├── projects.txt
└── skills.txt

✅ 5️⃣ Run your FastAPI server
```bash
uvicorn fastapi_main:app --reload
```

It will start on:
http://127.0.0.1:8000


✅ You can test with:

```bash
curl -X POST "http://127.0.0.1:8000/ask" \
    -H "Content-Type: application/json" \
    -d '{"question":"What companies has Shubham worked for?"}'
```
