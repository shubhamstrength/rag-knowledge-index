# 🚀 Personal RAG Knowledge Assistant with Pinecone & OpenAI

This is a complete Retrieval-Augmented Generation (RAG) project that turns your personal data (like your CV) into a question-answering assistant using:

- 🧠 **OpenAI** for generating embeddings & answering questions.
- 🗃 **Pinecone** as a vector database for fast semantic search.
- 🐍 **Python** with `dotenv` for secure config.

---

## 💡 What does it do?

✅ Loads your personal data from `data.txt`  
✅ Generates embeddings for each chunk using OpenAI  
✅ Stores them in Pinecone so you can **instantly query by meaning**  
✅ When you ask a question, it finds the most relevant data & GPT generates an answer

---

## 🏗 Full project structure

```
rag-pinecone-v3/
├── env/               # Python virtual environment
├── .env               # Your API keys (never commit this)
├── .gitignore          # Ignores secrets & env files
├── requirements.txt    # Python dependencies
├── data.txt            # Your personal knowledge base
└── main.py             # Python script to run everything
```

---

## 🔒 .env (example)

```
OPENAI_API_KEY=sk-...
PINECONE_API_KEY=your-pinecone-key
PINECONE_INDEX=rag-knowledge-index-openai
```

✅ Place this file in your project root.  
✅ Keep it safe — never commit to git.

---

## 🚫 .gitignore (example)

```
.env
env/
__pycache__/
*.pyc
.DS_Store
```

This prevents committing sensitive files.

---

## 📚 requirements.txt

```
openai
pinecone
python-dotenv
```

Install with:

```
pip install -r requirements.txt
```

---

## 📝 data.txt (example CV data)

```
Shubham Tiwari is a Senior Frontend Engineer with over 8+ years of experience, based in London, UK. 
He specializes in React, Angular, Playwright, CI/CD, Docker, and building scalable UI architectures.

At Directline Group, he improved automated testing to 90% coverage, reducing bugs by 80%.
At Rackspace, he revamped legacy apps, improving performance by 70%, and built common Design Systems.
At EY, he built customer-facing apps that increased productivity by 90%.

He's experienced with Nx, Turborepo, GitHub Actions, Terraform, and is currently exploring LangChain, Pinecone, OpenAI, and RAG.
```

✅ Place this next to `main.py`.

---

## 🚀 How to run locally

### 1️⃣ Clone the repo & create your virtual environment

```bash
git clone https://github.com/yourusername/rag-pinecone-v3.git
cd rag-pinecone-v3

python3 -m venv env
source env/bin/activate
```

---

### 2️⃣ Install dependencies

```bash
pip install -r requirements.txt
```

---

### 3️⃣ Set up your `.env` file

```
OPENAI_API_KEY=sk-...
PINECONE_API_KEY=your-pinecone-key
PINECONE_INDEX=rag-knowledge-index-openai
```

---

### 4️⃣ Run the app

```bash
python3 main.py
```

✅ It will read your `data.txt`, embed & upload it to Pinecone, then start an interactive chat:

```
✅ Uploaded X documents into Pinecone index 'rag-knowledge-index-openai'.
Ask a question (or type 'exit'):
```

---

## 🔍 Example questions

```
What companies has Shubham worked for?
What testing frameworks does he use?
What is he exploring right now?
What cloud tools has he used?
exit
```

---

## 🔒 Security tips

✅ Your `.env` is ignored by git (thanks to `.gitignore`), keeping your API keys safe.  
✅ If you ever accidentally committed it, rotate your API keys immediately.

---

## 🚀 Next ideas to enhance

✅ Auto chunk long files to 500 characters with overlap  
✅ Load PDFs or multiple text files  
✅ Build a FastAPI backend or a React frontend for a portfolio chatbot

---

## 🖐 Need help?

Open an issue or connect on LinkedIn:  
[Shubham Tiwari](https://www.linkedin.com/in/shubhamtiwari-appdev/)

---

**Enjoy building your personal RAG assistant! 🚀**
