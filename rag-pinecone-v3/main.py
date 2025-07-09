import os
from dotenv import load_dotenv
from openai import OpenAI
from pinecone import Pinecone

# Load environment variables
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
pinecone_api_key = os.getenv("PINECONE_API_KEY")
pinecone_index_name = os.getenv("PINECONE_INDEX")

if not (openai_api_key and pinecone_api_key and pinecone_index_name):
    raise ValueError("Missing .env configuration. Check OPENAI_API_KEY, PINECONE_API_KEY, and PINECONE_INDEX.")

# Init clients
client = OpenAI(api_key=openai_api_key)
pc = Pinecone(api_key=pinecone_api_key)
index = pc.Index(pinecone_index_name)

# 🔄 Load data from data.txt
file_path = "data.txt"
if not os.path.exists(file_path):
    raise FileNotFoundError(f"{file_path} does not exist. Please add your text data.")

with open(file_path, "r", encoding="utf-8") as f:
    content = f.read()

# Split into lines (or you can do paragraphs, etc.)
documents = [line.strip() for line in content.split("\n") if line.strip()]

# 🔄 Embed and upsert
for i, doc in enumerate(documents):
    embedding = client.embeddings.create(
        model="text-embedding-3-small",
        input=doc
    ).data[0].embedding
    index.upsert([(f"doc-{i}", embedding, {"text": doc})])

print(f"✅ Uploaded {len(documents)} documents into Pinecone index '{pinecone_index_name}'.")

# 🔍 Query loop
while True:
    query = input("\nAsk a question (or type 'exit'): ")
    if query.lower() == "exit":
        break

    q_embed = client.embeddings.create(
        model="text-embedding-3-small",
        input=query
    ).data[0].embedding

    res = index.query(vector=q_embed, top_k=3, include_metadata=True)
    contexts = [match["metadata"]["text"] for match in res["matches"]]

    print("\nTop related contexts:")
    for ctx in contexts:
        print("-", ctx)

    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "user", "content": 
             f"Use the following context to answer the question:\n\n{contexts}\n\nQuestion: {query}"}
        ]
    )
    print("\n🤖 Answer:", response.choices[0].message.content)
