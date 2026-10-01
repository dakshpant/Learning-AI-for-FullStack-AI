import os
from groq import Groq
from pathlib import Path
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct


load_dotenv()

QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# ==========================================================
# Part 2 : connect to Qdrant and create a collection
# ==========================================================

client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
print("Connected to Qdrant cloud!")

# ==========================================================
# Part 3 : Create a collection in Qdrant
# ==========================================================

collection_name = "knowledge"
embedding_size = 384

# Delete connection if already exists
if client.collection_exists(collection_name):
    print(f"Deleting existing collection '{collection_name}'...")
    client.delete_collection(collection_name)

# create collection
client.create_collection(
    collection_name=collection_name,
    vectors_config=VectorParams(
        size=embedding_size,  # vector size
        distance=Distance.COSINE  # algoritm to use for vector similarity
    ),
)

# ==========================================================
# Part4 : Load knowledge into the DB
# ==========================================================

with open("knowledge.txt", "r", encoding="utf-8") as f:
    documents = [
        line.strip() for line in f if line.strip()
    ]
    print(f"Loaded {len(documents)} documents")


# ==========================================================
# Part 5 : create Embeddings
# ==========================================================
print('Loading Embedding model...')
model = SentenceTransformer("all-MiniLM-L6-v2")
print('Embedding model loaded!')

embeddings = model.encode(documents)
print(f"Created embeddings for {len(embeddings)} documents")
print(f'Embedding size: {len(embeddings[0])}')

# ==========================================================
# Part 6 : create qdrant points
# ==========================================================

points = []

for i, embedding in enumerate(embeddings):
    point = PointStruct(
        id=i,  # unique id for the point
        vector=embedding.tolist(),  # embedding vector
        payload={
            "text": documents[i]  # original text
        }
    )
    points.append(point)

# ==========================================================
# Part 7 : upload points to Qdrant
# ==========================================================

client.upsert(  # upload + insert = upsert
    collection_name=collection_name,
    points=points
)
print(
    f'Uploaded {len(points)} points to Qdrant collection "{collection_name}"')

# ==========================================================
# Part 8 : Search Qdrant
# ==========================================================


def search_qdrnt(query: str, top_k: 3):

    # Convert ques into embedding
    query_vector = model.encode(query).tolist()

    # search in qdrant for simiar vector
    results = client.query_points(
        collection_name=collection_name,
        query=query_vector,
        limit=top_k,
        with_payload=True
    ).points

    return results

# ==========================================================
# Part 9 : Test Search
# ==========================================================


query = "How many vacations days do i get"

results = search_qdrnt(query, top_k=3)

for result in results:
    print(f'Score: {result.score:.3f}')
    print(result.payload["text"])
    print()

# ==========================================================
# Part 10 connect to groq

groq_client = Groq(
    api_key=GROQ_API_KEY
)

# ==========================================================
# Part 11 : ask LLM
# ==========================================================


def ask_llm(question: str, context: str):
    prompt = f"""
    Answer the question using only the information provided below.
    Context:
    {context}

    Question:
    {question}

    If the answer is not contained within the context, respond with "I don't knowbased on the provided information.
    "
    """

    response = groq_client.chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=[
            {"role": "user", "content": prompt}
        ]
    )
    return response.choices[0].message.content


# ==========================================================
# Part 12 : Complete RAG pipeline
# ==========================================================

question = "How many vacations days do i get"
results = search_qdrnt(query, top_k=3)

# Extract text from the search results
context = "\n".join(
    result.payload["text"]
    for result in results
)


answer = ask_llm(question, context)


print("\nFinal Answer:")
print(answer)
