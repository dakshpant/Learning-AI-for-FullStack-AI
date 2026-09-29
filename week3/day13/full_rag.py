import os
import json
from groq import Groq
from pathlib import Path
from dotenv import load_dotenv

import numpy as np
from sentence_transformers import SentenceTransformer

# load env
load_dotenv()

# choose and craeate embedding model
embedding_model = SentenceTransformer(
    'all-MiniLM-L6-v2')  # 384 vector length model

# Choose and create groq model (LLM) and create client
groq_model = "openai/gpt-oss-120b"

my_api_key = os.getenv("GROQ_API_KEY")

if not my_api_key:
    raise ValueError(
        "GROQ_API_KEY environment variable is not set. Please set it in your .env file.")

client = Groq(api_key=my_api_key)  # client created


# knowledge base
knowledge_base = [
    "Employees receive 24 days of paid leave per year.",

    "Employees work from the office on Tuesday, Wednesday and Thursday. "
    "Monday and Friday are optional work-from-home days.",

    "Employees receive Rs 3000 per month for gym reimbursement.",

    "Employees can claim Rs 2000 per month for home internet.",

    "Employees have a 90 day notice period."
]
# for the knowledge base, create embeddings of each knowledge(here in our case each line)
knowledge_base_embeddings = embedding_model.encode(knowledge_base)

# embedding cosine similarity function


def cosine_similarity(a, b):
    return np.dot(a, b)/np.linalg.norm(a) * np.linalg.norm(b)

# retrieve the most relevant knowledge from the knowledge base based on the query
# here we compare the query with each index of the knowledge base and generate the cosine similarity
# and returning the index and the index of the highest similarity score


def retrieve(query_embedding):
    scores = []
    # enumerate returns a tuple of index and value
    for i, knowledge_embedding in enumerate(knowledge_base_embeddings):
        similarity = cosine_similarity(query_embedding, knowledge_embedding)
        scores.append((similarity, knowledge_base[i]))
    scores.sort(reverse=True)
    return scores[0]  # 0.9, index(line )

# LLM function to complete the full RAG


def ask_llm(question: str, context):
    system_prompt = f"""
    answer in one line only based on the context and do not halucinate. Context : {context} 
    """  # context injection
    system_message = {
        "role": "system",
        "content": system_prompt
    }
    message = {
        "role": "user",
        "content": question
    }
    messages = [system_message, message]
    response = client.chat.completions.create(
        model=groq_model, messages=messages)
    ans = response.choices[0].message.content
    return ans


# get users query or question and create embedding for the query
query = "How much vacation do i get?"
query_embedding = embedding_model.encode(query)


# calculate cosine similarity between query embedding and knowledge base embeddings
# function written above retrieve()

# getting the final context wing the retrieve function
score, context = retrieve(query_embedding)
# print(score)
# print(context)
# context here is the string from the knowledge base
answer = ask_llm(query, context)
print(answer)
