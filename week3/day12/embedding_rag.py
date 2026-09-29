import os
import numpy as np
import json
from pathlib import Path
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

load_dotenv()


def cosine_similarity(vec1, vec2):
    return np.dot(vec1, vec2)/(np.linalg.norm(vec1) * np.linalg.norm(vec2))


model = SentenceTransformer('all-MiniLM-L6-v2')  # 384 vector length model

text = "Machine Learning Is fun."

# embedding = model.encode(text)

# print(embedding.shape)
# print(embedding[:10])  # print first 10 values of the embedding vector


t1 = "There are 24 paid leaves"
t2 = "i am a human"

v1 = model.encode(t1)
v2 = model.encode(t2)

# should be close to 1.0
print(cosine_similarity(v1, v2))
