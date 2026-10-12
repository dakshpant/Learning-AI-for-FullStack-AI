import os
import json
from groq import Groq
from pathlib import Path
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import (Distance, VectorParams, PointStruct,
                                  PayloadSchemaType)


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

collection_name = "rag_eval"
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


# Adding Filter at category level to filter out the results based on category
print(f"Created collection: {collection_name}")
print(f"Vector size: {embedding_size}")
print("Distance: COSINE")
client.create_payload_index(
    collection_name=collection_name,
    field_name="category",
    field_schema=PayloadSchemaType.KEYWORD,
)

# ==========================================================
# Part4 : Load knowledge into the DB
# ==========================================================

with open("knowledge.json", "r", encoding="utf-8") as f:
    documents = json.load(f)

print(
    f"Loaded {len(documents)} knowledge documents."
)
# ==========================================================
# Part 5 : create Embeddings
# ==========================================================
print('Loading Embedding model...')
model = SentenceTransformer("all-MiniLM-L6-v2")
print('Embedding model ready!')

texts = [document["text"] for document in documents]

embeddings = model.encode(texts)

print(f"Created embeddings for {len(embeddings)} documents")
print(f'Embedding size: {len(embeddings[0])}')

# ==========================================================
# Part 6 : create qdrant points
# ==========================================================

points = []

for i, embedding in enumerate(embeddings):
    point = PointStruct(
        id=i+1,  # unique id for the point
        vector=embedding.tolist(),  # embedding vector
        payload=documents[i]
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

# New Search with filrer ans paramter

# def search_with_filter(query: str, query_filter: None, top_k: 3):
#     query_vector = model.encode(query).tolist()
#     results = client.query_points(
#         collection_name=collection_name,
#         query=query_vector,
#         limit=top_k,
#         with_payload=True,
#         query_filter=query_filter
#     ).points
#     return results


# # Creating filter for category
# reimbursement_filter = Filter(
#     must=[
#         FieldCondition(
#             key="category",
#             match=MatchValue(value="reimbursement")
#         ),
#     ]
# )

# # ==========================================================
# # Part 9 : Test Search
# # ==========================================================


# query = "How many vacations days do i get"

# results = search_qdrnt(query, top_k=3)

# for result in results:
#     print(f'Score: {result.score:.3f}')
#     print(result.payload["text"])
#     print()

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
# Part 12 : Golden Dataset
# ==========================================================

golden_dataset = [

    {
        "id": 1,

        "question":
            "How many vacation days do I get?",

        "ground_truth":
            "Employees in India receive 24 days of paid leave per year.",

        "expected_information":
            "The retrieved context should contain information that employees receive 24 days of paid leave per year."
    },


    {
        "id": 2,

        "question":
            "What is the work from home policy?",

        "ground_truth":
            "Employees work from the office three days every week on Tuesday, Wednesday and Thursday.",

        "expected_information":
            "The retrieved context should contain the hybrid work policy and the three office days: Tuesday, Wednesday and Thursday."
    },


    {
        "id": 3,

        "question":
            "How much internet reimbursement do employees receive?",

        "ground_truth":
            "Employees receive Rs 2000 per month for home internet reimbursement.",

        "expected_information":
            "The retrieved context should contain the Rs 2000 monthly internet reimbursement."
    },


    {
        "id": 4,

        "question":
            "When do promotions happen?",

        "ground_truth":
            "Promotions happen twice every year during March and September.",

        "expected_information":
            "The retrieved context should contain that promotions happen in March and September."
    },


    {
        "id": 5,

        "question":
            "What is the gym benefit?",

        "ground_truth":
            "Employees receive Rs 3000 every month for gym memberships or fitness apps.",

        "expected_information":
            "The retrieved context should contain the Rs 3000 monthly wellness benefit for gym or fitness apps."
    }
]

# ===================
# LLM as a Judge
# ===================


def llm_judge(prompt):

    response = groq_client.chat.completions.create(

        model="openai/gpt-oss-120b",

        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],

        response_format={
            "type": "json_object"
        },

        temperature=0
    )

    return json.loads(
        response.choices[0].message.content
    )


# =====================
# Precision
# For every retrieved chunk, ask:
#
# "Is this chunk relevant to the question?"
#
# Precision =
#
# relevant retrieved chunks
# -------------------------
# total retrieved chunks
# =====================
def context_precision(
    question,
    retrieved_docs
):

    relevant_chunks = 0

    for i, doc in enumerate(
        retrieved_docs
    ):

        chunk = doc.payload["text"]

        prompt = f"""
You are evaluating the retrieval quality
of a RAG system.

Question:
{question}

Retrieved chunk:
{chunk}

Is this chunk relevant to answering
the question?

Return ONLY JSON:

{{
    "relevant": true,
    "reason": "short explanation"
}}

Return true if the chunk contains information
that is useful for answering the question.

Return false if it is unrelated.
"""

        result = llm_judge(prompt)

        if result["relevant"]:

            relevant_chunks += 1

    if len(retrieved_docs) == 0:

        return 0.0

    return (
        relevant_chunks
        /
        len(retrieved_docs)
    )


# ===============
# Recall
# ===============
def context_recall(
    question,
    context,
    ground_truth
):

    prompt = f"""
You are evaluating the retrieval quality
of a RAG system.

Question:
{question}

Ground Truth Answer:
{ground_truth}

Retrieved Context:
{context}

Does the retrieved context contain enough
information to produce the ground truth answer?

Return ONLY JSON:

{{
    "score": 0.0,
    "reason": "short explanation"
}}

Scoring:

1.0 = All important information needed for
      the answer is present.

0.7 = Most important information is present,
      but some details are missing.

0.5 = Some important information is present.

0.0 = The required information is absent.
"""

    result = llm_judge(
        prompt
    )

    return result

# ============================================================
# PART 17 — FAITHFULNESS
#
# Is the generated answer supported by
# the retrieved context?
# ============================================================


def evaluate_faithfulness(
    question,
    context,
    answer
):

    prompt = f"""
You are evaluating a RAG system.

Question:
{question}

Retrieved Context:
{context}

Generated Answer:
{answer}

Determine whether the claims in the
generated answer are supported by
the retrieved context.

Return ONLY JSON:

{{
    "score": 0.0,
    "reason": "short explanation"
}}

Scoring:

1.0 = All claims are supported.

0.7 = Mostly supported with minor issues.

0.5 = Some claims are supported.

0.0 = Unsupported or contradictory.
"""

    return llm_judge(
        prompt
    )


# ============================================================
# PART 18 — ANSWER RELEVANCY
# ============================================================

def evaluate_relevancy(
    question,
    answer
):

    prompt = f"""
You are evaluating a RAG system.

Question:
{question}

Generated Answer:
{answer}

Does the generated answer actually
answer the question?

Return ONLY JSON:

{{
    "score": 0.0,
    "reason": "short explanation"
}}

Scoring:

1.0 = Directly answers the question.

0.7 = Mostly answers the question.

0.5 = Partially answers the question.

0.0 = Completely off-topic.
"""

    return llm_judge(
        prompt
    )


# ============================================================
# PART 19 — ANSWER CORRECTNESS
# ============================================================

def evaluate_correctness(
    answer,
    ground_truth
):

    prompt = f"""
You are evaluating a RAG system.

Generated Answer:
{answer}

Ground Truth Answer:
{ground_truth}

Determine whether the generated answer
is factually correct compared with
the ground truth.

Return ONLY JSON:

{{
    "score": 0.0,
    "reason": "short explanation"
}}

Scoring:

1.0 = Completely correct.

0.7 = Mostly correct with minor omissions.

0.5 = Partially correct.

0.0 = Incorrect or contradictory.
"""

    return llm_judge(
        prompt
    )


# ============================================================
# PART 20 — RUN COMPLETE EVALUATION
# ============================================================

def run_evaluation():

    print("\n")

    print("=" * 70)

    print(
        "                    RAG EVALUATION"
    )

    print("=" * 70)

    all_scores = {

        "precision": [],

        "recall": [],

        "faithfulness": [],

        "relevancy": [],

        "correctness": []
    }

    # ========================================================
    # RUN EVERY GOLDEN QUESTION THROUGH THE REAL RAG
    # ========================================================

    for test in golden_dataset:

        question = test["question"]

        print("\n")

        print("-" * 70)

        print(
            f"QUESTION: {question}"
        )

        # ----------------------------------------------------
        # RETRIEVAL
        # ----------------------------------------------------

        results = search_qdrnt(
            question,
            top_k=3
        )

        # --------------------------------------------i--------
        # DISPLAY RETRIEVED DOCUMENTS
        # ----------------------------------------------------

        print("\nRetrieved Documents:")

        for i, result in enumerate(
            results
        ):

            print(
                f"\nChunk {i + 1}"
            )

            print(
                f"Score: {result.score:.3f}"
            )

            print(
                result.payload["text"]
            )

        # ----------------------------------------------------
        # CREATE CONTEXT
        # ----------------------------------------------------

        context = "\n".join(

            result.payload["text"]

            for result in results
        )

        # ----------------------------------------------------
        # GENERATE ANSWER
        # ----------------------------------------------------

        answer = ask_llm(
            question,
            context
        )

        print("\nGenerated Answer:")

        print(answer)

        # ====================================================
        # METRIC 1 — CONTEXT PRECISION
        # ====================================================

        precision = context_precision(

            question,

            results
        )

        # ====================================================
        # METRIC 2 — CONTEXT RECALL
        # ====================================================

        recall_result = context_recall(

            question,

            context,

            test["ground_truth"]
        )

        recall = float(
            recall_result["score"]
        )

        # ====================================================
        # METRIC 3 — FAITHFULNESS
        # ====================================================

        faithfulness_result = (
            evaluate_faithfulness(

                question,

                context,

                answer
            )
        )

        faithfulness = float(
            faithfulness_result["score"]
        )

        # ====================================================
        # METRIC 4 — ANSWER RELEVANCY
        # ====================================================

        relevancy_result = evaluate_relevancy(

            question,

            answer
        )

        relevancy = float(
            relevancy_result["score"]
        )

        # ====================================================
        # METRIC 5 — ANSWER CORRECTNESS
        # ====================================================

        correctness_result = (
            evaluate_correctness(

                answer,

                test["ground_truth"]
            )
        )

        correctness = float(
            correctness_result["score"]
        )

        # ====================================================
        # SAVE SCORES
        # ====================================================

        all_scores["precision"].append(
            precision
        )

        all_scores["recall"].append(
            recall
        )

        all_scores["faithfulness"].append(
            faithfulness
        )

        all_scores["relevancy"].append(
            relevancy
        )

        all_scores["correctness"].append(
            correctness
        )

        # ====================================================
        # PRINT SCORES
        # ====================================================

        print("\nScores")

        print(
            f"Context Precision : {precision:.2f}"
        )

        print(
            f"Context Recall    : {recall:.2f}"
        )

        print(
            f"Faithfulness      : {faithfulness:.2f}"
        )

        print(
            f"Answer Relevancy  : {relevancy:.2f}"
        )

        print(
            f"Answer Correctness: {correctness:.2f}"
        )

        # ====================================================
        # DIAGNOSIS
        # ====================================================

        print("\nDiagnosis")

        if precision < 0.7:

            print(
                "❌ Retrieval is returning irrelevant chunks."
            )

        if recall < 0.7:

            print(
                "❌ Retrieval is missing important information."
            )

        if faithfulness < 0.7:

            print(
                "❌ Answer is not properly grounded."
            )

        if relevancy < 0.7:

            print(
                "❌ Answer is not sufficiently relevant."
            )

        if correctness < 0.7:

            print(
                "❌ Answer is not sufficiently correct."
            )

        if (
            precision >= 0.7
            and recall >= 0.7
            and faithfulness >= 0.7
            and relevancy >= 0.7
            and correctness >= 0.7
        ):

            print(
                "✅ Good RAG response."
            )

    # ========================================================
    # FINAL SUMMARY
    # ========================================================

    print("\n")

    print("=" * 70)

    print(
        "                    FINAL SUMMARY"
    )

    print("=" * 70)

    for metric, values in all_scores.items():

        average = (
            sum(values)
            /
            len(values)
        )

        print(
            f"{metric.capitalize():20}: {average:.2f}"
        )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    run_evaluation()
