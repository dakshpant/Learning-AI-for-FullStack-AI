# RAG Evaluation System

This project implements a comprehensive evaluation framework for a Retrieval-Augmented Generation (RAG) pipeline. It uses a "LLM-as-a-Judge" approach to quantify the performance of both the retrieval and generation components.

## 🚀 System Architecture

The system is divided into two main phases: **Knowledge Indexing** and **Evaluation**.

### 1. Knowledge Indexing (The RAG Pipeline)
Before evaluation, the system sets up a searchable knowledge base:
- **Data Source**: Loads documents from `knowledge.json`.
- **Embedding**: Uses the `all-MiniLM-L6-v2` model from `sentence-transformers` to convert text into 384-dimensional vectors.
- **Vector Database**: Stores embeddings and metadata in **Qdrant**.
- **Retrieval**: A `search_qdrnt` function retrieves the top-K most similar document chunks based on cosine similarity.
- **Generation**: A Groq-powered LLM takes the retrieved context and a user question to generate a grounded answer.

### 2. Evaluation Framework
The system uses a **Golden Dataset** (a set of questions with ground-truth answers) to test the pipeline. For each question, it calculates five key metrics:

#### A. Retrieval Metrics (Measuring the Search)
- **Context Precision**: 
    - *Question*: "Is this retrieved chunk actually relevant to the question?"
    - *Calculation*: $\frac{\text{Relevant Chunks}}{\text{Total Retrieved Chunks}}$
- **Context Recall**: 
    - *Question*: "Does the retrieved context contain all the info needed to produce the ground truth answer?"
    - *Calculation*: LLM assigns a score from 0.0 to 1.0 based on information coverage.

#### B. Generation Metrics (Measuring the Answer)
- **Faithfulness (Groundedness)**: 
    - *Question*: "Is the generated answer derived solely from the retrieved context, or did the LLM hallucinate?"
    - *Calculation*: LLM checks if every claim in the answer is supported by the context.
- **Answer Relevancy**: 
    - *Question*: "Does the answer actually address the user's question?"
    - *Calculation*: LLM scores how directly the answer addresses the prompt.
- **Answer Correctness**: 
    - *Question*: "Is the generated answer factually correct compared to the ground truth?"
    - *Calculation*: LLM compares the generated response with the known correct answer.

## 🛠️ Working Flow

1. **Initialize**: Connects to Qdrant, creates a collection, and indexes the `knowledge.json` data.
2. **Iterate**: For every entry in the `golden_dataset`:
   - **Retrieve**: Get top-3 chunks from Qdrant.
   - **Generate**: Generate an answer using the LLM.
   - **Judge**: Pass the (Question, Context, Answer, Ground Truth) to the `llm_judge` to calculate the 5 metrics.
   - **Diagnose**: Provide immediate feedback (e.g., "Retrieval is missing important information") if any score falls below 0.7.
3. **Summarize**: Calculate and print the average score for each metric across the entire dataset.

## ⚙️ Setup

### Environment Variables
Create a `.env` file with:
```env
QDRANT_URL=your_qdrant_url
QDRANT_API_KEY=your_qdrant_api_key
GROQ_API_KEY=your_groq_api_key
```

### Running the Evaluation
```bash
python eval.py
```

## 📊 Summary of Metrics

| Metric | Component | Evaluates | Goal |
| :--- | :--- | :--- | :--- |
| **Precision** | Retrieval | Noise vs. Signal | Minimize irrelevant chunks |
| **Recall** | Retrieval | Coverage | Ensure all facts are found |
| **Faithfulness** | Generation | Hallucinations | Answer must stay in context |
| **Relevancy** | Generation | Utility | Answer must be helpful |
| **Correctness** | Generation | Accuracy | Answer must match truth |
