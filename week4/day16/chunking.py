from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
    CharacterTextSplitter
)

text = """
Retrieval-Augmented Generation, commonly known as RAG, is a technique that combines information retrieval with large language models. Instead of relying entirely on the knowledge learned during training, a RAG system retrieves relevant information from an external knowledge source before generating a response. This approach helps make AI applications more accurate, context-aware, and useful for domain-specific questions.

For example, imagine building an AI assistant that answers questions about a company's internal documentation. Without RAG, the language model might not know anything about the company's internal policies. With RAG, the system can search through company documents, retrieve relevant passages, and provide an answer based on the retrieved information.
"""


# Fixed Size chunkig

fixed = CharacterTextSplitter(
    separator="",
    chunk_size=100,
    chunk_overlap=0
)
print("\n==========Fixed Size================")

for i, chunk in enumerate(fixed.split_text(text)):
    print(f"Chunk {i}: \n")
    print(chunk)
    print("---------------------")

#Paragraph chunking


para = CharacterTextSplitter(
    separator="\n\n",
    chunk_size=100,
    chunk_overlap=0
)
print("\n==========paragraph================")

for i, chunk in enumerate(para.split_text(text)):
    print(f"Chunk {i}: \n")
    print(chunk)
    print("---------------------")


#Recursive Chinlig

recursive= RecursiveCharacterTextSplitter(
    chunk_size=100,
    chunk_overlap=20
)
print("\n==========Recursive================")

for i, chunk in enumerate(recursive.split_text(text)):
    print(f"Chunk {i}: \n")
    print(chunk)
    print("---------------------")