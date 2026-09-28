import os
from groq import Groq
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

my_api_key = os.getenv("GROQ_API_KEY")


if not my_api_key:
    raise ValueError(
        "GROQ_API_KEY environment variable is not set. Please set it in your .env file.")

client = Groq(api_key=my_api_key)
model = "openai/gpt-oss-120b"


# step 1 : create knowledge base in any firmat pdf, dictionart(Array of dict), text file, csv, etc. Here we are creating a knowledge base in the form of an array of dictionaries.

knowledge_base = {"age": "the age of virat kohli is 35 years",
                  "net worth": "the net worth of virat kohli is 125 million dollars",
                  "achievements": "virat kohli has won many awards including ICC cricketer of the year, ICC ODI player of the year, and many more"}

# step 2 : Retrieval
def retrieve_info(question: str):
    question_lower = question.lower()

    if "age" in question_lower:
        return knowledge_base["age"]
    elif "net worth" in question_lower:
        return knowledge_base["net worth"]
    elif "achievements" in question_lower:
        return knowledge_base["achievements"]
    else:
        return None


def ask_llm(question: str):
    system_prompt = f"""
    Answer in one line only. And answer only based on this context do not halucinate. Contect : {knowledge_base}"""

    system_message = {
        "role": "system",
        "content": system_prompt
    }
    message = {
        "role": "user",
        "content": question
    }
    messages = [system_message, message]
    response = client.chat.completions.create(model=model, messages=messages)
    ans = response.choices[0].message.content
    return ans


question = "how old is virat"


print(ask_llm(question))
