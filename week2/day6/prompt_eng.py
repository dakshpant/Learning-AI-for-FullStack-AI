from httpx import Response
from groq.types import model
from groq import Client
import os
from pathlib import Path
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

my_api_key = os.getenv("GROQ_API_KEY")
if not my_api_key: 
    raise ValueError("GROQ_API_KEY not found in environment variables")

client = Groq(api_key=my_api_key)
model = "llama-3.3-70b-versatile"

def llm_ans(prompt):
    message={
        "role":"user",
        "content":prompt
    }
    messages = [message]
    response = client.chat.completions.create(model=model, messages= messages)
    return response.choices[0].message.content

#Bag Prompt

# bad_prompt = """
# This is a user complaint:
# My Laptop is not working
# classify this
# """

# Optimized Prompt

goog_prompt = """
#Role:
You are a support assistant at a mobile/laptop company
#Task:
You have to classify the issue in a category

#Constraints:
You have to classify the issue in one of the three categories,
mainly Billing, Technical and Return 

#Output Format:
Your Answer should be in one word only, the one word should be one of the categories given in the above constraints

#Example:
For instance if user complaint says thet he/she wants a refund then the category is Return.

#Fallback
If the issue is unrellated to any of the categories given in the constraints the return Unrelated or others.

This is a user complaint:
Bhai is my mobile charger compatible with this new mobile?
"""

print(llm_ans(goog_prompt))