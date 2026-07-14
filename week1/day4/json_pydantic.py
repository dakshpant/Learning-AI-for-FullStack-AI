from multiprocessing.connection import answer_challenge
import email
import string
import os
from pathlib import Path
from dotenv import load_dotenv
from groq import Groq

load_dotenv()
my_api_key = os.getenv("GROQ_API_KEY")
if not my_api_key:
    raise ValueError("API Key is missing")

client = Groq(api_key=my_api_key)

model = "llama-3.3-70b-versatile"

role = "user"

# structured Output => using pydantic
from pydantic import BaseModel
class Ticket(BaseModel):
    name : str
    email : str
    issue : str

schema = Ticket.model_json_schema()

response_format = {
    "type" : "json_object"
}
system_prompt = f"""
Extract the personal information strictly based on this schema and give me a JSON output. {schema} 
"""

message_system = {
    "role" : "system",
    "content" : system_prompt
}

text = "Hello My name is Daksh Pant. Kal kuch asa hua ki I hae an Iphone which has stopped working. My address is Delhi 6, Neecha paan ki dunak uppar july ka maakan. My email is test@test.com. My contact number is 9219190239."

prompt = f"""
This is a customer ticket please extract personal information form this {text}.
 """


message = {
    "role" : role,
    "content" : prompt
}

messages = [message_system,message]

response = client.chat.completions.create(
    model=model,
    messages=messages,
    response_format = response_format
)

print("================================================")
print("JSON String Returned by the LLM")
print("================================================")
answer = response.choices[0].message.content
print(answer)

# Isko phadte ksa hae
import json
raw_json = answer

data_file = json.loads(raw_json)
tickets = Ticket(**data_file)

print("\n\n================================================")
print("Accessing Data from the Pydantic Model")
print("================================================")
print(tickets.name)
print(tickets.email)
print(tickets.issue)
