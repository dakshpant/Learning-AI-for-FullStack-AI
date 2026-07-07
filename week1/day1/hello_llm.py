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

prompt = "DO You Know Daksh Pant?"

message = {
    "role" : role,
    "content" : prompt
}

messages = [message]

response = client.chat.completions.create(
    model=model,
    messages=messages
)

print(response.choices[0].message.content)