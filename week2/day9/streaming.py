import os
from pathlib import Path
from dotenv import load_dotenv
from groq import Groq

load_dotenv()
my_api_key = os.getenv("GROQ_API_KEY")
if not my_api_key:
    raise ValueError("API Key is missing")

client = Groq(api_key=my_api_key)

model = "openai/gpt-oss-120b"

role = "user"

prompt = "Explain How internet works."

message = {
    "role": role,
    "content": prompt
}

messages = [message]

# without streaming enabled this is default behavior, the entire response is returned at once
# response = client.chat.completions.create(
#     model=model,
#     messages=messages
# )
# print(response)
# answer = response.choices[0].message.content
# print(answer)

# with streaming enabled, the response is returned in chunks as they are generated
streamResponse = client.chat.completions.create(
    model=model,
    messages=messages,
    stream=True
)

for chunk in streamResponse:
    content = chunk.choices[0].delta.content
    if content:
        print(content, end="", flush = True)
