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
# 3 prompts

prompt_1 = "Hi!"
prompt_2 = "Explain The general Theory Of relativity in detail under 100 words"
prompt_3 = "Write a 1000 words essay on Machine Learning"

prompts = [prompt_1, prompt_3, prompt_2]
for prompt in prompts:
    message = {
    "role" : role,
    "content" : prompt
    }
    messages = [message]
    response = client.chat.completions.create(model=model,messages=messages, max_tokens=50)
    # print(response.choices[0].message.content)
    usage = response.usage
    print(f"Prompt:{prompt} --> Your_Tokens:{usage.prompt_tokens} , Complition_Tokens:{usage.completion_tokens}, Total_Tokens : {usage.total_tokens} --> Finish_reason: {response.choices[0].finish_reason}")
    print("----------------------------------------------------------------------------------------------------\n")