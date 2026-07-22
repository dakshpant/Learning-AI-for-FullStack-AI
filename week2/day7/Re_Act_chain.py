from httpx import Response
from groq import Client
import os
import re
from time import sleep
from pathlib import Path
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

my_api_key = os.getenv("GROQ_API_KEY")
if not my_api_key: 
    raise ValueError("GROQ_API_KEY not found in environment variables")

client = Groq(api_key=my_api_key)
model = "llama-3.3-70b-versatile"

#tools
def get_product_price(product):
    if product == 'Iphone 17':
        return 1000
    elif product == "Iphone 15":
        return 500
    else: return 100

def calculator(expression):
    try:
        return eval(expression)
    except:
        return "Calcultation error"

tools = {
    "get_product_price": get_product_price,
    "calculator": calculator,
    "calculate": calculator
}

system_prompt = """
You are a shopping assistant.
You have these tools:
get_product_price(product)
calculator(expression)

IMPORTANT: call tools exactly like these expamples:

Action: get_product_price("Iphone 17")
Action: calculator("5000-1000")

Never Write:
get_product_price(product = "Iphone 17")

Follow these rules:

1. Decide what you need to do next.
2. Call ONLY ONE tool at a time.
3. After writing an Action, STOP immediately.
4. Never guess or invent a tool result.
5. Wait until you receive an Observation.
6. Then decide your next action.
7. When the task is complete, give the Final Answer.

Format:

Thought: what you need to do
Action: tool_name(argument)

When finished:

Final Answer: your answer
 """

def run_agent(question):
    messages = [
        {
            "role" : "system",
            "content":system_prompt
        },
        {
            "role": "user",
            "content" : question
        }
    ]

    for steps in range(5):
        print("\n---------------------------")
        print("Step", steps+1)
        print("\n---------------------------")

        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0
        )
        answer = response.choices[0].message.content
        
        print(answer)

        #Agent has finished
        if "Final Answer:" in answer:
            break

        # Find the Action
        match = re.search(
            r"Action:\s*(\w+)\((.*?)\)",
            answer
        )
        
        observation = "No action found"
        if match:
            tool_name = match.group(1)
            tool_input = match.group(2)
            tool_input = tool_input.strip()
            tool_input = tool_input.strip('"')

            # Run the tool
            if tool_name in tools:
                tool = tools[tool_name]
                observation = tool(tool_input)
            else:
                observation = f"Tool '{tool_name}' not found"

        print(
            "Observation:",
            observation
        )

        # Add LLM response to memory
        messages.append({"role": "assistant", "content": answer})

        # Give tool result back to LLM
        messages.append({
            "role": "user",
            "content": "Observation: " + str(observation)
        })
        sleep(5)

prompt="""
I have 5000 rupees. What is the price of an iphone 17?
and how much money will I have left?
"""
run_agent(prompt)