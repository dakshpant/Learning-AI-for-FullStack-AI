import os
from typing import List, Union
from dotenv import load_dotenv
from groq import Groq
from tavily import TavilyClient

# Load environment variables
load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

# Initialize clients
groq_client = Groq(api_key=GROQ_API_KEY)
tavily_client = TavilyClient(api_key=TAVILY_API_KEY)


def web_search(query: str) -> str:
    """
    Performs a web search using Tavily to find real-time information.
    """
    print(f"Executing web_search with query: {query}")
    response = tavily_client.search(query=query, search_depth="basic")
    results = [obj['content'] for obj in response['results']]
    return "\n".join(results)


def calculate(expression: str) -> str:
    """
    Evaluates a basic mathematical expression.
    """
    print(f"Executing calculate with expression: {expression}")
    try:
        # Use eval safely for basic math expressions (in a production app, use a proper parser)
        # Allowed characters: numbers, operators, parentheses, spaces
        allowed_chars = "0123456789+-*/(). "
        if any(char not in allowed_chars for char in expression):
            return "Error: Invalid characters in expression."

        result = eval(expression)
        return str(result)
    except Exception as e:
        return f"Error calculating expression: {str(e)}"


# Tool definitions for Groq (OpenAI format)
tools = [
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Search the web for current information or specific questions",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query to look up on the web",
                    }
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calculate",
            "description": "Perform basic mathematical calculations",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "The math expression to evaluate, e.g., '2+2' or '10*5'",
                    }
                },
                "required": ["expression"],
            },
        },
    },
]


def run_agent(user_prompt: str, max_iterations: int = 3):
    # Initial call to LLM to decide which tool to use
    messages = [
        {"role": "system", "content": "You are a helpful assistant. Whenever you are asked about a person, a current event, or a specific fact that isn't common knowledge, ALWAYS use the web_search tool to get the most accurate and up-to-date information. Use the calculate tool for any math operations."},
        {"role": "user", "content": user_prompt},
    ]

    iteration = 0
    while iteration < max_iterations:
        iteration += 1

        response = groq_client.chat.completions.create(
            model="openai/gpt-oss-120b",
            messages=messages,
            tools=tools,
            tool_choice="auto",
        )

        response_message = response.choices[0].message
        tool_calls = response_message.tool_calls

        if not tool_calls:
            return response_message.content

        # The LLM wants to call one or more tools
        messages.append(response_message)

        for tool_call in tool_calls:
            function_name = tool_call.function.name
            # Safely parse JSON args instead of eval
            import json
            try:
                function_args = json.loads(tool_call.function.arguments)
            except json.JSONDecodeError:
                function_args = {}

            if function_name == "web_search":
                tool_output = web_search(function_args.get("query", ""))
            elif function_name == "calculate":
                tool_output = calculate(function_args.get("expression", ""))
            else:
                tool_output = "Tool not found"

            messages.append({
                "tool_call_id": tool_call.id,
                "role": "tool",
                "name": function_name,
                "content": tool_output,
            })

    return "I'm sorry, I reached the maximum number of tool iterations without finding a final answer."


if __name__ == "__main__":
    try:
        q = input("Enter your question: ")
        if q.strip():
            print("Thinking...")
            response = run_agent(q)
            print(f"\nAnswer: {response}")
        else:
            print("No question provided.")
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f"An error occurred: {e}")
