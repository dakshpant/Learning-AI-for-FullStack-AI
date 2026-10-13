import os
import json
from dotenv import load_dotenv
from groq import Groq
from tavily import TavilyClient

# Load environment Variables
load_dotenv()

groq = Groq(api_key=os.getenv("GROQ_API_KEY"))
tavily = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))

# =========================
# Tool 1 => WEB SEARCH
# =========================

def search_web(query: str) -> str:
    """ Search the for the current information. """
    try:
        response = tavily.search(
            query=query,
            max_results=3,
            search_depth="advanced"
        )
        results = []
        for result in response["results"]:
            results.append(
                f"Title:{result['title']}\n"
                f"URL: {result['url']}\n"
                f"Content: {result['content']}"
            )

        return "\n\n".join(results)
    except Exception as e:
        return f"Search failed:{e}"


# =========================
# Tool 2 => Calculator
# =========================
def calculator(expression: str) -> str:
    """
    Perform a mathematical Calculation"""
    try:
        # FIXED: Corrected typo from __bultins__ to __builtins__ to ensure eval works correctly
        return str(eval(
            expression,
            {"__builtins__": {}},
            {}
        ))
    except Exception as e:
        return f"Calculation failed: {e}"


# =========================
# Tool registring => registring available tools
# =========================
available_tools = {
    "web_search": search_web,
    "calculate": calculator
}

# ============================================================
# TOOL DEFINITIONS FOR THE LLM
# ============================================================
tools = [
    # defined 1st Tool => web search
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": (
                "Search the web for current information, "
                "news, recent events, prices, statistics,"
                "or facts that may have changed."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "A specific and concise web search query."
                    }
                },
                "required": ["query"]
            }
        }
    },
    # defined 2nd Tool = >Calculator
    {
        "type": "function",
        "function": {
            "name": "calculate",
            "description": (
                "Perform mathematical calculations "
                "such as percentages, multiplication, "
                "division, or arithmetic."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "A mathematical expression such as '2500 * 0.15'."
                    }
                },
                "required": ["expression"]  # FIXED: Moved inside parameters to comply with API schema
            }
        }
    }
]

# ===================
# Agent Instruction
# ===================
SYSTEM_PROMPT = """
You are a research assistant.

Rules:

1. Use web_search for current or changing information.
2. Use calculate for mathematical calculations.
3. You may use tools multiple times.
4. When you have enough information, answer the user.
5. Include useful source URLs from web search results.
"""


# ============================================================
# AGENT LOOP
# ============================================================
def run_agent(question: str, max_iterations: int = 5):

    messages = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT
        },
        {
            "role": "user",
            "content": question
        }
    ]

    for iteration in range(max_iterations):
        print(f"\nIteration {iteration + 1} ----")

        response = groq.chat.completions.create(
            model="openai/gpt-oss-120b",
            messages=messages,
            tools=tools,
            tool_choice="auto"
        )

        message = response.choices[0].message
        messages.append(message)

        if not message.tool_calls:
            print("\nFINAL ANSWER:")
            print(message.content)
            return message.content

        print(f"LLM required {len(message.tool_calls)} tool(s)")
        for tool_call in message.tool_calls:
            tool_name = tool_call.function.name
            arguments = json.loads(tool_call.function.arguments)
            
            print(f"Tool: {tool_name} | Arguments: {arguments}")
            
            if tool_name not in available_tools:
                result = f"Unknown tool: {tool_name}"
            else:
                result = available_tools[tool_name](**arguments)
                print(f"Tool result received: {result[:100]}...")

            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": result
            })

    print("\nMaximum iterations reached.")
    return "I couldn't complete the task within the maximum iterations."


# ============================================================
# RUN THE AGENT
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("AI RESEARCH AGENT")
    print("=" * 60)

    question = input("\nAsk me anything: ")
    if question.strip():
        run_agent(question)
    else:
        print("Please enter a valid question.")
