import random
import operator
import os
import difflib
import sys
from typing import Annotated, TypedDict, Union, List, Literal, Optional

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, END, START
from langgraph.graph.message import add_messages
from langchain_core.runnables import RunnableConfig

# Load environment variables
load_dotenv()

# Fix console encoding for fancy quotes
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# --- Menu Definition ---
MENU = {
    "burger": 10,
    "pizza": 5,
    "pasta": 0,
    "salad": 20
}

# Constants
ORDER_RETRIES_MAX = 3
COOK_RETRIES_MAX = 2
SERVE_RETRIES_MAX = 2
COOK_SUCCESS_PROB = 0.6
SERVE_SUCCESS_PROB = 0.6

# --- State Definition ---

class OrderDetails(TypedDict):
    dish_name: str
    required_quantity: int
    available_quantity: int

class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]
    order: OrderDetails
    status: Literal["new", "awaiting_user", "placed", "confirmed", "partial", "unavailable", "cook", "ready", "cook_failed", "serve", "complete", "serve_failed", "done", "failed"]
    order_retries: int
    cook_retries: int
    serve_retries: int
    final_result: str

# --- LLM Helpers ---

class OrderIntent(BaseModel):
    intent: Literal["new_order", "accept_partial", "unrelated"] = Field(
        description="new_order: user is ordering food. accept_partial: user agrees to take what's available. unrelated: not about food."
    )
    dish_name: Optional[str] = Field(default=None, description="Dish name, lowercase, singular")
    quantity: Optional[int] = Field(default=None, description="Quantity requested")

MODELS_TO_TRY = ["openai/gpt-oss-120b"]

def get_working_llm():
    for model_id in MODELS_TO_TRY:
        try:
            llm = ChatGroq(model=model_id, temperature=0)
            llm.invoke("hi")
            return llm
        except: continue
    raise RuntimeError("No working Groq models found.")

llm = get_working_llm()

def find_menu_item(name: str) -> str | None:
    name = name.strip().lower()
    if name in MENU: return name
    for key in MENU:
        if key in name or name in key: return key
    close = difflib.get_close_matches(name, MENU.keys(), n=1, cutoff=0.6)
    return close[0] if close else None

def say(instruction: str, state: AgentState) -> str:
    system = (
        "You are the friendly assistant of a restaurant's order system. "
        "Write ONE or TWO short sentences to the customer. No markdown.\n"
        f"Current order: {state['order']}\n"
        f"Retries left -> order: {state['order_retries']}, cook: {state['cook_retries']}, serve: {state['serve_retries']}\n"
        f"What to tell the customer: {instruction}"
    )
    return llm.invoke([SystemMessage(system)]).content.strip()

# --- Nodes ---

def user_input_node(state: AgentState, config: RunnableConfig) -> dict:
    # Deterministic testing support: get input from config if available
    inputs = config.get("configurable", {}).get("user_inputs")
    if inputs and len(inputs) > 0:
        user_text = inputs.pop(0)
    else:
        user_text = input("\nUser: ")
    return {"messages": [HumanMessage(content=user_text)]}

def llm_node(state: AgentState) -> dict:
    status = state["status"]
    order = dict(state["order"])
    
    # 1. Handle New Orders or Responses to Partial Offers
    if status in ("new", "awaiting_user"):
        partial_offer = "yes" if status == "awaiting_user" and order["available_quantity"] > 0 else "no"
        prompt = f"Menu: {MENU}. Was user offered partial? {partial_offer}. Extract intent from latest message."
        structured = llm.with_structured_output(OrderIntent)
        last_user = next(m for m in reversed(state["messages"]) if isinstance(m, HumanMessage))
        parsed = structured.invoke([SystemMessage(prompt), last_user])

        if parsed.intent == "accept_partial" and status == "awaiting_user" and order["available_quantity"] > 0:
            order["required_quantity"] = order["available_quantity"]
            msg = say(f"Confirm we are going ahead with {order['required_quantity']} {order['dish_name']}.", state)
            return {"messages": [AIMessage(msg)], "order": order, "status": "cook"}

        if parsed.intent == "new_order" and parsed.dish_name:
            dish = parsed.dish_name.strip().lower()
            match = find_menu_item(dish)
            order = {
                "dish_name": match if match else dish,
                "required_quantity": max(1, parsed.quantity or 1),
                "available_quantity": 0,
            }
            return {"order": order, "status": "placed"}

        # Unrelated query - costs one retry
        retries = state["order_retries"] - 1
        if retries <= 0:
            msg = say("Apologize: no valid order after 3 attempts, closing session.", state)
            return {"messages": [AIMessage(msg)], "order_retries": 0, "status": "failed", "final_result": "FAIL: Retries exhausted"}
        msg = say("Tell them you only handle food orders and ask for dish + qty. Mention retries left.", state)
        return {"messages": [AIMessage(msg)], "order_retries": retries, "status": "awaiting_user"}

    # 2. Handle confirmation results
    if status == "confirmed":
        msg = say(f"Confirm order of {order['required_quantity']} {order['dish_name']} is available and going to kitchen.", state)
        return {"messages": [AIMessage(msg)], "status": "cook"}

    if status in ("partial", "unavailable"):
        retries = state["order_retries"] - 1
        if retries <= 0:
            msg = say("Apologize: cannot fulfill and retries used up, closing session.", state)
            return {"messages": [AIMessage(msg)], "order_retries": 0, "status": "failed", "final_result": "FAIL: Stock exhausted"}
        
        instr = f"Only {order['available_quantity']} available. Ask to proceed or change order. Retries left: {retries}" if status == "partial" \
                else f"{order['dish_name']} unavailable. Ask for different order. Retries left: {retries}"
        msg = say(instr, state)
        return {"messages": [AIMessage(msg)], "order_retries": retries, "status": "awaiting_user"}

    # 3. Handle cook results
    if status == "ready":
        msg = say("Say food is ready and being served.", state)
        return {"messages": [AIMessage(msg)], "status": "serve"}

    if status == "cook_failed":
        if state["cook_retries"] > 0:
            msg = say("Say kitchen problem, cooking again.", state)
            return {"messages": [AIMessage(msg)], "status": "cook", "cook_retries": state["cook_retries"] - 1}
        msg = say("Sincere apology: kitchen failed after retries, order cancelled.", state)
        return {"messages": [AIMessage(msg)], "status": "failed", "final_result": "FAIL: Cook failed"}

    # 4. Handle serve results
    if status == "complete":
        msg = say("Order complete, enjoy your meal!", state)
        return {"messages": [AIMessage(msg)], "status": "done", "final_result": "SUCCESS"}

    if status == "serve_failed":
        if state["serve_retries"] > 0 and state["cook_retries"] > 0:
            msg = say("Serving failed, re-cooking and serving again.", state)
            return {"messages": [AIMessage(msg)], "status": "cook", "cook_retries": state["cook_retries"] - 1, "serve_retries": state["serve_retries"] - 1}
        msg = say("Sincere apology: serve failed and retries exhausted, order cancelled.", state)
        return {"messages": [AIMessage(msg)], "status": "failed", "final_result": "FAIL: Serve failed"}

    return {"status": "failed"}

def order_confirm(state: AgentState) -> dict:
    order = dict(state["order"])
    available = MENU.get(order["dish_name"], 0)
    order["available_quantity"] = available
    status = "confirmed" if available >= order["required_quantity"] else ("partial" if available > 0 else "unavailable")
    return {"order": order, "status": status}

def cook(state: AgentState, config: RunnableConfig) -> dict:
    outcomes = config.get("configurable", {}).get("cook_outcomes")
    if outcomes and len(outcomes) > 0:
        success = outcomes.pop(0)
    else:
        success = random.random() < COOK_SUCCESS_PROB
    return {"status": "ready" if success else "cook_failed"}

def serve(state: AgentState, config: RunnableConfig) -> dict:
    outcomes = config.get("configurable", {}).get("serve_outcomes")
    if outcomes and len(outcomes) > 0:
        success = outcomes.pop(0)
    else:
        success = random.random() < SERVE_SUCCESS_PROB
    return {"status": "complete" if success else "serve_failed"}

# --- Graph ---

builder = StateGraph(AgentState)
builder.add_node("user_input", user_input_node)
builder.add_node("llm", llm_node)
builder.add_node("order_confirm", order_confirm)
builder.add_node("cook", cook)
builder.add_node("serve", serve)

builder.add_edge(START, "user_input")
builder.add_edge("user_input", "llm")
builder.add_edge("order_confirm", "llm")
builder.add_edge("cook", "llm")
builder.add_edge("serve", "llm")

def route_llm(state: AgentState):
    s = state["status"]
    if s == "awaiting_user": return "user_input"
    if s == "placed": return "order_confirm"
    if s == "cook": return "cook"
    if s == "serve": return "serve"
    return END

builder.add_conditional_edges("llm", route_llm, {"user_input": "user_input", "order_confirm": "order_confirm", "cook": "cook", "serve": "serve", END: END})
graph = builder.compile()

def run_test(test_inputs: List[str], cook_outcomes: List[bool] = None, serve_outcomes: List[bool] = None):
    print("\n--- Starting Test Case ---")
    state = {
        "messages": [], "order": {"dish_name": "", "required_quantity": 0, "available_quantity": 0},
        "status": "new", "order_retries": ORDER_RETRIES_MAX, "cook_retries": COOK_RETRIES_MAX, "serve_retries": SERVE_RETRIES_MAX, "final_result": ""
    }
    
    # We must copy the outcomes lists because they are popped
    c_out = list(cook_outcomes) if cook_outcomes else []
    s_out = list(serve_outcomes) if serve_outcomes else []
    
    for user_input in test_inputs:
        print(f"\nUser: {user_input}")
        state["messages"].append(HumanMessage(content=user_input))
        output = graph.invoke(state, config={"configurable": {"cook_outcomes": c_out, "serve_outcomes": s_out}})
        state = output
        for m in reversed(state["messages"]):
            if isinstance(m, AIMessage):
                print(f"Agent: {m.content}")
                break
        if state["status"] in ("done", "failed"): break

    print(f"\nFINAL RESULT: {state['final_result']}")
    return state['final_result']

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "--test":
        print("\nRunning TC1 (Exhaust Retries)...")
        run_test(["Hello!", "I want 100 Burgers", "No thanks, I want 5 Pasta"])
        print("\nRunning TC2 (Success after failure)...")
        run_test(["I want 1 Burger"], cook_outcomes=[False, True], serve_outcomes=[False, True])
        print("\nRunning TC3 (Fail after recovery)...")
        run_test(["I want 100 Burgers", "I want 1 Burger"], cook_outcomes=[False, True], serve_outcomes=[False, False])
    else:
        print(f"Welcome! Menu: {MENU}")
        state = {
            "messages": [], "order": {"dish_name": "", "required_quantity": 0, "available_quantity": 0},
            "status": "new", "order_retries": ORDER_RETRIES_MAX, "cook_retries": COOK_RETRIES_MAX, "serve_retries": SERVE_RETRIES_MAX, "final_result": ""
        }
        while True:
            result = graph.invoke(state)
            state = result
            for m in reversed(state["messages"]):
                if isinstance(m, AIMessage):
                    print(f"Agent: {m.content}")
                    break
            if state["status"] in ("done", "failed"):
                print(f"\nFINAL RESULT: {state['final_result']}")
                break
            # Wait for user input to restart the graph
            user_text = input("\nUser: ")
            state["messages"].append(HumanMessage(content=user_text))
