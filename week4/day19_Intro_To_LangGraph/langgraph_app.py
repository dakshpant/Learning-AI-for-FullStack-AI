import os
from langgraph.graph import END, StateGraph
from typing import TypedDict


class State(TypedDict):
    number: int


def double(state: State) -> dict:
    stateNumber = state["number"]  # reads board Number
    newNumber = stateNumber * 2
    print("in double")
    print(newNumber)
    return {"number": newNumber}


def finish(state: State) -> dict:
    boardNum = state["number"]
    print("in finish")
    print(boardNum)
    return {"number": boardNum}


def decision(state: State) -> str:
    if state["number"] < 100:
        return "double"
    else:
        return "finish"


builder = StateGraph(State)

# format(NodeName, and exact function Name) name can be anything
builder.add_node("double", double)
builder.add_node("finish", finish)

# entry point => Stource Node like we did in DSA

builder.set_entry_point("double")  # use the node name not the function name

builder.add_conditional_edges(
    "double",
    decision,
    {
        "double": "double",
        "finish": "finish"
    },
)

builder.add_edge("finish", END)
graph = builder.compile()

result = graph.invoke({"number": 5})
print(result)
