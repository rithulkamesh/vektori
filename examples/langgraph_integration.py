"""
Vektori memory for LangGraph agents.

TODO: Implement full LangGraph integration.
      Use as a custom memory node in the graph.
"""

# Prerequisites: pip install langgraph vektori

# from langgraph.graph import StateGraph
# from vektori import Vektori
#
# v = Vektori()
#
# async def memory_node(state: dict) -> dict:
#     """LangGraph node: fetch relevant memories and inject into state."""
#     query = state["messages"][-1]["content"]
#     results = await v.search(query, user_id=state["user_id"], depth="l1")
#     state["memory_context"] = results
#     return state

print("LangGraph integration — TODO. See docs for integration pattern.")
