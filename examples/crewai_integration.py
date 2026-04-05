"""
Vektori memory for CrewAI agents.

TODO: Implement full CrewAI integration.
      Wire v.search() into agent context and v.add() after each task.
"""

# Prerequisites: pip install crewai vektori

# from crewai import Agent, Task, Crew
# from vektori import Vektori
#
# v = Vektori()
#
# async def get_memory_context(query: str, user_id: str) -> str:
#     results = await v.search(query, user_id=user_id, depth="l1")
#     facts = "\n".join(f"- {f['text']}" for f in results.get("facts", []))
#     episodes = "\n".join(f"- {ep['text']}" for ep in results.get("episodes", []))
#     return f"Facts:\n{facts}\n\nEpisodes:\n{episodes}"

print("CrewAI integration — TODO. See docs for integration pattern.")
