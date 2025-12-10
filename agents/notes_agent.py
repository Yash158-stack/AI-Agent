# agents/notes_agent.py
from groq_llm import generate
from agents.prompts import NOTES_PROMPT

class NotesAgent:
    @staticmethod
    def run(query: str, context: str) -> dict:
        prompt = NOTES_PROMPT.format(context=context, query=query)

        try:
            resp = generate(prompt, max_tokens=600, temperature=0.1)
            return {
                "agent": "NotesAgent",
                "output": (resp.get("text", "") or "").strip()
            }
        except Exception as e:
            return {"agent": "NotesAgent", "output": f"⚠️ NotesAgent error: {e}"}
