# agents/notes_agent.py
import google.generativeai as genai
from agents.prompts import NOTES_PROMPT

class NotesAgent:
    @staticmethod
    def run(query: str, context: str) -> dict:
        model = genai.GenerativeModel("gemini-2.5-flash")
        prompt = NOTES_PROMPT.format(context=context, query=query)
        try:
            resp = model.generate_content(prompt)
            return {"agent": "NotesAgent", "output": (resp.text or "").strip()}
        except Exception as e:
            return {"agent": "NotesAgent", "output": f"⚠️ NotesAgent error: {e}"}
