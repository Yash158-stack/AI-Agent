# agents/intent_agent.py
import google.generativeai as genai

class IntentAgent:
    model = genai.GenerativeModel("gemini-2.5-flash")

    @staticmethod
    def classify(query: str) -> str:
        """
        Return one of: "summary", "questions", "notes", "qa"
        Fallback to "qa" on any error/unclear result.
        """
        prompt = f"""Classify the user's intent into one of: summary, questions, notes, qa.
Reply with exactly one word: summary OR questions OR notes OR qa.

User query:
\"\"\"{query}\"\"\"
"""
        try:
            resp = IntentAgent.model.generate_content(prompt)
            intent = (resp.text or "").strip().lower()
        except Exception:
            intent = "qa"

        if intent not in {"summary", "questions", "notes", "qa"}:
            intent = "qa"
        return intent
