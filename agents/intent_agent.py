# agents/intent_agent.py
from groq_llm import generate

class IntentAgent:
    @staticmethod
    def classify(query: str) -> str:
        """
        Return one of: "summary", "questions", "notes", "qa"
        Fallback to "qa" on any error/unclear result.
        """
        prompt = f"""Classify the user's intent into one of: summary, questions, notes, qa.
Reply with exactly one word: summary OR questions OR notes OR qa.

User query:
\"\"\"{query}\"\"\""""

        try:
            resp = generate(prompt, max_tokens=10, temperature=0.0)
            intent = (resp.get("text", "") or "").strip().lower()
        except Exception:
            intent = "qa"

        if intent not in {"summary", "questions", "notes", "qa"}:
            intent = "qa"

        return intent
