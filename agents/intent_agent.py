import os
import google.generativeai as genai

try:
    from config import DEFAULT_GEMINI_MODEL
except ImportError:
    DEFAULT_GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")


class IntentAgent:
    @staticmethod
    def classify(query: str) -> str:
        """
        Return one of:
        "summary", "questions", "notes", "qa"
        Fallback to "qa" on any error/unclear result.
        """
        prompt = f"""
        Classify the user's intent into one of:
        summary, questions, notes, qa.

        Reply with exactly one word:
        summary OR questions OR notes OR qa.

        User query:
        \"\"\"{query}\"\"\"
        """

        try:
            model = genai.GenerativeModel(DEFAULT_GEMINI_MODEL)
            resp = model.generate_content(prompt)
            intent = (resp.text or "").strip().lower()
        except Exception:
            intent = "qa"

        if intent not in {"summary", "questions", "notes", "qa"}:
            intent = "qa"

        return intent
