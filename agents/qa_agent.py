import os
import google.generativeai as genai
from agents.prompts import DOC_QA_SYSTEM_PROMPT

try:
    from config import DEFAULT_GEMINI_MODEL
except ImportError:
    DEFAULT_GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")


class QAAgent:
    @staticmethod
    def run(query: str, context: str) -> dict:
        model = genai.GenerativeModel(DEFAULT_GEMINI_MODEL)
        prompt = (
            f"{DOC_QA_SYSTEM_PROMPT}\n\n"
            f"Context:\n{context}\n\n"
            f"Question:\n{query}"
        )

        try:
            resp = model.generate_content(prompt)
            return {
                "agent": "QAAgent",
                "output": (resp.text or "").strip(),
            }
        except Exception as e:
            return {
                "agent": "QAAgent",
                "output": f"⚠️ QAAgent error: {e}",
            }
