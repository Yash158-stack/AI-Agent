import os
import google.generativeai as genai
from agents.prompts import SUMMARY_PROMPT

try:
    from config import DEFAULT_GEMINI_MODEL
except ImportError:
    DEFAULT_GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")


class SummaryAgent:
    @staticmethod
    def run(query: str, context: str) -> dict:
        model = genai.GenerativeModel(DEFAULT_GEMINI_MODEL)
        prompt = SUMMARY_PROMPT.format(context=context, query=query)

        try:
            resp = model.generate_content(prompt)
            return {
                "agent": "SummaryAgent",
                "output": (resp.text or "").strip(),
            }
        except Exception as e:
            return {
                "agent": "SummaryAgent",
                "output": f"⚠️ SummaryAgent error: {e}",
            }
