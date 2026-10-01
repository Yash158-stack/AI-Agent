import os
import re
import google.generativeai as genai
from agents.keywords import SMALLTALK_KEYS
from agents.prompts import SMALLTALK_PROMPT

try:
    from config import DEFAULT_GEMINI_MODEL
except ImportError:
    DEFAULT_GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")


class SmallTalkAgent:
    @staticmethod
    def is_smalltalk(query: str) -> bool:
        q = (query or "").lower().strip()
        if not q:
            return False
        for key in SMALLTALK_KEYS:
            pattern = rf"\b{re.escape(key)}\b"
            if re.search(pattern, q):
                return True
        return False

    @staticmethod
    def run(query: str, context=None) -> dict:
        model = genai.GenerativeModel(DEFAULT_GEMINI_MODEL)
        prompt = SMALLTALK_PROMPT.format(query=query)

        try:
            resp = model.generate_content(prompt)
            return {
                "agent": "SmallTalkAgent",
                "output": (resp.text or "").strip(),
            }
        except Exception as e:
            return {
                "agent": "SmallTalkAgent",
                "output": f"⚠️ SmallTalkAgent error: {e}",
            }
