# agents/smalltalk_agent.py
import google.generativeai as genai
from agents.prompts import SMALLTALK_PROMPT
from agents.keywords import SMALLTALK_KEYS

class SmallTalkAgent:
    @staticmethod
    def is_smalltalk(query: str) -> bool:
        q = (query or "").lower()
        return any(k in q for k in SMALLTALK_KEYS)

    @staticmethod
    def run(query: str, context=None) -> dict:
        model = genai.GenerativeModel("gemini-2.5-flash")
        prompt = SMALLTALK_PROMPT.format(query=query)
        try:
            resp = model.generate_content(prompt)
            return {"agent": "SmallTalkAgent", "output": (resp.text or "").strip()}
        except Exception as e:
            return {"agent": "SmallTalkAgent", "output": f"⚠️ SmallTalkAgent error: {e}"}
