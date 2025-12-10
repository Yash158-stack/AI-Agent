# agents/smalltalk_agent.py
from groq_llm import generate
from agents.prompts import SMALLTALK_PROMPT
from agents.keywords import SMALLTALK_KEYS

class SmallTalkAgent:
    @staticmethod
    def is_smalltalk(query: str) -> bool:
        q = (query or "").lower()
        return any(k in q for k in SMALLTALK_KEYS)

    @staticmethod
    def run(query: str, context=None) -> dict:
        prompt = SMALLTALK_PROMPT.format(query=query)

        try:
            resp = generate(prompt, max_tokens=150, temperature=0.6)
            return {
                "agent": "SmallTalkAgent",
                "output": (resp.get("text", "") or "").strip()
            }
        except Exception as e:
            return {"agent": "SmallTalkAgent", "output": f"⚠️ SmallTalkAgent error: {e}"}
