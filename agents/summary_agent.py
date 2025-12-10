# agents/summary_agent.py
from groq_llm import generate
from agents.prompts import SUMMARY_PROMPT

class SummaryAgent:
    @staticmethod
    def run(query: str, context: str) -> dict:
        prompt = SUMMARY_PROMPT.format(context=context, query=query)

        try:
            resp = generate(prompt, max_tokens=600, temperature=0.0)
            return {
                "agent": "SummaryAgent",
                "output": (resp.get("text", "") or "").strip()
            }
        except Exception as e:
            return {"agent": "SummaryAgent", "output": f"⚠️ SummaryAgent error: {e}"}
