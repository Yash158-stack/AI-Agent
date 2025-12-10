# agents/summary_agent.py
import google.generativeai as genai
from agents.prompts import SUMMARY_PROMPT

class SummaryAgent:
    @staticmethod
    def run(query: str, context: str) -> dict:
        model = genai.GenerativeModel("gemini-2.5-flash")
        prompt = SUMMARY_PROMPT.format(context=context, query=query)
        try:
            resp = model.generate_content(prompt)
            return {"agent": "SummaryAgent", "output": (resp.text or "").strip()}
        except Exception as e:
            return {"agent": "SummaryAgent", "output": f"⚠️ SummaryAgent error: {e}"}
