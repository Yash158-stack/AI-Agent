# agents/qa_agent.py
import google.generativeai as genai
from agents.prompts import DOC_QA_SYSTEM_PROMPT

class QAAgent:
    @staticmethod
    def run(query: str, context: str) -> dict:
        model = genai.GenerativeModel("gemini-2.5-flash")
        prompt = f"{DOC_QA_SYSTEM_PROMPT}\n\nContext:\n{context}\n\nQuestion:\n{query}"
        try:
            resp = model.generate_content(prompt)
            return {"agent": "QAAgent", "output": (resp.text or "").strip()}
        except Exception as e:
            return {"agent": "QAAgent", "output": f"⚠️ QAAgent error: {e}"}
