# agents/qa_agent.py
from groq_llm import generate
from agents.prompts import DOC_QA_SYSTEM_PROMPT

class QAAgent:
    @staticmethod
    def run(query: str, context: str) -> dict:
        prompt = f"{DOC_QA_SYSTEM_PROMPT}\n\nContext:\n{context}\n\nQuestion:\n{query}"

        try:
            resp = generate(prompt, max_tokens=600, temperature=0.0)
            return {
                "agent": "QAAgent",
                "output": (resp.get('text', '') or '').strip()
            }
        except Exception as e:
            return {"agent": "QAAgent", "output": f"⚠️ QAAgent error: {e}"}
