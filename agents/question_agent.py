# agents/question_agent.py
from groq_llm import generate
from agents.prompts import QUESTIONS_PROMPT

class QuestionAgent:
    @staticmethod
    def run(query: str, context: str) -> dict:
        prompt = QUESTIONS_PROMPT.format(context=context, query=query)

        try:
            resp = generate(prompt, max_tokens=600, temperature=0.0)
            return {
                "agent": "QuestionAgent",
                "output": (resp.get("text", "") or "").strip()
            }
        except Exception as e:
            return {"agent": "QuestionAgent", "output": f"⚠️ QuestionAgent error: {e}"}
