import google.generativeai as genai
from agents.prompts import QUESTIONS_PROMPT


class QuestionAgent:
    @staticmethod
    def run(query: str, context: str) -> dict:
        model = genai.GenerativeModel("gemini-2.5-flash")
        prompt = QUESTIONS_PROMPT.format(context=context, query=query)

        try:
            resp = model.generate_content(prompt)
            return {
                "agent": "QuestionAgent",
                "output": (resp.text or "").strip()
            }
        except Exception as e:
            return {
                "agent": "QuestionAgent",
                "output": f"⚠️ QuestionAgent error: {e}"
            }
