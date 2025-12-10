# groq_llm.py  -- FINAL + FULLY WORKING VERSION (no SDK needed)

import os
import requests

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")

API_URL = "https://api.groq.com/openai/v1/chat/completions"


def generate(prompt: str, max_tokens: int = 512, temperature: float = 0.0):
    """
    Works exactly like OpenAI API format (Groq uses same schema).
    No SDK needed. Uses pure HTTPS POST.
    """

    if not GROQ_API_KEY:
        return {
            "text": "[ERROR] Missing GROQ_API_KEY environment variable."
        }

    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": GROQ_MODEL,
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "max_tokens": max_tokens,
        "temperature": temperature
    }

    try:
        response = requests.post(API_URL, json=payload, headers=headers, timeout=20)
        data = response.json()

        if "choices" in data:
            return {"text": data["choices"][0]["message"]["content"]}

        return {"text": f"[Groq API Error] {data}"}

    except Exception as e:
        return {"text": f"[Groq Exception] {e}"}
