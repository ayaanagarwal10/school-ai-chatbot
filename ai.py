from openai import OpenAI
from config import OPENROUTER_API_KEY, MODEL

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY,
)


SYSTEM_PROMPT = """
You are Singhania AI, the official AI assistant for a school.

You help students with:
- Homework
- Science
- Mathematics
- English
- Computer Science
- General Knowledge

Explain concepts simply and accurately.
Be friendly, concise, and encouraging.
"""


def ask_ai(message: str) -> str:
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {
                "role": "system",
                "content": SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": message,
            },
        ],
    )

    return response.choices[0].message.content