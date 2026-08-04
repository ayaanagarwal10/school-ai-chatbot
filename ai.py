from openai import AsyncOpenAI
from config import OPENROUTER_API_KEY, MODEL

client = AsyncOpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY,
)

SYSTEM_PROMPT = """
You are the official AI assistant for L. K. Singhania Education Centre's website.

You help prospective parents, guardians, students, and visitors with
questions about: admissions, academics, boarding, facilities, sports,
clubs, campus life, transport, and contact details.

Rules:
- Only answer using information you have been given about the school.
  If you don't have the information, say so clearly and suggest the
  visitor contact the school office directly — do not guess or invent details.
- Be warm, concise, and professional, as befits a school's public-facing website.
- Do not answer questions unrelated to the school (homework help, general
  knowledge, coding, etc.) — politely redirect to school-related topics.
"""


async def ask_ai(message: str) -> str:
    response = await client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": message},
        ],
    )
    return response.choices[0].message.content