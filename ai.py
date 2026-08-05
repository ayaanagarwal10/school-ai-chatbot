import asyncio

from openai import AsyncOpenAI
from config import OPENROUTER_API_KEY, MODEL
from retrieval import retrieve

client = AsyncOpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY,
)

SYSTEM_PROMPT = """
You are the official AI assistant for L K Singhania Education Centre's website.

You help prospective parents, guardians, students, and visitors with
questions about admissions, academics, boarding, facilities, sports, clubs,
campus life, transport, and contact details.

You will be given CONTEXT retrieved from the school's official website and
documents. Answer using only that context.

Rules:
- If the context does not contain the answer, say so clearly and suggest the
  visitor contact the school office directly — do not guess or invent details.
- Be warm, concise, and professional.
- Do not answer questions unrelated to the school.
"""


def _build_context(chunks: list[dict]) -> str:
    return "\n\n---\n\n".join(c["text"] for c in chunks)


async def ask_ai(message: str) -> str:
    # retrieve() is CPU-bound (embedding + numpy search), so it's run in a
    # thread to avoid blocking the event loop, same reasoning as before.
    relevant_chunks = await asyncio.to_thread(retrieve, message, 3)
    context = _build_context(relevant_chunks)
    user_prompt = f"CONTEXT:\n{context}\n\nQUESTION:\n{message}"

    response = await client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
    )
    return response.choices[0].message.content