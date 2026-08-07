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
- You have access to conversation history — use it to understand follow-up
  questions and give coherent, contextual answers.
"""


def _build_context(chunks: list[dict]) -> str:
    return "\n\n---\n\n".join(c["text"] for c in chunks)


async def ask_ai(message: str, history: list[dict] = []) -> str:
    relevant_chunks = await asyncio.to_thread(retrieve, message, 5)
    context = _build_context(relevant_chunks)

    # Inject context into the latest user message only — not into history,
    # since historical turns already got their own context when they were sent.
    user_prompt = f"CONTEXT:\n{context}\n\nQUESTION:\n{message}"

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        *history,
        {"role": "user", "content": user_prompt},
    ]

    response = await client.chat.completions.create(
        model=MODEL,
        messages=messages,
    )
    return response.choices[0].message.content