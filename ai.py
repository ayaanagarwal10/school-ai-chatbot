import asyncio

from openai import AsyncOpenAI, RateLimitError
from config import OPENROUTER_API_KEY, MODEL
from retrieval import retrieve


client = AsyncOpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY,
    timeout=20.0,
    max_retries=0,
)


SYSTEM_PROMPT = """
You are the official AI assistant for L K Singhania Education Centre.

Answer questions about:
- Admissions
- Academics
- Boarding
- Facilities
- Sports
- Clubs
- Campus life
- Transport
- Contact information

Use ONLY the provided school context.

Rules:
- Never invent information.
- If the context does not contain the answer, say you don't have
  that information and recommend contacting the school office.
- Keep answers concise, but include all important details from the context.
- Never stop in the middle of a sentence.
- Be friendly and professional.
- Avoid repeating the same information.
- Prefer short paragraphs or bullet points.
- If a user asks something unrelated to the school, politely say
  you can only help with school-related questions.
"""


def _build_context(chunks):
    return "\n\n---\n\n".join(
        c["text"]
        for c in chunks
        if isinstance(c, dict) and c.get("text")
    )


async def ask_ai(message: str, history=None) -> str:

    if history is None:
        history = []

    # Retrieve the 3 most relevant school information chunks
    relevant_chunks = await asyncio.to_thread(
        retrieve,
        message,
        3
    )

    context = _build_context(relevant_chunks)

    user_prompt = f"""SCHOOL CONTEXT:
{context}

QUESTION:
{message}
"""

    # Keep only recent conversation history
    recent_history = history[-6:]

    messages = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT
        },
        *recent_history,
        {
            "role": "user",
            "content": user_prompt
        },
    ]

    try:

        response = await client.chat.completions.create(
            model=MODEL,
            messages=messages,
            max_tokens=450,
            temperature=0.2,
        )

        content = response.choices[0].message.content

        if content:
            return content.strip()

        print("WARNING: OpenRouter returned empty content:")
        print(response)

        return (
            "I couldn't generate an answer right now. "
            "Please try asking the question again."
        )

    except RateLimitError:

        return (
            "I'm receiving too many requests right now. "
            "Please wait a moment and try again."
        )