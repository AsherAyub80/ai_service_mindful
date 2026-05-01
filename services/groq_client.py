# ai_service/services/groq_client.py
import os
import json
import re
from groq import AsyncGroq
from functools import lru_cache

_client = None

def get_groq():
    global _client
    if _client is None:
        _client = AsyncGroq(api_key=os.getenv("GROQ_API_KEY"))
    return _client

SMART_MODEL = "llama-3.3-70b-versatile"  # Best quality, still FREE
FAST_MODEL  = "llama-3.1-8b-instant"     # Faster/cheaper for simple tasks


async def chat(messages: list, model: str = SMART_MODEL, temperature: float = 0.7, max_tokens: int = 1024) -> str:
    client = get_groq()
    response = await client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return response.choices[0].message.content


async def chat_json(messages: list, model: str = SMART_MODEL, max_tokens: int = 1024) -> dict | list:
    """Call Groq and parse the response as JSON."""
    # Instruct model to return JSON
    json_messages = messages.copy()
    json_messages[-1]["content"] += "\n\nRespond with ONLY valid JSON. No markdown, no explanation."

    text = await chat(json_messages, model=model, temperature=0.3, max_tokens=max_tokens)

    # Try direct parse
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Extract JSON from response (model sometimes adds preamble)
    match = re.search(r'(\{[\s\S]*\}|\[[\s\S]*\])', text)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass

    raise ValueError(f"Could not parse JSON from Groq response: {text[:300]}")
