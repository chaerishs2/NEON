import os

from openai import OpenAI

from agent.prompt import SYSTEM


_client = None


def classify(user_prompt):
    global _client
    if _client is None:
        _client = OpenAI(timeout=10, max_retries=0)
    res = _client.chat.completions.create(
        model=os.getenv("MODEL_NAME") or "gpt-4o-mini",
        response_format={"type": "json_object"},
        temperature=0.2,
        messages=[
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": user_prompt},
        ],
    )
    return res.choices[0].message.content
