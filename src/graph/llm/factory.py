import os
from functools import lru_cache
from typing import Optional

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()


@lru_cache(maxsize=8)
def get_chat_llm(
    *,
    temperature: float = 0.0,
    streaming: bool = False,
    model: Optional[str] = None,
) -> ChatOpenAI:
    """
    Central factory for ChatOpenAI instances.

    - Cached to avoid re-instantiating clients
    - Safe to import anywhere (chains, services, nodes)
    - Explicit args keep behavior predictable
    """

    return ChatOpenAI(
        api_key=os.getenv("OPENAI_API_KEY"),
        model=model or os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        temperature=temperature,
        streaming=streaming,
    )
