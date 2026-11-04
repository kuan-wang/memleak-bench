"""LLM utility: thin wrappers around Anthropic, OpenAI, and Google APIs.

Backends:
  1. Anthropic Claude (default)
  2. Google Gemini
  3. OpenAI GPT-4o-mini (fallback)
"""

from __future__ import annotations

import os
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def _load_dotenv():
    """Load .env file from project root if it exists."""
    env_path = Path(__file__).parent.parent / ".env"
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, _, value = line.partition("=")
                os.environ.setdefault(key.strip(), value.strip())


_load_dotenv()


def _messages_contain_images(messages: list[dict]) -> bool:
    """Check if any message contains base64 image content."""
    for msg in messages:
        content = msg.get("content")
        if isinstance(content, list):
            for part in content:
                if isinstance(part, dict) and part.get("type") == "image_url":
                    return True
    return False


async def call_openai(
    messages: list[dict] | None = None,
    prompt: str | None = None,
    model: str = "gpt-4o-mini",
    temperature: float = 0.0,
    max_tokens: int = 512,
) -> str:
    """Call OpenAI API. Accepts either messages list or single prompt string.

    Auto-upgrades to gpt-4o when messages contain images.
    """
    from openai import AsyncOpenAI

    client = AsyncOpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

    if messages is None and prompt is not None:
        messages = [{"role": "user", "content": prompt}]

    # Auto-upgrade to vision model if images are present
    if model == "gpt-4o-mini" and messages and _messages_contain_images(messages):
        model = "gpt-4o"

    response = await client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return response.choices[0].message.content or ""


async def call_openai_with_system(
    system_prompt: str,
    user_message: str,
    images: list[str] | None = None,
    model: str = "gpt-4o-mini",
    temperature: float = 0.0,
    max_tokens: int = 512,
) -> str:
    """Call OpenAI with system prompt + user message + optional images.

    Auto-upgrades to gpt-4o when images are present (gpt-4o-mini has
    limited vision capability).
    """
    import base64 as b64mod

    # Auto-upgrade model for vision
    real_images = [p for p in (images or []) if not p.startswith("PLACEHOLDER:") and Path(p).exists()]
    if real_images and model == "gpt-4o-mini":
        model = "gpt-4o"

    # Build user content with images
    if real_images:
        content: list[dict] = [{"type": "text", "text": user_message}]
        for img_path in real_images:
            img_bytes = Path(img_path).read_bytes()
            img_b64 = b64mod.b64encode(img_bytes).decode()
            suffix = Path(img_path).suffix.lstrip(".")
            media = f"image/{suffix}" if suffix != "jpg" else "image/jpeg"
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:{media};base64,{img_b64}"},
            })
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": content},
        ]
    else:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ]

    return await call_openai(messages=messages, model=model,
                              temperature=temperature, max_tokens=max_tokens)


# ---------------------------------------------------------------------------
# Anthropic Claude (default)
# ---------------------------------------------------------------------------

async def call_claude(
    messages: list[dict] | None = None,
    prompt: str | None = None,
    model: str = "claude-sonnet-4-6",
    temperature: float = 0.0,
    max_tokens: int = 512,
) -> str:
    """Call Anthropic Claude API. Default for most tasks.

    Accepts OpenAI-style messages list or single prompt string.
    Converts system messages to Anthropic's system parameter.
    """
    import anthropic

    client = anthropic.AsyncAnthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

    if messages is None and prompt is not None:
        messages = [{"role": "user", "content": prompt}]

    # Extract system message (Anthropic separates it)
    system_text = ""
    user_messages = []
    for msg in messages:
        if msg["role"] == "system":
            system_text += msg["content"] if isinstance(msg["content"], str) else str(msg["content"])
            system_text += "\n"
        else:
            # Convert OpenAI content format to Anthropic format
            content = msg["content"]
            if isinstance(content, list):
                anthropic_content = []
                for part in content:
                    if part.get("type") == "text":
                        anthropic_content.append({"type": "text", "text": part["text"]})
                    elif part.get("type") == "image_url":
                        url = part["image_url"]["url"]
                        if url.startswith("data:"):
                            # Parse data URI: data:image/jpeg;base64,<data>
                            header, b64_data = url.split(",", 1)
                            media_type = header.split(":")[1].split(";")[0]
                            anthropic_content.append({
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": media_type,
                                    "data": b64_data,
                                },
                            })
                content = anthropic_content
            user_messages.append({"role": msg["role"], "content": content})

    kwargs = {
        "model": model,
        "messages": user_messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if system_text.strip():
        kwargs["system"] = system_text.strip()

    response = await client.messages.create(**kwargs)
    return response.content[0].text if response.content else ""


async def call_claude_with_system(
    system_prompt: str,
    user_message: str,
    images: list[str] | None = None,
    model: str = "claude-sonnet-4-6",
    temperature: float = 0.0,
    max_tokens: int = 512,
) -> str:
    """Call Claude with system prompt + user message + optional images.

    Drop-in replacement for call_openai_with_system.
    Uses OpenAI-style image_url format — call_claude converts internally.
    """
    import base64 as b64mod

    real_images = [p for p in (images or []) if not p.startswith("PLACEHOLDER:") and Path(p).exists()]

    if real_images:
        # Use OpenAI-style format — call_claude handles conversion
        content: list[dict] = [{"type": "text", "text": user_message}]
        for img_path in real_images:
            img_bytes = Path(img_path).read_bytes()
            img_b64 = b64mod.b64encode(img_bytes).decode()
            suffix = Path(img_path).suffix.lstrip(".")
            media = f"image/{suffix}" if suffix != "jpg" else "image/jpeg"
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:{media};base64,{img_b64}"},
            })
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": content},
        ]
    else:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ]

    return await call_claude(messages=messages, model=model,
                              temperature=temperature, max_tokens=max_tokens)


# ---------------------------------------------------------------------------
# Google Gemini (free tier — third judge)
# ---------------------------------------------------------------------------

async def call_gemini(
    prompt: str,
    model: str = "gemini-2.5-flash",
    temperature: float = 0.0,
    max_tokens: int = 512,
) -> str:
    """Call Google Gemini API."""
    try:
        from google import genai

        client = genai.Client(api_key=os.environ.get("GOOGLE_API_KEY"))
        response = client.models.generate_content(
            model=model,
            contents=prompt,
            config=genai.types.GenerateContentConfig(
                temperature=temperature,
                max_output_tokens=max_tokens,
            ),
        )
        return response.text or ""
    except ImportError:
        logger.warning("google-genai not installed, falling back to OpenAI")
        return await call_openai(prompt=prompt, temperature=temperature, max_tokens=max_tokens)
    except Exception as e:
        logger.warning(f"Gemini call failed: {e}, falling back to OpenAI")
        return await call_openai(prompt=prompt, temperature=temperature, max_tokens=max_tokens)
