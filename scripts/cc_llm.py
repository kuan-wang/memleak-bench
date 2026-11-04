#!/usr/bin/env python3
"""Claude calls routed through the Claude Code CLI (`claude -p`).

Optional drop-in Claude backend used for some experiments instead of the
Anthropic API. Text calls are single-turn with a custom system prompt and all tools
disabled — semantically equivalent to a plain API call. Vision calls stage the
images in a temp dir and let the model Read them (file-read apparatus; disclose
in any cross-run comparison).

Set CLAUDE_BACKEND=cc in the environment to activate in the experiment scripts.
"""

from __future__ import annotations

import asyncio
import os
import shutil
import tempfile
from pathlib import Path

CLAUDE_BIN = shutil.which("claude") or os.path.expanduser("~/.local/bin/claude")
_TIMEOUT = 300


class CCError(RuntimeError):
    pass


CC_CONFIG_DIR = os.path.expanduser(os.environ.get("MEMLEAK_CLAUDE_CONFIG_DIR", ""))  # optional separate CLI config dir


async def _run(args: list[str], cwd: str | None = None) -> str:
    env = {k: v for k, v in os.environ.items() if k != "ANTHROPIC_API_KEY"}
    if CC_CONFIG_DIR and os.path.isdir(CC_CONFIG_DIR):
        env["CLAUDE_CONFIG_DIR"] = CC_CONFIG_DIR
    proc = await asyncio.create_subprocess_exec(
        CLAUDE_BIN, *args,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        cwd=cwd,
        env=env,  # drop the API key so the CLI uses its own login
    )
    try:
        out, err = await asyncio.wait_for(proc.communicate(), timeout=_TIMEOUT)
    except asyncio.TimeoutError:
        proc.kill()
        raise CCError(f"claude -p timed out after {_TIMEOUT}s")
    text = out.decode().strip()
    if proc.returncode != 0:
        raise CCError(f"claude -p exit {proc.returncode}: {err.decode()[:300] or text[:300]}")
    if not text:
        raise CCError(f"claude -p empty output: {err.decode()[:300]}")
    return text


async def _retry(fn, max_retries: int):
    last = None
    limit_waits = 0
    attempt = 0
    while attempt <= max_retries:
        try:
            return await fn()
        except CCError as e:
            last = e
            if "limit" in str(e).lower():
                # usage-window limits can last hours; wait them out (up to ~6h)
                limit_waits += 1
                if limit_waits > 24:
                    raise
                await asyncio.sleep(900)
                continue  # limit waits don't consume regular retries
            attempt += 1
            await asyncio.sleep(3 * attempt)
    raise last


async def cc_text(model: str, prompt: str, system: str | None = None, max_retries: int = 3) -> str:
    """Single-turn text call, no tools."""
    args = ["-p", prompt, "--model", model, "--disallowedTools", "*",
            "--setting-sources", ""]
    if system:
        args += ["--system-prompt", system]
    return await _retry(lambda: _run(args), max_retries)


async def cc_vision(model: str, images: list[str], question: str,
                    system: str | None = None, max_retries: int = 2) -> str:
    """Vision call: stage images in a temp dir, model Reads them, answers."""
    tmp = tempfile.mkdtemp(prefix="ccvis_")
    try:
        names = []
        for i, src in enumerate(images):
            name = f"img_{i:02d}{Path(src).suffix}"
            os.symlink(os.path.abspath(src), os.path.join(tmp, name))
            names.append(name)
        prompt = (
            f"Read ALL of these image files in the current directory: {', '.join(names)}. "
            f"After viewing every image, answer the following.\n\n{question}"
        )
        args = ["-p", prompt, "--model", model, "--allowedTools", "Read",
                "--setting-sources", ""]
        if system:
            args += ["--system-prompt", system]
        return await _retry(lambda: _run(args, cwd=tmp), max_retries)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
