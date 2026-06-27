"""Dependency-injected, retrying, schema-validating LLM client.

The client never talks to the Anthropic API directly. It depends on a
*transport* — any object exposing::

    text(*, model, system, user, max_tokens) -> str

This keeps the whole pipeline unit-testable: tests inject a fake transport that
returns canned strings (or raises) with no network and no API key. The real
transport lives in ``draftforge.llm.anthropic_transport``.

``complete_json`` makes up to two attempts. Each attempt calls the transport,
extracts the JSON substring from the reply, and validates it against a Pydantic
model. Two failure modes are handled differently:

* **Bad output** (no JSON found, malformed JSON, or schema-validation failure):
  the validation error is appended to the user prompt and the model is asked
  again immediately — no backoff, because the problem is the content, not the
  service.
* **Transport error** (rate limit, connection error, anything else raised by
  the transport): we back off ``2 ** attempt`` seconds via the injected
  ``sleep`` and retry.

If both attempts fail, an :class:`LLMError` is raised.
"""

from __future__ import annotations

import json
import time
from typing import Protocol, TypeVar

from pydantic import BaseModel, ValidationError

MAX_ATTEMPTS = 2
# Exactly one repair attempt from the pristine `user` prompt (try, then retry once
# with the validation error fed back). Raising this to 3+ changes the retry/cost
# contract, so it is a deliberate, reviewed change — this assert makes a silent
# bump impossible.
assert MAX_ATTEMPTS == 2

ModelT = TypeVar("ModelT", bound=BaseModel)


class LLMError(Exception):
    """Raised when the LLM cannot produce valid, schema-conforming output."""


class Transport(Protocol):
    """Structural type for an injectable transport."""

    def text(self, *, model: str, system: str, user: str, max_tokens: int) -> str:
        ...


class _JSONExtractError(ValueError):
    """No JSON object/array could be located in the model's reply."""


def _strip_code_fence(text: str) -> str:
    """Drop a leading ```json / ``` code fence (and its closing fence) if present.

    Models very often wrap the JSON in a fenced code block. Removing the fence
    markers is harmless when absent and lets the balanced-scan start cleanly.
    """
    stripped = text.strip()
    if stripped.startswith("```"):
        # Remove the opening fence line (``` or ```json etc.) ...
        newline = stripped.find("\n")
        if newline != -1:
            stripped = stripped[newline + 1 :]
        # ... and a trailing closing fence, if any.
        if stripped.rstrip().endswith("```"):
            stripped = stripped.rstrip()[: -len("```")]
    return stripped


def _iter_json_candidates(text: str):
    """Yield successive top-level balanced ``{...}`` / ``[...]`` substrings.

    Models wrap JSON in prose or code fences, and the prose itself may contain a
    stray ``{...}`` (e.g. "in {curly} style"). A single-shot "first balanced
    block" extractor would return that stray block and waste the sole retry. So
    instead we yield EVERY top-level balanced block in order, depth/in-string
    aware, restarting the scan after each closed block. The caller tries each
    candidate against the schema and keeps the first that validates.

    A leading code fence is stripped first.
    """
    text = _strip_code_fence(text)
    i = 0
    n = len(text)
    while i < n:
        ch = text[i]
        if ch not in "{[":
            i += 1
            continue
        end = _scan_balanced(text, i)
        if end is None:
            # Unterminated from here on; no further top-level block can close.
            return
        yield text[i : end + 1]
        i = end + 1


def _scan_balanced(text: str, start: int) -> int | None:
    """Index of the close matching the open delimiter at ``start``, or ``None``.

    Depth- and string-literal-aware (escapes handled), so braces/brackets inside
    string values do not throw off the balance.
    """
    open_char = text[start]
    close_char = "}" if open_char == "{" else "]"
    depth = 0
    in_string = False
    escaped = False

    for i in range(start, len(text)):
        ch = text[i]
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == open_char:
            depth += 1
        elif ch == close_char:
            depth -= 1
            if depth == 0:
                return i
    return None


def _validate_first_candidate(reply: str, schema: type[ModelT]) -> ModelT:
    """Return the first balanced JSON candidate in ``reply`` that ``schema`` accepts.

    Walks every top-level balanced block (see :func:`_iter_json_candidates`),
    trying each against the schema, and returns the first that validates. If a
    candidate exists but none validate, the LAST validation/decode error is
    re-raised (so the caller's retry path feeds a real, actionable error back to
    the model). If no candidate exists at all, raises :class:`_JSONExtractError`.
    """
    last_error: Exception | None = None
    saw_candidate = False
    for payload in _iter_json_candidates(reply):
        saw_candidate = True
        try:
            return schema.model_validate_json(payload)
        except (json.JSONDecodeError, ValidationError) as exc:
            last_error = exc

    if not saw_candidate:
        raise _JSONExtractError("no JSON object or array found in reply")
    assert last_error is not None  # saw_candidate True => at least one error
    raise last_error


class LLMClient:
    """Retrying, schema-validating wrapper around an injected transport."""

    def __init__(
        self,
        transport: Transport,
        model_fast: str,
        model_smart: str,
        sleep=time.sleep,
    ) -> None:
        self._transport = transport
        self._model_fast = model_fast
        self._model_smart = model_smart
        self._sleep = sleep

    def complete_json(
        self,
        system: str,
        user: str,
        schema: type[ModelT],
        *,
        fast: bool = False,
        max_tokens: int = 2000,
    ) -> ModelT:
        """Call the model and return a validated instance of ``schema``.

        Makes up to :data:`MAX_ATTEMPTS` attempts, feeding parse/validation
        errors back into the prompt and backing off on transport errors. Raises
        :class:`LLMError` if no attempt yields valid output.
        """
        model = self._model_fast if fast else self._model_smart
        current_user = user

        for attempt in range(MAX_ATTEMPTS):
            try:
                reply = self._transport.text(
                    model=model,
                    system=system,
                    user=current_user,
                    max_tokens=max_tokens,
                )
            except Exception as exc:  # transport failure (e.g. rate limit)
                if attempt == MAX_ATTEMPTS - 1:
                    raise LLMError(
                        f"transport failed after {MAX_ATTEMPTS} attempts: {exc}"
                    ) from exc
                self._sleep(2 ** attempt)
                continue

            try:
                return _validate_first_candidate(reply, schema)
            except (_JSONExtractError, json.JSONDecodeError, ValidationError) as exc:
                if attempt == MAX_ATTEMPTS - 1:
                    raise LLMError(
                        f"model output failed validation after {MAX_ATTEMPTS} "
                        f"attempts: {exc}"
                    ) from exc
                current_user = (
                    f"{user}\n\nYour previous reply was invalid ({exc}). "
                    "Return ONLY valid JSON matching the schema."
                )

        # Unreachable: the loop either returns or raises on the final attempt.
        raise LLMError("exhausted attempts without a result")
