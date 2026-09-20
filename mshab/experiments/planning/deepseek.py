"""The real model behind the boundary: DeepSeek through its OpenAI-compatible API.

``DeepSeekProposer`` answers the two calls with one chat completion each.
The system prompt is fixed (:mod:`mshab.experiments.planning.prompts`); the
user message is the request document as JSON; the reply is parsed as JSON and
then through the strict ``from_dict`` of the response document, so the model's
JSON shape is never trusted.  When the validator refuses an answer and asks
again, the request comes back with its ``rejections`` filled in and the
proposer continues the same conversation with one further user turn that
lists them.

Text only.  The DeepSeek chat API takes no image input, so a request whose
``images`` is not empty is refused here rather than silently truncated.

The transport is a plain callable from messages to a reply, so tests inject a
fake and never touch the network.  :class:`DeepSeekChat` is the real one; it
imports the ``openai`` package lazily (``pip install 'mshab[planning]'``) and
reads the key from ``DEEPSEEK_API_KEY``.
"""

from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple, Union

from mshab.experiments.planning.documents import (
    DecompositionRequest,
    DecompositionResponse,
    SubgraphRequest,
    SubgraphResponse,
)
from mshab.experiments.planning.prompts import PROMPTS, PromptSet
from mshab.experiments.planning.proposer import GraphProposer, ProposerUnavailable
from mshab.experiments.planning.validator import error_message


DEEPSEEK_BASE_URL = "https://api.deepseek.com"
DEEPSEEK_DEFAULT_MODEL = "deepseek-chat"
DEEPSEEK_API_KEY_VARIABLE = "DEEPSEEK_API_KEY"
DEFAULT_MAX_TOKENS = 4096
DEFAULT_TIMEOUT = 120.0

_FENCED_JSON = re.compile(r"```(?:json)?\s*(\{.*\})\s*```", re.DOTALL)
_PARSE_ERRORS = (ValueError, TypeError, KeyError)


@dataclass(frozen=True)
class ChatReply:
    """What a transport returns: the assistant text and what the API said about it."""

    content: str
    model: Optional[str] = None
    usage: Mapping[str, int] = field(default_factory=dict)
    finish_reason: Optional[str] = None
    reasoning: Optional[str] = None

    def __post_init__(self) -> None:
        if not isinstance(self.content, str):
            raise TypeError("a chat reply's content must be a string")
        object.__setattr__(self, "usage", dict(self.usage))


#: A transport: the conversation so far, as ``{"role", "content"}`` messages,
#: to the assistant's reply.  A plain string is accepted as the reply text.
ChatFunction = Callable[[Sequence[Mapping[str, str]]], Union[str, ChatReply]]


@dataclass(frozen=True)
class Exchange:
    """One model call as it happened: what was sent, what came back, how it parsed."""

    call: str
    fingerprint: Tuple[Any, ...]
    turn: int
    messages: Tuple[Mapping[str, str], ...]
    reply: ChatReply
    elapsed: float
    error: Optional[str] = None

    def as_dict(self) -> Dict[str, Any]:
        return {
            "call": self.call,
            "fingerprint": list(self.fingerprint),
            "turn": self.turn,
            "messages": [dict(message) for message in self.messages],
            "reply": self.reply.content,
            "model": self.reply.model,
            "usage": dict(self.reply.usage),
            "finish_reason": self.reply.finish_reason,
            "reasoning": self.reply.reasoning,
            "elapsed": round(self.elapsed, 3),
            "error": self.error,
        }


def _excerpt(text: str, limit: int = 200) -> str:
    return text if len(text) <= limit else text[: limit - 3] + "..."


def extract_json_object(text: str) -> Mapping[str, Any]:
    """The one JSON object in a model reply, tolerating fences and prose around it.

    Tried in order: the whole reply, a fenced ```json block, the span from the
    first ``{`` to the last ``}``.  Anything that is not a JSON object raises
    ``ValueError`` with an excerpt, which the validator records as the rejection.
    """

    stripped = text.strip()
    if not stripped:
        raise ValueError("the model reply is empty")
    candidates = [stripped]
    fenced = _FENCED_JSON.search(stripped)
    if fenced is not None:
        candidates.append(fenced.group(1))
    start, end = stripped.find("{"), stripped.rfind("}")
    if start != -1 and end > start:
        candidates.append(stripped[start : end + 1])
    for candidate in candidates:
        try:
            value = json.loads(candidate)
        except ValueError:
            continue
        if isinstance(value, Mapping):
            return value
    raise ValueError(
        "the model reply is not a JSON object: {!r}".format(_excerpt(stripped))
    )


class DeepSeekChat:
    """The transport: one chat completion per call through the ``openai`` SDK.

    DeepSeek's API is OpenAI-compatible, so the SDK is pointed at
    ``base_url``.  ``json_mode`` asks for ``response_format`` ``json_object``,
    which ``deepseek-chat`` supports; the reply is parsed leniently anyway.
    ``temperature`` is left to the server default when ``None``.
    """

    def __init__(
        self,
        *,
        model: str = DEEPSEEK_DEFAULT_MODEL,
        api_key: Optional[str] = None,
        base_url: str = DEEPSEEK_BASE_URL,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = DEFAULT_MAX_TOKENS,
        json_mode: bool = True,
        timeout: float = DEFAULT_TIMEOUT,
        max_retries: int = 2,
    ) -> None:
        try:
            import openai
        except ImportError as exc:
            raise ProposerUnavailable(
                "the DeepSeek transport needs the 'openai' package; install the "
                "planning extra: pip install 'mshab[planning]'"
            ) from exc
        key = api_key or os.environ.get(DEEPSEEK_API_KEY_VARIABLE)
        if not key:
            raise ProposerUnavailable(
                "no DeepSeek API key: set {} or pass api_key".format(
                    DEEPSEEK_API_KEY_VARIABLE
                )
            )
        self.model = model
        self.base_url = base_url
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.json_mode = json_mode
        self._client = openai.OpenAI(
            api_key=key, base_url=base_url, timeout=timeout, max_retries=max_retries
        )
        self._api_error = openai.APIError

    def describe(self) -> Dict[str, Any]:
        """The configuration a trace records; never the key."""

        return {
            "transport": "deepseek",
            "model": self.model,
            "base_url": self.base_url,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "json_mode": self.json_mode,
        }

    def __call__(self, messages: Sequence[Mapping[str, str]]) -> ChatReply:
        options: Dict[str, Any] = {
            "model": self.model,
            "messages": [dict(message) for message in messages],
        }
        if self.json_mode:
            options["response_format"] = {"type": "json_object"}
        if self.temperature is not None:
            options["temperature"] = self.temperature
        if self.max_tokens is not None:
            options["max_tokens"] = self.max_tokens
        try:
            completion = self._client.chat.completions.create(**options)
        except self._api_error as exc:
            raise ProposerUnavailable("DeepSeek request failed: {}".format(exc)) from exc
        if not completion.choices:
            raise ProposerUnavailable("DeepSeek returned a completion without choices")
        choice = completion.choices[0]
        usage = completion.usage
        counts = {}
        if usage is not None:
            for name in ("prompt_tokens", "completion_tokens", "total_tokens"):
                value = getattr(usage, name, None)
                if isinstance(value, int):
                    counts[name] = value
        return ChatReply(
            content=choice.message.content or "",
            model=completion.model,
            usage=counts,
            finish_reason=choice.finish_reason,
            reasoning=getattr(choice.message, "reasoning_content", None),
        )


class DeepSeekProposer(GraphProposer):
    """The two boundary calls answered by one chat completion each.

    A conversation is kept per request fingerprint.  A fresh request starts
    one: the system prompt and the request JSON.  A request that comes back
    with ``rejections`` continues it with one further user turn listing them,
    so the model sees its own refused answer.  Every call is recorded in
    :attr:`exchanges`, parse failures included, because a reply the strict
    parser refuses is exactly what the evaluation wants to see.
    """

    def __init__(
        self,
        chat: Optional[ChatFunction] = None,
        *,
        prompts: PromptSet = PROMPTS,
        **transport_options: Any,
    ) -> None:
        if chat is not None and transport_options:
            raise TypeError(
                "transport options {} apply to the built-in DeepSeekChat only".format(
                    sorted(transport_options)
                )
            )
        self.chat: ChatFunction = chat if chat is not None else DeepSeekChat(**transport_options)
        self.prompts = prompts
        self._transcripts: Dict[Tuple[str, Tuple[Any, ...]], List[Dict[str, str]]] = {}
        self._exchanges: List[Exchange] = []

    @property
    def exchanges(self) -> Tuple[Exchange, ...]:
        return tuple(self._exchanges)

    def describe(self) -> Dict[str, Any]:
        describe = getattr(self.chat, "describe", None)
        record = {"proposer": type(self).__name__}
        if callable(describe):
            record.update(describe())
        else:
            record["transport"] = type(self.chat).__name__
        return record

    def transcript(
        self, call: str, request: Union[DecompositionRequest, SubgraphRequest]
    ) -> Tuple[Mapping[str, str], ...]:
        """The conversation held so far for one request, if any."""

        return tuple(dict(m) for m in self._transcripts.get((call, request.fingerprint()), ()))

    def save_exchanges(self, path: Path) -> None:
        Path(path).write_text(
            json.dumps([item.as_dict() for item in self._exchanges], indent=2, sort_keys=True)
            + "\n"
        )

    # -- the two calls -------------------------------------------------------------

    def decompose(self, request: DecompositionRequest) -> DecompositionResponse:
        return self._answer(
            "decompose", request, self.prompts.decomposition, DecompositionResponse.from_dict
        )

    def plan_subgraph(self, request: SubgraphRequest) -> SubgraphResponse:
        task = request.task
        return self._answer(
            "plan_subgraph",
            request,
            self.prompts.subgraph,
            lambda payload: SubgraphResponse.from_dict(payload, task=task),
        )

    def _answer(
        self,
        call: str,
        request: Union[DecompositionRequest, SubgraphRequest],
        system_prompt: str,
        parse: Callable[[Mapping[str, Any]], Any],
    ) -> Any:
        if request.images:
            raise ValueError(
                "{} is text only; the request carries {} image reference(s)".format(
                    type(self).__name__, len(request.images)
                )
            )
        key = (call, request.fingerprint())
        previous = self._transcripts.get(key)
        if request.rejections and previous is not None:
            messages = list(previous)
            messages.append(
                {"role": "user", "content": self.prompts.rejection_turn(request.rejections)}
            )
        else:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": self.prompts.request_turn(request)},
            ]
        turn = sum(1 for message in messages if message["role"] == "user") - 1
        started = time.monotonic()
        reply = self.chat(messages)
        elapsed = time.monotonic() - started
        if isinstance(reply, str):
            reply = ChatReply(reply)
        if not isinstance(reply, ChatReply):
            raise TypeError(
                "the chat transport must return a string or a ChatReply, got {}".format(
                    type(reply).__name__
                )
            )
        self._transcripts[key] = messages + [{"role": "assistant", "content": reply.content}]
        error: Optional[str] = None
        try:
            return parse(extract_json_object(reply.content))
        except _PARSE_ERRORS as exc:
            error = error_message(exc)
            raise
        finally:
            self._exchanges.append(
                Exchange(
                    call,
                    request.fingerprint(),
                    turn,
                    tuple(dict(message) for message in messages),
                    reply,
                    elapsed,
                    error,
                )
            )
