"""Pydantic v2 schemas for the serving layer.

Every model is ``extra="ignore"``: unsupported OpenAI fields are accepted and dropped
rather than rejected (lightweight-compat posture, Q7). Validation failures surface as
400s through the app's RequestValidationError handler.
"""
from typing import Literal

from pydantic import BaseModel, ConfigDict


class _Schema(BaseModel):
    model_config = ConfigDict(extra="ignore")


class StreamOptions(_Schema):
    include_usage: bool = False


class ChatMessage(_Schema):
    role: Literal["system", "user", "assistant"]
    content: str


class ChatCompletionRequest(_Schema):
    messages: list[ChatMessage]
    model: str | None = None
    temperature: float | None = None
    max_tokens: int | None = None
    max_completion_tokens: int | None = None
    n: int | None = None
    stream: bool = False
    stream_options: StreamOptions | None = None


class CompletionRequest(_Schema):
    prompt: str
    model: str | None = None
    temperature: float | None = None
    max_tokens: int | None = None
    n: int | None = None
    stream: bool = False
    stream_options: StreamOptions | None = None


class GenerateRequest(_Schema):
    prompt: str
    sampling_params: dict | None = None
    stream: bool = False


class Usage(_Schema):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class ResponseMessage(_Schema):
    role: str = "assistant"
    content: str


class ChatChoice(_Schema):
    index: int
    message: ResponseMessage
    finish_reason: str


class ChatCompletionResponse(_Schema):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: list[ChatChoice]
    usage: Usage


class CompletionChoice(_Schema):
    index: int
    text: str
    logprobs: None = None
    finish_reason: str


class CompletionResponse(_Schema):
    id: str
    object: str = "text_completion"
    created: int
    model: str
    choices: list[CompletionChoice]
    usage: Usage


class ModelCard(_Schema):
    id: str
    object: str = "model"
    created: int
    owned_by: str = "nanovllm"


class ModelList(_Schema):
    object: str = "list"
    data: list[ModelCard]


class ErrorDetail(_Schema):
    message: str
    type: str
    param: str
    code: int


class ErrorResponse(_Schema):
    error: ErrorDetail
