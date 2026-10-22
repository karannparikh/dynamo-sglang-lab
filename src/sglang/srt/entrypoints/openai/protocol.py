"""OpenAI-shaped request models.

Only used when the worker runs with SGLang's own tokenizer
(``--use-sglang-tokenizer`` / ``--dyn-chat-processor sglang``); the default
token-based path never constructs these.
"""

from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field


class ChatCompletionRequest(BaseModel):
    model: str = ""
    messages: List[Dict[str, Any]] = Field(default_factory=list)
    stream: bool = False
    max_tokens: Optional[int] = None
    max_completion_tokens: Optional[int] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    top_k: Optional[int] = None
    min_p: Optional[float] = None
    n: Optional[int] = 1
    stop: Optional[Union[str, List[str]]] = None
    presence_penalty: Optional[float] = None
    frequency_penalty: Optional[float] = None
    repetition_penalty: Optional[float] = None
    logprobs: Optional[bool] = None
    top_logprobs: Optional[int] = None
    seed: Optional[int] = None
    user: Optional[str] = None

    model_config = {"extra": "allow"}


class CompletionRequest(BaseModel):
    model: str = ""
    prompt: Union[str, List[str], List[int], List[List[int]]] = ""
    stream: bool = False
    max_tokens: Optional[int] = 16
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    n: Optional[int] = 1
    stop: Optional[Union[str, List[str]]] = None
    logprobs: Optional[int] = None
    seed: Optional[int] = None

    model_config = {"extra": "allow"}
