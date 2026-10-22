"""SGLang sampling parameters."""

from typing import Any, List, Optional


class SamplingParams:
    def __init__(
        self,
        max_new_tokens: int = 128,
        min_new_tokens: int = 0,
        n: int = 1,
        temperature: float = 1.0,
        top_p: float = 1.0,
        top_k: int = -1,
        min_p: float = 0.0,
        frequency_penalty: float = 0.0,
        presence_penalty: float = 0.0,
        repetition_penalty: float = 1.0,
        stop: Optional[List[str]] = None,
        stop_token_ids: Optional[List[int]] = None,
        ignore_eos: bool = False,
        skip_special_tokens: bool = True,
        **kwargs: Any,
    ) -> None:
        self.max_new_tokens = max_new_tokens
        self.min_new_tokens = min_new_tokens
        self.n = n
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.min_p = min_p
        self.frequency_penalty = frequency_penalty
        self.presence_penalty = presence_penalty
        self.repetition_penalty = repetition_penalty
        self.stop = stop or []
        self.stop_token_ids = stop_token_ids or []
        self.ignore_eos = ignore_eos
        self.skip_special_tokens = skip_special_tokens
        for key, value in kwargs.items():
            setattr(self, key, value)

    def normalize(self, tokenizer=None) -> None:
        """Resolve stop strings against a tokenizer, as SGLang does.

        The fake engine never sees text, so there is nothing to tokenize; the
        method exists because Dynamo calls it on the tokenizer-free path and
        expects the object to stay usable afterwards.
        """
        if self.max_new_tokens is not None and self.max_new_tokens < 0:
            raise ValueError("max_new_tokens must be non-negative")
        if self.min_new_tokens and self.max_new_tokens is not None:
            self.min_new_tokens = min(self.min_new_tokens, self.max_new_tokens)
        self.stop_strs = list(self.stop)
        self.stop_token_ids = list(self.stop_token_ids)

    def verify(self, *args, **kwargs) -> None:
        self.normalize()
