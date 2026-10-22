"""Multimodal processor base. Dynamo's ``_compat.ensure_sglang_tensor_image_size``
monkey-patches ``resolve_image_token_counts`` on this class."""

from typing import Any, List


class BaseMultimodalProcessor:
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self._processor = None

    def resolve_image_token_counts(self, images: List[Any]) -> List[int]:
        raise NotImplementedError("fake-sglang does not process images")
