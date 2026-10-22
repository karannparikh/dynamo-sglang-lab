"""Diffusion generator surface. Image/video workers are out of scope."""


class DiffGenerator:
    def __init__(self, *args, **kwargs):
        raise NotImplementedError(
            "fake-sglang does not simulate diffusion workers; "
            "run an LLM worker (aggregated, prefill, or decode) instead."
        )
