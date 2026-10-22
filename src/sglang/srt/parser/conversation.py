"""Chat-template registry. Only the multimodal encode worker reads this."""

from typing import Any, Dict


class Conversation:
    def __init__(self, name: str = "default", **kwargs: Any) -> None:
        self.name = name
        self.image_token = "<image>"
        self.video_token = "<video>"
        self.audio_token = "<audio>"
        for key, value in kwargs.items():
            setattr(self, key, value)

    def copy(self) -> "Conversation":
        clone = Conversation(self.name)
        clone.__dict__.update(self.__dict__)
        return clone


class _ChatTemplates(Dict[str, Conversation]):
    def __missing__(self, key: str) -> Conversation:
        value = Conversation(key)
        self[key] = value
        return value


chat_templates: Dict[str, Conversation] = _ChatTemplates()
