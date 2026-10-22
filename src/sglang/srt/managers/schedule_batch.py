"""Scheduler batch types. Only ``Modality`` is read by Dynamo."""

from enum import Enum


class Modality(Enum):
    IMAGE = "image"
    MULTI_IMAGES = "multi-images"
    VIDEO = "video"
    AUDIO = "audio"
