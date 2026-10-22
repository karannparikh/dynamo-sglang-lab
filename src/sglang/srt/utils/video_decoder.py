"""Video decoding surface. Multimodal workers are out of scope."""


class VideoDecoderWrapper:
    def __init__(self, *args, **kwargs):
        raise NotImplementedError("fake-sglang does not decode video")

    @staticmethod
    def load_video(*args, **kwargs):
        raise NotImplementedError("fake-sglang does not decode video")
