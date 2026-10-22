"""Request/response structs exchanged with SGLang's tokenizer manager.

These are stdlib dataclasses on purpose: Dynamo's ``engine_generate.py`` wraps
``GenerateReqInput`` in a pydantic ``TypeAdapter`` and calls
``validate_python`` on a raw dict, which works for dataclasses and would not
for a plain class.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Union


def _is_batched(text, input_ids) -> bool:
    if isinstance(text, (list, tuple)):
        return True
    if isinstance(input_ids, (list, tuple)) and input_ids:
        return isinstance(input_ids[0], (list, tuple))
    return False


def _derive_rids(rid, batch_size: int):
    """One request id per batch element, as SGLang does.

    A caller that already supplied a list owns those ids -- Dynamo derives
    trace-aligned ones (``<trace>-0``, ``<trace>-1``) before it hands the
    request over, and overwriting them would break request correlation.
    """
    if isinstance(rid, (list, tuple)):
        return list(rid)
    if rid:
        return [f"{rid}-{i}" for i in range(batch_size)]
    return [uuid.uuid4().hex for _ in range(batch_size)]


class _BatchNormalizationMixin:
    def normalize_batch_and_arguments(self) -> None:
        """Resolve single-vs-batch shape and assign request ids.

        SGLang calls this on the way into the tokenizer manager; Dynamo's
        handlers and tests rely on the ids it settles on.
        """
        batched = _is_batched(self.text, self.input_ids)
        self.is_single = not batched
        if batched:
            source = self.text if isinstance(self.text, (list, tuple)) else self.input_ids
            self.batch_size = len(source)
            self.rid = _derive_rids(self.rid, self.batch_size)
        else:
            self.batch_size = 1
            if not self.rid:
                self.rid = uuid.uuid4().hex


@dataclass
class GenerateReqInput(_BatchNormalizationMixin):
    """SGLang's native ``/generate`` body.

    Dynamo reconstructs this from the opaque ``extra_args.sglang_tito``
    passthrough, replacing input, routing state, and bootstrap fields.
    """

    text: Optional[Union[str, List[str]]] = None
    input_ids: Optional[Union[List[int], List[List[int]]]] = None
    input_embeds: Optional[Any] = None
    rid: Optional[Union[str, List[str]]] = None
    sampling_params: Optional[Union[Dict[str, Any], List[Dict[str, Any]]]] = None
    stream: bool = False
    return_logprob: bool = False
    logprob_start_len: int = -1
    top_logprobs_num: int = 0
    token_ids_logprob: Optional[List[int]] = None
    return_text_in_logprobs: bool = False
    return_hidden_states: bool = False
    lora_path: Optional[Union[str, List[Optional[str]]]] = None
    lora_id: Optional[str] = None
    session_params: Optional[Dict[str, Any]] = None
    session_id: Optional[str] = None
    parent_session_id: Optional[str] = None
    custom_logit_processor: Optional[str] = None
    priority: Optional[int] = None
    # Disaggregation handoff. Dynamo's PrefillRouter round-trips these three.
    bootstrap_host: Optional[Union[str, List[str]]] = None
    bootstrap_port: Optional[Union[int, List[int]]] = None
    bootstrap_room: Optional[Union[int, List[int]]] = None
    disagg_prefill_dp_rank: Optional[int] = None
    routed_dp_rank: Optional[int] = None
    data_parallel_rank: Optional[int] = None
    external_trace_header: Optional[Dict[str, str]] = None
    image_data: Optional[Any] = None
    video_data: Optional[Any] = None
    audio_data: Optional[Any] = None
    mm_hashes: Optional[Any] = None
    require_reasoning: bool = False
    return_routed_experts: bool = False
    routed_experts_start_len: int = -1
    extra_key: Optional[str] = None

    def __post_init__(self) -> None:
        if self.text is None and self.input_ids is None:
            raise ValueError("GenerateReqInput needs either `text` or `input_ids`")


@dataclass
class EmbeddingReqInput(_BatchNormalizationMixin):
    text: Optional[Union[str, List[str]]] = None
    input_ids: Optional[Union[List[int], List[List[int]]]] = None
    rid: Optional[Union[str, List[str]]] = None
    image_data: Optional[Any] = None
    audio_data: Optional[Any] = None
    video_data: Optional[Any] = None
    is_cross_encoder_request: bool = False
    priority: Optional[int] = None
    external_trace_header: Optional[Dict[str, str]] = None
    data_parallel_rank: Optional[int] = None


class ProfileReqType(Enum):
    START_PROFILE = 1
    STOP_PROFILE = 2


@dataclass
class ProfileReq:
    type: Optional[ProfileReqType] = None
    output_dir: Optional[str] = None
    start_step: Optional[int] = None
    num_steps: Optional[int] = None
    activities: Optional[List[str]] = None
    with_stack: Optional[bool] = None
    record_shapes: Optional[bool] = None
    profile_by_stage: bool = False
    profile_id: Optional[str] = None


@dataclass
class ScaleElasticEPReqInput:
    new_ep_size: int = 0
    pause_children: bool = False


@dataclass
class UpdateWeightFromDiskReqInput:
    model_path: str = ""
    load_format: Optional[str] = None
    abort_all_requests: bool = False
    weight_version: Optional[str] = None


@dataclass
class UpdateWeightsFromDistributedReqInput:
    names: List[str] = field(default_factory=list)
    dtypes: List[str] = field(default_factory=list)
    shapes: List[List[int]] = field(default_factory=list)
    group_name: str = "weight_update_group"
    flush_cache: bool = True
    abort_all_requests: bool = False
    weight_version: Optional[str] = None


@dataclass
class UpdateWeightsFromTensorReqInput:
    serialized_named_tensors: List[Any] = field(default_factory=list)
    load_format: Optional[str] = None
    flush_cache: bool = True
    abort_all_requests: bool = False
    weight_version: Optional[str] = None


@dataclass
class UpdateWeightsFromIPCReqInput:
    zmq_handles: Dict[str, str] = field(default_factory=dict)
    flush_cache: bool = True
    abort_all_requests: bool = False
    weight_version: Optional[str] = None


@dataclass
class LoadLoRAAdapterReqInput:
    lora_name: str = ""
    lora_path: str = ""
    lora_id: Optional[str] = None
    pinned: bool = False


@dataclass
class UnloadLoRAAdapterReqInput:
    lora_name: str = ""
    lora_id: Optional[str] = None


@dataclass
class UpdateWeightVersionReqInput:
    new_version: str = ""
    abort_all_requests: bool = False


@dataclass
class ReleaseMemoryOccupationReqInput:
    tags: Optional[List[str]] = None


@dataclass
class ResumeMemoryOccupationReqInput:
    tags: Optional[List[str]] = None


@dataclass
class PauseGenerationReqInput:
    pass


@dataclass
class ContinueGenerationReqInput:
    pass


@dataclass
class FlushCacheReqInput:
    pass


@dataclass
class AbortReq:
    rid: str = ""
    abort_all: bool = False
    finished_reason: Optional[Dict[str, Any]] = None
