#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
#
# Licensed under the Apache License, Version 2.0 (the "License");

"""Edge-cloud patch for GDN attention metadata.

In edge-cloud mode the worker may contain heterogeneous attention backends
(e.g. GDN on the edge side and FlashAttention on the cloud side).
GDNAttentionMetadata does not have FlashAttention-specific fields such as
seq_lens_list / attn_params / handles / events, so update_full_graph_params
would crash when it tries to access them.

This wrapper sets skip_graph_params_update on the metadata returned by the
already-patched GDNAttentionMetadataBuilder.build so that
_update_full_graph_params_if_needed (model_runner_v1.py) filters it out.
"""

from vllm.v1.attention.backends.gdn_attn import GDNAttentionMetadataBuilder

_original_gdn_build = GDNAttentionMetadataBuilder.build


def _edge_cloud_gdn_build(
    self,
    common_prefix_len: int,
    common_attn_metadata,
    num_accepted_tokens=None,
    num_decode_draft_tokens_cpu=None,
    fast_build: bool = False,
):
    attn_metadata = _original_gdn_build(
        self,
        common_prefix_len,
        common_attn_metadata,
        num_accepted_tokens=num_accepted_tokens,
        num_decode_draft_tokens_cpu=num_decode_draft_tokens_cpu,
        fast_build=fast_build,
    )
    # GDN layers do not participate in update_full_graph_params (no
    # attn_params/handles/events).  Mark the metadata so that the outer
    # filter in _update_full_graph_params_if_needed skips it and avoids
    # AttributeError when accessing FlashAttention-only fields.
    attn_metadata.skip_graph_params_update = True
    return attn_metadata


GDNAttentionMetadataBuilder.build = _edge_cloud_gdn_build  # type: ignore[misc]
