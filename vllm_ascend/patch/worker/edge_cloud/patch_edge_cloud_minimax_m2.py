from vllm.model_executor.models.minimax_m2 import MiniMaxM2MoE
from vllm_ascend.utils import enable_sp

# ---------------------------------------------------------------------------
# MiniMaxM2MoE.__init__: inject is_sequence_parallel into FusedMoE
# ---------------------------------------------------------------------------
_original_moe_init = MiniMaxM2MoE.__init__
_original_fusedmoe_init = None


def _patched_moe_init(self, config, quant_config=None, prefix=""):
    """Patch to inject is_sequence_parallel into FusedMoE without modifying upstream.

    This temporarily replaces FusedMoE.__init__ so that any FusedMoE created
    inside the original MiniMaxM2MoE.__init__ automatically receives the correct
    is_sequence_parallel value from global vLLM config.
    """
    from vllm.model_executor.layers.fused_moe import FusedMoE

    global _original_fusedmoe_init
    if _original_fusedmoe_init is None:
        _original_fusedmoe_init = FusedMoE.__init__

    is_sequence_parallel = enable_sp()

    def _inject_sp_init(fm_self, *args, **kwargs):
        if "is_sequence_parallel" not in kwargs:
            kwargs["is_sequence_parallel"] = is_sequence_parallel
        return _original_fusedmoe_init(fm_self, *args, **kwargs)

    # Temporarily patch FusedMoE.__init__
    FusedMoE.__init__ = _inject_sp_init
    try:
        _original_moe_init(self, config, quant_config, prefix)
    finally:
        FusedMoE.__init__ = _original_fusedmoe_init


MiniMaxM2MoE.__init__ = _patched_moe_init  # type: ignore[misc]