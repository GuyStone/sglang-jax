"""FlashInfer kernel compilation, FFI registration, and JAX-facing functions.

Uses the jax-tvm-ffi bridge to call FlashInfer CUDA kernels from JAX.
Three-step pattern per kernel: BUILD & LOAD → REGISTER → CALL.
"""

from __future__ import annotations

import functools
import logging
import math
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import numpy as np

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Dtype mappings
# ---------------------------------------------------------------------------

_JAX_TO_TORCH_DTYPE = None


def _jax_to_torch(jax_dtype):
    """Convert a JAX/numpy dtype to a torch dtype (lazy import)."""
    global _JAX_TO_TORCH_DTYPE
    if _JAX_TO_TORCH_DTYPE is None:
        import torch

        _JAX_TO_TORCH_DTYPE = {
            jnp.float16: torch.float16,
            jnp.bfloat16: torch.bfloat16,
            jnp.float32: torch.float32,
            np.dtype("float16"): torch.float16,
            np.dtype("bfloat16"): torch.bfloat16,
            np.dtype("float32"): torch.float32,
        }
    return _JAX_TO_TORCH_DTYPE[jax_dtype]


def _torch_int32():
    import torch

    return torch.int32


# ---------------------------------------------------------------------------
# Module compilation (cached on disk by FlashInfer JIT)
# ---------------------------------------------------------------------------


@functools.cache
def _get_batch_attention_module(
    dtype_q_str: str,
    dtype_kv_str: str,
    head_dim_qk: int,
    head_dim_vo: int,
    use_logits_soft_cap: bool,
):
    """Compile and load the BatchAttention TVM FFI module."""
    import torch

    from flashinfer.jit import gen_batch_attention_module

    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    dtype_q = dtype_map[dtype_q_str]
    dtype_kv = dtype_map[dtype_kv_str]
    dtype_o = dtype_q
    dtype_idx = torch.int32

    module = gen_batch_attention_module(
        dtype_q=dtype_q,
        dtype_kv=dtype_kv,
        dtype_o=dtype_o,
        dtype_idx=dtype_idx,
        head_dim_qk=head_dim_qk,
        head_dim_vo=head_dim_vo,
        pos_encoding_mode=0,  # PosEncodingMode.kNone
        use_logits_soft_cap=use_logits_soft_cap,
        use_profiler=False,
    ).build_and_load()
    logger.info(
        "Compiled FlashInfer BatchAttention module: %s/%s head_dim=%d/%d logits_cap=%s",
        dtype_q_str,
        dtype_kv_str,
        head_dim_qk,
        head_dim_vo,
        use_logits_soft_cap,
    )
    return module


@functools.cache
def _get_page_module():
    """Compile and load the page operations TVM FFI module."""
    from flashinfer.jit.page import gen_page_module

    module = gen_page_module().build_and_load()
    logger.info("Compiled FlashInfer page module")
    return module


# ---------------------------------------------------------------------------
# FFI target registration (called once per config)
# ---------------------------------------------------------------------------

_registered_targets: set[str] = set()


def _attention_target_name(dtype_str: str, head_dim_qk: int, head_dim_vo: int, use_cap: bool):
    cap = "cap" if use_cap else "nocap"
    return f"flashinfer.batch_attn_{dtype_str}_h{head_dim_qk}x{head_dim_vo}_{cap}"


def _register_batch_attention(
    dtype_str: str,
    head_dim_qk: int,
    head_dim_vo: int,
    use_logits_soft_cap: bool,
):
    target = _attention_target_name(dtype_str, head_dim_qk, head_dim_vo, use_logits_soft_cap)
    if target in _registered_targets:
        return target

    import jax_tvm_ffi

    module = _get_batch_attention_module(
        dtype_str, dtype_str, head_dim_qk, head_dim_vo, use_logits_soft_cap
    )
    _run = module.run

    # Wrapper: reorder from JAX convention (rets, args, attrs) to TVM signature.
    #
    # TVM run signature:
    #   run(float_ws, int_ws, plan_info, q, k_cache, v_cache, kv_indices,
    #       o, maybe_lse, mask_mode, layout, num_qo_heads, num_kv_heads,
    #       page_size, v_scale, sm_scale, logits_soft_cap)
    #
    # arg_spec layout:
    #   rets:  out, lse_or_empty
    #   args:  float_ws, int_ws, q, k_cache, v_cache, kv_indices, plan_info_arr
    #   attrs: mask_mode, layout, num_qo_heads, num_kv_heads, page_size,
    #          v_scale, sm_scale, logits_soft_cap
    def _run_wrapper(
        out,
        lse_or_empty,
        float_ws,
        int_ws,
        q,
        k_cache,
        v_cache,
        kv_indices,
        plan_info_arr,
        mask_mode,
        layout,
        num_qo_heads,
        num_kv_heads,
        page_size,
        v_scale,
        sm_scale,
        logits_soft_cap,
    ):
        import tvm.ffi as _tvm_ffi

        plan_info = _tvm_ffi.Array(
            [int(plan_info_arr[i]) for i in range(plan_info_arr.shape[0])]
        )
        lse = None if lse_or_empty.shape[0] == 0 else lse_or_empty
        _run(
            float_ws,
            int_ws,
            plan_info,
            q,
            k_cache,
            v_cache,
            kv_indices,
            out,
            lse,
            mask_mode,
            layout,
            num_qo_heads,
            num_kv_heads,
            page_size,
            v_scale,
            sm_scale,
            logits_soft_cap,
        )

    jax_tvm_ffi.register_ffi_target(
        target,
        _run_wrapper,
        arg_spec=[
            "rets",
            "args",
            "attrs.mask_mode",
            "attrs.layout",
            "attrs.num_qo_heads",
            "attrs.num_kv_heads",
            "attrs.page_size",
            "attrs.v_scale",
            "attrs.sm_scale",
            "attrs.logits_soft_cap",
        ],
        platform="gpu",
        allow_cuda_graph=True,
        pass_owned_tensor=True,
    )

    _registered_targets.add(target)
    logger.info("Registered FFI target: %s", target)
    return target


_APPEND_REGISTERED = False


def _register_append_paged_kv_cache():
    global _APPEND_REGISTERED
    if _APPEND_REGISTERED:
        return
    _APPEND_REGISTERED = True

    import jax_tvm_ffi

    page_mod = _get_page_module()
    _append_fn = page_mod.append_paged_kv_cache

    # TVM signature:
    #   append_paged_kv_cache(append_key, append_value, batch_indices, positions,
    #                         paged_k_cache, paged_v_cache,
    #                         kv_indices, kv_indptr, kv_last_page_len, layout)
    #
    # With input_output_aliases, rets (updated_k, updated_v) are aliased to
    # args (paged_k_cache, paged_v_cache). The wrapper receives them as the
    # SAME GPU buffer — the kernel mutates in place.
    #
    # arg_spec:
    #   rets:  updated_k, updated_v
    #   args:  append_key, append_value, batch_indices, positions,
    #          paged_k_cache, paged_v_cache, kv_indices, kv_indptr, kv_last_page_len
    #   attrs: layout
    def _append_wrapper(
        updated_k,
        updated_v,
        append_key,
        append_value,
        batch_indices,
        positions,
        paged_k_cache,
        paged_v_cache,
        kv_indices,
        kv_indptr,
        kv_last_page_len,
        layout,
    ):
        _append_fn(
            append_key,
            append_value,
            batch_indices,
            positions,
            updated_k,
            updated_v,
            kv_indices,
            kv_indptr,
            kv_last_page_len,
            layout,
        )

    jax_tvm_ffi.register_ffi_target(
        "flashinfer.append_paged_kv_cache",
        _append_wrapper,
        arg_spec=["rets", "args", "attrs.layout"],
        platform="gpu",
        allow_cuda_graph=True,
        pass_owned_tensor=True,
    )
    logger.info("Registered FFI target: flashinfer.append_paged_kv_cache")


# ---------------------------------------------------------------------------
# Public: ensure registration
# ---------------------------------------------------------------------------


def ensure_flashinfer_registered(dtype, head_dim: int, use_logits_soft_cap: bool = False):
    """Ensure FlashInfer FFI targets are compiled and registered for this config."""
    dtype_str = {
        jnp.float16: "float16",
        jnp.bfloat16: "bfloat16",
        jnp.float32: "float32",
        np.dtype("float16"): "float16",
        np.dtype("bfloat16"): "bfloat16",
        np.dtype("float32"): "float32",
    }.get(dtype, str(dtype))

    _register_batch_attention(dtype_str, head_dim, head_dim, use_logits_soft_cap)
    if use_logits_soft_cap:
        _register_batch_attention(dtype_str, head_dim, head_dim, False)
    _register_append_paged_kv_cache()


# ---------------------------------------------------------------------------
# Public: plan() — called on CPU outside JAX tracing
# ---------------------------------------------------------------------------


def flashinfer_plan(
    float_workspace: np.ndarray,
    int_workspace: np.ndarray,
    page_locked_workspace: np.ndarray,
    qo_indptr: np.ndarray,
    kv_indptr: np.ndarray,
    kv_len_arr: np.ndarray,
    batch_size: int,
    num_qo_heads: int,
    num_kv_heads: int,
    head_dim: int,
    causal: bool,
    dtype_str: str,
    use_logits_soft_cap: bool = False,
) -> np.ndarray:
    """Call BatchAttention.plan() on the CPU. Returns plan_info as int64 numpy array."""
    module = _get_batch_attention_module(
        dtype_str, dtype_str, head_dim, head_dim, use_logits_soft_cap
    )

    import tvm.ffi as _tvm_ffi

    plan_info_tvm = module.plan(
        _tvm_ffi.from_numpy(float_workspace),
        _tvm_ffi.from_numpy(int_workspace),
        _tvm_ffi.from_numpy(page_locked_workspace),
        _tvm_ffi.from_numpy(qo_indptr),
        _tvm_ffi.from_numpy(kv_indptr),
        _tvm_ffi.from_numpy(kv_len_arr),
        batch_size,
        num_qo_heads,
        num_kv_heads,
        head_dim,
        causal,
    )

    return np.array([int(x) for x in plan_info_tvm], dtype=np.int64)


# ---------------------------------------------------------------------------
# Public: attention run — inside @jax.jit
# ---------------------------------------------------------------------------


def flashinfer_attention_run(
    q: jax.Array,
    k_cache: jax.Array,
    v_cache: jax.Array,
    kv_indices: jax.Array,
    plan_info: jax.Array,
    float_workspace: jax.Array,
    int_workspace: jax.Array,
    *,
    num_qo_heads: int,
    num_kv_heads: int,
    page_size: int,
    sm_scale: float,
    logits_soft_cap: float = 0.0,
    causal: bool = True,
    dtype_str: str = "bfloat16",
) -> jax.Array:
    """Run FlashInfer BatchAttention inside a JAX computation graph."""
    target = _attention_target_name(
        dtype_str, q.shape[-1], q.shape[-1], logits_soft_cap > 0.0
    )
    mask_mode = 1 if causal else 0  # CAUSAL=1, NON_CAUSAL=0

    out, _ = jax.ffi.ffi_call(
        target,
        (
            jax.ShapeDtypeStruct(q.shape, q.dtype),
            jax.ShapeDtypeStruct((0,), jnp.float32),  # lse sentinel
        ),
    )(
        float_workspace,
        int_workspace,
        q,
        k_cache,
        v_cache,
        kv_indices,
        plan_info,
        mask_mode=mask_mode,
        layout=0,  # NHD
        num_qo_heads=num_qo_heads,
        num_kv_heads=num_kv_heads,
        page_size=page_size,
        v_scale=1.0,
        sm_scale=sm_scale,
        logits_soft_cap=logits_soft_cap,
    )
    return out


# ---------------------------------------------------------------------------
# Public: append_paged_kv_cache — inside @jax.jit
# ---------------------------------------------------------------------------


def flashinfer_append_paged_kv_cache(
    append_key: jax.Array,
    append_value: jax.Array,
    batch_indices: jax.Array,
    positions: jax.Array,
    paged_k_cache: jax.Array,
    paged_v_cache: jax.Array,
    kv_indices: jax.Array,
    kv_indptr: jax.Array,
    kv_last_page_len: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """Append new K/V tokens to paged cache via FlashInfer CUDA kernel.

    Uses input_output_aliases for zero-copy in-place mutation:
    output 0 aliases input 4 (paged_k_cache),
    output 1 aliases input 5 (paged_v_cache).
    """
    updated_k, updated_v = jax.ffi.ffi_call(
        "flashinfer.append_paged_kv_cache",
        (
            jax.ShapeDtypeStruct(paged_k_cache.shape, paged_k_cache.dtype),
            jax.ShapeDtypeStruct(paged_v_cache.shape, paged_v_cache.dtype),
        ),
        input_output_aliases={0: 4, 1: 5},
    )(
        append_key,
        append_value,
        batch_indices,
        positions,
        paged_k_cache,
        paged_v_cache,
        kv_indices,
        kv_indptr,
        kv_last_page_len,
        layout=0,  # NHD
    )
    return updated_k, updated_v
