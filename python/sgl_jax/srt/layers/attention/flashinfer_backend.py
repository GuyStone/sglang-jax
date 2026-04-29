"""FlashInfer attention backend for sglang-jax.

Routes attention computation through FlashInfer's BatchAttention CUDA kernels
via the jax-tvm-ffi bridge. KV cache is stored in FlashInfer's native NHD
page layout (separate K/V, no interleaving or padding).
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P
from jax.tree_util import register_pytree_node_class

from sgl_jax.srt.kernels.flashinfer_ffi.kernels import (
    ensure_flashinfer_registered,
    flashinfer_attention_run,
    flashinfer_plan,
)
from sgl_jax.srt.layers.attention.base_attn_backend import AttentionBackend
from sgl_jax.srt.layers.radix_attention import RadixAttention
from sgl_jax.srt.managers.schedule_batch import ModelWorkerBatch
from sgl_jax.srt.mem_cache.flashinfer_kv_pool import FlashInferKVPool
from sgl_jax.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode
from sgl_jax.srt.utils import cdiv
from sgl_jax.srt.utils.jax_utils import device_array
from sgl_jax.srt.utils.profiling_utils import named_scope

if TYPE_CHECKING:
    from sgl_jax.srt.mem_cache.memory_pool import KVCache

logger = logging.getLogger(__name__)


@register_pytree_node_class
@dataclass
class FlashInferMetadata:
    """Metadata for FlashInfer BatchAttention, initialized once per forward pass."""

    qo_indptr: jax.Array = None
    kv_indptr: jax.Array = None
    kv_indices: jax.Array = None
    kv_len_arr: jax.Array = None
    plan_info: jax.Array = None

    def tree_flatten(self):
        children = (
            self.qo_indptr,
            self.kv_indptr,
            self.kv_indices,
            self.kv_len_arr,
            self.plan_info,
        )
        return (children, {})

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        obj = cls.__new__(cls)
        obj.qo_indptr = children[0]
        obj.kv_indptr = children[1]
        obj.kv_indices = children[2]
        obj.kv_len_arr = children[3]
        obj.plan_info = children[4]
        return obj


@dataclass
class FlashInferAttention(AttentionBackend):
    """FlashInfer attention backend using BatchAttention CUDA kernels."""

    def __init__(
        self,
        num_attn_heads: int,
        num_kv_heads: int,
        head_dim: int,
        page_size: int = 1,
        kv_partition_axis: str = "tensor",
        mesh: jax.sharding.Mesh = None,
        dtype=jnp.bfloat16,
    ):
        self.num_heads = num_attn_heads
        self.num_kv_heads = num_kv_heads if num_kv_heads is not None else num_attn_heads
        self.head_dim = head_dim
        self.page_size = page_size
        self.kv_partition_axis = kv_partition_axis
        self.mesh = mesh
        self.forward_metadata = nnx.data(FlashInferMetadata())

        self._dtype_str = {
            jnp.float16: "float16",
            jnp.bfloat16: "bfloat16",
            jnp.float32: "float32",
            np.dtype("float16"): "float16",
            np.dtype("bfloat16"): "bfloat16",
            np.dtype("float32"): "float32",
        }.get(dtype, "bfloat16")

        # CPU-side workspace buffers for plan() (numpy, pinned)
        self._float_ws_np = np.zeros(384 * 1024 * 1024, dtype=np.uint8)
        self._int_ws_np = np.zeros(8 * 1024 * 1024, dtype=np.uint8)
        self._page_locked_ws = np.zeros(8 * 1024 * 1024, dtype=np.uint8)

        # GPU workspace buffers for run() (JAX arrays, allocated once)
        self.float_workspace = jnp.zeros(384 * 1024 * 1024, dtype=jnp.uint8)
        self.int_workspace = jnp.zeros(8 * 1024 * 1024, dtype=jnp.uint8)

        ensure_flashinfer_registered(dtype, head_dim, use_logits_soft_cap=False)
        ensure_flashinfer_registered(dtype, head_dim, use_logits_soft_cap=True)

    def get_forward_metadata(self, batch: ModelWorkerBatch):
        """Convert sglang-jax batch metadata to FlashInfer format and call plan()."""
        metadata = FlashInferMetadata()

        # 1. qo_indptr: cumulative query token counts
        if batch.forward_mode == ForwardMode.EXTEND:
            qo_indptr = np.concatenate(
                [
                    np.array([0], dtype=np.int32),
                    np.cumsum(batch.extend_seq_lens, dtype=np.int32),
                ]
            )
        elif batch.forward_mode == ForwardMode.DECODE:
            qo_indptr = np.arange(len(batch.seq_lens) + 1, dtype=np.int32)
        else:
            raise ValueError(f"Unsupported forward mode: {batch.forward_mode}")

        # 2. kv_indptr, kv_indices: page-level CSR from cache_loc
        pages_per_seq = cdiv(batch.seq_lens, self.page_size)
        kv_indptr = np.concatenate(
            [
                np.array([0], dtype=np.int32),
                np.cumsum(pages_per_seq, dtype=np.int32),
            ]
        )

        page_slots = np.arange(0, len(batch.cache_loc), self.page_size)
        kv_indices = (batch.cache_loc[page_slots] // self.page_size).astype(np.int32)

        # 3. kv_len_arr
        kv_len_arr = batch.seq_lens.astype(np.int32)

        # 4. Determine causal mode
        causal = batch.forward_mode != ForwardMode.DECODE

        # 5. Call plan() on CPU (outside JAX tracing)
        batch_size = len(batch.seq_lens)
        plan_info_np = flashinfer_plan(
            float_workspace=self._float_ws_np,
            int_workspace=self._int_ws_np,
            page_locked_workspace=self._page_locked_ws,
            qo_indptr=qo_indptr,
            kv_indptr=kv_indptr,
            kv_len_arr=kv_len_arr,
            batch_size=batch_size,
            num_qo_heads=self.num_heads,
            num_kv_heads=self.num_kv_heads,
            head_dim=self.head_dim,
            causal=causal,
            dtype_str=self._dtype_str,
            use_logits_soft_cap=False,
        )

        # 6. Transfer to device
        (
            metadata.qo_indptr,
            metadata.kv_indptr,
            metadata.kv_indices,
            metadata.kv_len_arr,
            metadata.plan_info,
        ) = device_array(
            (qo_indptr, kv_indptr, kv_indices, kv_len_arr, plan_info_np),
            sharding=(NamedSharding(self.mesh, P()) if jax.process_count() == 1 else None),
        )

        return metadata

    def tree_flatten(self):
        children = (
            self.forward_metadata,
            self.float_workspace,
            self.int_workspace,
        )
        aux_data = {
            "num_heads": self.num_heads,
            "num_kv_heads": self.num_kv_heads,
            "head_dim": self.head_dim,
            "page_size": self.page_size,
            "kv_partition_axis": self.kv_partition_axis,
            "dtype_str": self._dtype_str,
        }
        return (children, aux_data)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        obj = object.__new__(cls)
        obj.forward_metadata = children[0]
        obj.float_workspace = children[1]
        obj.int_workspace = children[2]
        obj.num_heads = aux_data["num_heads"]
        obj.num_kv_heads = aux_data["num_kv_heads"]
        obj.head_dim = aux_data["head_dim"]
        obj.page_size = aux_data["page_size"]
        obj.kv_partition_axis = aux_data["kv_partition_axis"]
        obj._dtype_str = aux_data["dtype_str"]
        obj.mesh = None
        obj._float_ws_np = None
        obj._int_ws_np = None
        obj._page_locked_ws = None
        return obj

    @named_scope
    def __call__(
        self,
        q: jax.Array,
        k: jax.Array,
        v: jax.Array,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        token_to_kv_pool: KVCache,
        causal: int = 1,
        attention_sink: jax.Array = None,
    ):
        # Step 1: Write new K/V into paged cache via JAX scatter
        token_to_kv_pool.set_kv_buffer(
            layer.layer_id,
            forward_batch.out_cache_loc,
            k,
            v,
            is_decode=(forward_batch.forward_mode == ForwardMode.DECODE),
        )

        # Step 2: Get the updated K/V cache buffers
        k_cache, v_cache = token_to_kv_pool.get_kv_buffers(layer.layer_id)

        # Step 3: Compute attention scale
        scale = (
            1.0 / math.sqrt(layer.head_dim)
            if (layer is None or layer.scaling is None)
            else layer.scaling
        )

        logits_soft_cap = layer.logit_cap if layer.logit_cap else 0.0

        # Step 4: Determine causal mode
        is_causal = True
        if forward_batch.forward_mode == ForwardMode.DECODE:
            is_causal = False

        # Step 5: Run FlashInfer attention via shard_map for TP
        q_reshaped = q.reshape(q.shape[0], -1, self.head_dim)

        in_specs = (
            P(None, self.kv_partition_axis, None),
            P(None, None, self.kv_partition_axis, None),
            P(None, None, self.kv_partition_axis, None),
            P(),
            P(),
            P(),
            P(),
        )
        out_specs = P(None, self.kv_partition_axis, None)

        def _flashinfer_attention(
            q, k_cache, v_cache, kv_indices, plan_info, float_ws, int_ws
        ):
            return flashinfer_attention_run(
                q,
                k_cache,
                v_cache,
                kv_indices,
                plan_info,
                float_ws,
                int_ws,
                num_qo_heads=self.num_heads,
                num_kv_heads=self.num_kv_heads,
                page_size=self.page_size,
                sm_scale=scale,
                logits_soft_cap=logits_soft_cap,
                causal=is_causal,
                dtype_str=self._dtype_str,
            )

        attn_output = jax.shard_map(
            _flashinfer_attention,
            in_specs=in_specs,
            out_specs=out_specs,
            check_vma=False,
        )(
            q_reshaped,
            k_cache,
            v_cache,
            self.forward_metadata.kv_indices,
            self.forward_metadata.plan_info,
            self.float_workspace,
            self.int_workspace,
        )

        return attn_output.reshape(q.shape[0], -1), (k_cache, v_cache)

    @staticmethod
    def get_max_running_reqests(max_context_len: int, page_size: int) -> int:
        num_page_per_req = cdiv(max_context_len, page_size)
        res = 1024 * 1024 // 2 // num_page_per_req // 4
        assert (
            res > 0
        ), f"max running requests: {res} must be > 0, increase page size or decrease max context length"
        return res
