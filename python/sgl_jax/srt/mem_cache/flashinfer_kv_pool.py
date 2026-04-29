"""FlashInfer-native KV cache pool with separate K/V in NHD page layout."""

from __future__ import annotations

import logging
import time

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P
from jax.tree_util import register_pytree_node_class

from sgl_jax.srt.mem_cache.memory_pool import KVCache, merge_kv

logger = logging.getLogger(__name__)


@register_pytree_node_class
class FlashInferKVPool(KVCache):
    """KV cache stored as separate K and V page tensors in FlashInfer NHD format.

    Buffer shape per layer: [num_pages, page_size, num_kv_heads, head_dim]
    No interleaving, packing, or alignment padding — matches FlashInfer's native layout.
    """

    def __init__(
        self,
        size: int,
        page_size: int,
        dtype: jnp.dtype,
        head_num: int,
        head_dim: int,
        layer_num: int,
        mesh: Mesh,
        start_layer: int | None = None,
        end_layer: int | None = None,
    ):
        super().__init__(size, page_size, dtype, layer_num, mesh, start_layer, end_layer)
        self.head_num = head_num
        self.head_dim = head_dim
        self.kv_partition_axis = "tensor"

        self._create_buffers()
        self._calculate_memory_usage()

    def _create_buffers(self):
        self.k_sharding = NamedSharding(
            self.mesh, P(None, None, self.kv_partition_axis, None)
        )
        self.v_sharding = NamedSharding(
            self.mesh, P(None, None, self.kv_partition_axis, None)
        )
        self.kv_sharding = (self.k_sharding, self.v_sharding)

        assert (
            self.size % self.page_size == 0
        ), "Cache size must be divisible by page size"

        num_pages = self.size // self.page_size + 1
        buf_shape = (num_pages, self.page_size, self.head_num, self.head_dim)

        logger.info(
            "Creating FlashInfer KV buffers: %d layers, shape %s, dtype %s",
            self.layer_num,
            buf_shape,
            self.dtype,
        )
        start_time = time.time()

        self.k_buffers = []
        self.v_buffers = []
        for _ in range(self.layer_num):
            k = jnp.zeros(buf_shape, dtype=self.dtype)
            v = jnp.zeros(buf_shape, dtype=self.dtype)
            k = jax.device_put(k, self.k_sharding)
            v = jax.device_put(v, self.v_sharding)
            self.k_buffers.append(k)
            self.v_buffers.append(v)

        elapsed = time.time() - start_time
        logger.info(
            "FlashInfer KV buffers created in %.2fs, total %.2f GB",
            elapsed,
            self.mem_usage / (1024**3),
        )

    def _calculate_memory_usage(self):
        num_pages = self.size // self.page_size + 1
        bytes_per_element = jnp.dtype(self.dtype).itemsize
        per_layer = (
            2 * num_pages * self.page_size * self.head_num * self.head_dim * bytes_per_element
        )
        self.mem_usage = per_layer * self.layer_num

    def tree_flatten(self):
        parent_children, parent_aux_data = super().tree_flatten()
        children = tuple(self.k_buffers) + tuple(self.v_buffers) + parent_children
        aux_data = {
            **parent_aux_data,
            "head_num": self.head_num,
            "head_dim": self.head_dim,
            "kv_partition_axis": self.kv_partition_axis,
        }
        return (children, aux_data)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        obj = object.__new__(cls)
        layer_num = aux_data["layer_num"]
        k_buffers = list(children[:layer_num])
        v_buffers = list(children[layer_num : 2 * layer_num])
        parent_children = children[2 * layer_num :]

        parent_obj = KVCache.tree_unflatten(aux_data, parent_children)
        for attr in [
            "size",
            "page_size",
            "dtype",
            "layer_num",
            "mesh",
            "start_layer",
            "end_layer",
            "mem_usage",
        ]:
            setattr(obj, attr, getattr(parent_obj, attr))

        obj.head_num = aux_data["head_num"]
        obj.head_dim = aux_data["head_dim"]
        obj.kv_partition_axis = aux_data["kv_partition_axis"]
        obj.k_buffers = k_buffers
        obj.v_buffers = v_buffers

        obj.k_sharding = NamedSharding(
            obj.mesh, P(None, None, obj.kv_partition_axis, None)
        )
        obj.v_sharding = NamedSharding(
            obj.mesh, P(None, None, obj.kv_partition_axis, None)
        )
        obj.kv_sharding = (obj.k_sharding, obj.v_sharding)
        return obj

    # -----------------------------------------------------------------------
    # KVCache interface
    # -----------------------------------------------------------------------

    def get_kv_buffers(self, layer_id: int) -> tuple[jax.Array, jax.Array]:
        """Return (k_cache, v_cache) in FlashInfer NHD page layout."""
        idx = layer_id - self.start_layer
        return self.k_buffers[idx], self.v_buffers[idx]

    def get_fused_kv_buffer(self, layer_id: int) -> jax.Array:
        """Construct fused 5D view on demand (compatibility only)."""
        k, v = self.get_kv_buffers(layer_id)
        k_flat = k.reshape(-1, self.head_num, self.head_dim)
        v_flat = v.reshape(-1, self.head_num, self.head_dim)
        return merge_kv(k_flat, v_flat)

    def get_kv_buffer(self, layer_id: int) -> tuple[jax.Array, jax.Array]:
        """Return separate 3D K/V: [total_tokens, num_kv_heads, head_dim]."""
        k, v = self.get_kv_buffers(layer_id)
        return (
            k.reshape(-1, self.head_num, self.head_dim),
            v.reshape(-1, self.head_num, self.head_dim),
        )

    def set_kv_buffer(
        self,
        layer_id: int,
        loc: jax.Array,
        cache_k: jax.Array,
        cache_v: jax.Array,
        is_decode: bool = True,
    ) -> None:
        """Scatter-write new K/V into cache pages (JAX fallback path).

        The FlashInfer backend's primary path uses flashinfer_append_paged_kv_cache
        via FFI, but this method is provided for compatibility with code paths
        that call set_kv_buffer directly.
        """
        idx = layer_id - self.start_layer
        k_buf = self.k_buffers[idx]
        v_buf = self.v_buffers[idx]

        page_idx = loc // self.page_size
        slot_idx = loc % self.page_size

        k_3d = cache_k.reshape(-1, self.head_num, self.head_dim)
        v_3d = cache_v.reshape(-1, self.head_num, self.head_dim)

        self.k_buffers[idx] = k_buf.at[page_idx, slot_idx].set(k_3d)
        self.v_buffers[idx] = v_buf.at[page_idx, slot_idx].set(v_3d)

    def replace_kv_buffer(self, kv_pairs: list) -> None:
        """Replace buffers with list of (k, v) tuples from JIT output."""
        for i, (k, v) in enumerate(kv_pairs):
            self.k_buffers[self.start_layer + i] = k
            self.v_buffers[self.start_layer + i] = v

    def get_kv_size_bytes(self):
        return self.mem_usage

    def get_cpu_copy(self, indices):
        kv_cache_host = []
        for layer_id in range(self.layer_num):
            k, v = self.get_kv_buffers(layer_id + self.start_layer)
            k_host = jax.device_get(k[indices])
            v_host = jax.device_get(v[indices])
            kv_cache_host.append([k_host, v_host])
        return kv_cache_host

    def load_cpu_copy(self, kv_cache_host, indices):
        for layer_id in range(self.layer_num):
            k_host, v_host = kv_cache_host[layer_id]
            idx = layer_id
            k_dev = jax.device_put(k_host, self.k_sharding)
            v_dev = jax.device_put(v_host, self.v_sharding)
            self.k_buffers[idx] = self.k_buffers[idx].at[indices].set(k_dev)
            self.v_buffers[idx] = self.v_buffers[idx].at[indices].set(v_dev)

    def clear_cache(self, indices: jax.Array):
        for i in range(self.layer_num):
            self.k_buffers[i] = self.k_buffers[i].at[indices].set(0)
            self.v_buffers[i] = self.v_buffers[i].at[indices].set(0)
