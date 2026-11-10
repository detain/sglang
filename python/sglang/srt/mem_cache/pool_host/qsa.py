from __future__ import annotations

from sglang.srt.mem_cache.device_pool_info import IndexKeyBufferInfo
from sglang.srt.mem_cache.memory_pool_host import DeepSeekV4PagedHostPool


class QSAIndexerPoolHost(DeepSeekV4PagedHostPool):
    """Page-row mirror of QSA compressed keys, indexed by full-KV token slots.

    L2 restore fencing: the INDEXER transfer is built from the same
    ``full_layer_mapping`` as the anchor KV pool and executed by
    ``L2TransferEngine.submit_host_to_device``, which iterates every transfer
    for a loop layer on ONE host_to_device stream before calling
    ``on_layer_done(layer_id)`` (l2_transfer.py:88-110).  The per-layer event
    ``QSATokenToKVPool.get_qsa_compressed_k_buffer`` waits on is recorded on
    that same stream afterwards (cache_controller.py LayerLoadingEvent.complete
    records on the current stream), so it transitively covers this pool's
    bytes for the same global layer -- including under the ``direct`` io
    backend, whose kernels/memcpys all use getCurrentCUDAStream().  No
    separate completion fence is needed for this sidecar.
    """

    def __init__(
        self,
        buffer_info: IndexKeyBufferInfo,
        *,
        num_host_pages: int,
        layout: str,
        pin_memory: bool = True,
        device: str = "cpu",
        allocator_type: str = "default",
    ):
        buffer_info.validate()
        # A full-KV page and its compressed groups share ownership. Packing the
        # groups into a byte row lets the existing whole-page kernels move both
        # coordinate spaces using the same full-KV page number.
        super().__init__(
            pool_name="qsa_indexer",
            device_buffers=list(buffer_info.page_views()),
            item_bytes=buffer_info.page_bytes,
            num_host_pages=num_host_pages,
            slot_page_size=buffer_info.page_size,
            layout=layout,
            device=device,
            pin_memory=pin_memory,
            allocator_type=allocator_type,
            page_aligned_only=True,
        )
        self.size_per_token = self.get_size_per_token()

    def get_size_per_token(self):
        return self.layer_num * self.item_bytes // self.slot_page_size

    def get_ksize_per_token(self):
        return self.get_size_per_token()
