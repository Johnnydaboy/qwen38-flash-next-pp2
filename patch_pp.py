import pathlib

# 1) model_state.py (nvidia + amd)
for p in [pathlib.Path("vllm/models/qwen4_exp/nvidia/model_state.py"),
          pathlib.Path("vllm/models/qwen4_exp/amd/model_state.py")]:
    t = p.read_text()
    if "get_pp_group" not in t:
        t = t.replace("from vllm.config import VllmConfig",
                      "from vllm.config import VllmConfig\nfrom vllm.distributed.parallel_state import get_pp_group")
    old = """        self.uses_ngram_embedding = bool(config.ple_layer_ids)
        if not self.uses_ngram_embedding:
            self.ngram_context_len = 0
            self.ngram_eos_token_id = 0
            return

        if vllm_config.parallel_config.pipeline_parallel_size > 1:
            raise RuntimeError(
                "N-gram PLE embedding currently requires "
                "pipeline_parallel_size=1 because non-first pipeline ranks do "
                "not receive the raw input_ids required by PLE. Please run "
                "with PP=1."
            )"""
    new = """        try:
            is_first_pp = get_pp_group().is_first_rank
        except Exception:
            is_first_pp = True
        self.uses_ngram_embedding = bool(config.ple_layer_ids) and is_first_pp
        if not self.uses_ngram_embedding:
            self.ngram_context_len = 0
            self.ngram_eos_token_id = 0
            return"""
    if old in t:
        t = t.replace(old, new)
        print("patched model_state.py", p)
    else:
        print("WARNING model_state pattern not found:", p)
    p.write_text(t)

# 2) config.py — Qwen4Exp N-gram PLE check
p = pathlib.Path("vllm/model_executor/models/config.py")
t = p.read_text()
old_check = """        if text_config.ple_layer_ids and parallel_config.pipeline_parallel_size > 1:
            raise NotImplementedError(
                "Qwen4Exp N-gram PLE embedding requires pipeline_parallel_size=1 "
                "because non-first pipeline ranks do not receive the raw input_ids "
                "it needs. Please run with PP=1."
            )"""
new_check = """        # PP patched (qwen38-flash-next:pp2): PLE offload is PP-aware — only
        # the first pipeline stage holds PLE layers, so PP > 1 is supported.
        pass"""
if old_check in t:
    t = t.replace(old_check, new_check)
    print("patched config.py PLE check")
else:
    print("WARNING config.py PLE check pattern not found")
p.write_text(t)

# 3) gpu_worker.py
p = pathlib.Path("vllm/v1/worker/gpu_worker.py")
t = p.read_text()
old_pp = """        if parallel_config.pipeline_parallel_size != 1:
            unsupported.append(f"PP={parallel_config.pipeline_parallel_size}")"""
new_pp = """        pass"""
if old_pp in t:
    t = t.replace(old_pp, new_pp)
    print("patched gpu_worker validate")
else:
    print("WARNING gpu_worker PP check pattern not found")
p.write_text(t)

# 4) model.py (nvidia + amd)
for p in [pathlib.Path("vllm/models/qwen4_exp/nvidia/model.py"),
          pathlib.Path("vllm/models/qwen4_exp/amd/model.py")]:
    t = p.read_text()
    old_tuple = """        skip_substrs = (
            "hashstats_",
            "token_lookup",
            "hyper_connection_mixer.block_inject_weight",
        )"""
    old_list = """        skip_substrs = [
            "hashstats_",
            "token_lookup",
            "hyper_connection_mixer.block_inject_weight",
        ]"""
    new_hc = """        skip_substrs = [
            "hashstats_",
            "token_lookup",
            "hyper_connection_mixer.block_inject_weight",
        ]
        if not get_pp_group().is_last_rank:
            skip_substrs.append("hyper_connection_mixer.")"""
    if old_tuple in t:
        t = t.replace(old_tuple, new_hc)
        print("patched model.py HC skip (tuple)", p)
    elif old_list in t:
        t = t.replace(old_list, new_hc)
        print("patched model.py HC skip (list)", p)
    else:
        print("WARNING model.py HC skip pattern not found:", p)
    p.write_text(t)

# 5) model_runner.py
p = pathlib.Path("vllm/v1/worker/gpu/model_runner.py")
t = p.read_text()
old_setup = """        query_start_loc_source = getattr(self.model_state, "ple_query_start_loc", None)
        ngram_context_source = getattr(self.model_state, "ngram_context", None)
        if not isinstance(query_start_loc_source, torch.Tensor):
            raise RuntimeError("PLE offload requires a query_start_loc source")"""
new_setup = """        query_start_loc_source = getattr(self.model_state, "ple_query_start_loc", None)
        ngram_context_source = getattr(self.model_state, "ngram_context", None)
        from vllm.distributed.parallel_state import get_pp_group as _get_pp_group
        try:
            _is_first_pp = _get_pp_group().is_first_rank
        except Exception:
            _is_first_pp = True
        if not isinstance(query_start_loc_source, torch.Tensor):
            if not _is_first_pp:
                return
            raise RuntimeError("PLE offload requires a query_start_loc source")"""
if old_setup in t:
    t = t.replace(old_setup, new_setup)
    print("patched model_runner setup_ple_offload")
else:
    print("WARNING model_runner setup_ple_offload pattern not found")

old_runner_init_kv = """        for kv_cache_group in kv_cache_config.kv_cache_groups:
            spec = kv_cache_group.kv_cache_spec
            block_sizes.append(spec.block_size)
            layer_spec = (
                spec.first_spec if isinstance(spec, UniformTypeKVCacheSpecs) else spec
            )
            slot_mapping_enabled.append(not isinstance(layer_spec, CircularBufferSpec))"""
new_runner_init_kv = """        for kv_cache_group in kv_cache_config.kv_cache_groups:
            spec = kv_cache_group.kv_cache_spec
            block_sizes.append(spec.block_size)
            layer_spec = (
                spec.first_spec if isinstance(spec, UniformTypeKVCacheSpecs) else spec
            )
            if layer_spec is None:
                slot_mapping_enabled.append(False)
                max_num_blocks_per_group.append(0)
                continue
            slot_mapping_enabled.append(not isinstance(layer_spec, CircularBufferSpec))"""
if old_runner_init_kv in t:
    t = t.replace(old_runner_init_kv, new_runner_init_kv)
    print("patched model_runner initialize_kv_cache")
else:
    print("WARNING model_runner initialize_kv_cache pattern not found")

p.write_text(t)

# 6) mtp.py (nvidia + amd)
for p in [pathlib.Path("vllm/models/qwen4_exp/nvidia/mtp.py"),
          pathlib.Path("vllm/models/qwen4_exp/amd/mtp.py")]:
    t = p.read_text()
    old = """        if get_pp_group().is_first_rank:
            assert hidden_states is not None"""
    new = """        if get_pp_group().is_first_rank or hidden_states is not None:
            assert hidden_states is not None"""
    if old in t:
        t = t.replace(old, new)
        print("patched mtp.py draft-branch", p)
    else:
        print("WARNING mtp.py pattern not found:", p)
    p.write_text(t)

# 7) connector.py
p = pathlib.Path("vllm/v1/ple_offload/connector.py")
t = p.read_text()
if "get_pp_group" not in t:
    t = t.replace("from vllm.distributed.parallel_state import get_dp_group, get_tp_group",
                  "from vllm.distributed.parallel_state import get_dp_group, get_pp_group, get_tp_group")
old_init = "        self.dp_rank = get_dp_group().rank_in_group\n        self.tp_rank = get_tp_group().rank_in_group\n        self._layers = self._setup_layers(vllm_config, model)"
new_init = ("        self.dp_rank = get_dp_group().rank_in_group\n"
            "        self.tp_rank = get_tp_group().rank_in_group\n"
            "        try:\n"
            "            self.is_first_pp = get_pp_group().is_first_rank\n"
            "        except Exception:\n"
            "            self.is_first_pp = True\n"
            "        self._layers = self._setup_layers(vllm_config, model)\n"
            "        if not self._layers and not self.is_first_pp:\n"
            "            return")
if old_init in t:
    t = t.replace(old_init, new_init)
    print("patched connector init")
else:
    print("WARNING connector init pattern not found")
old_setup = """        if not layers:
            raise RuntimeError(
                "VLLM_PLE_CPU_OFFLOAD is enabled, but the model has no PleOffloadLayer"
            )"""
new_setup = """        if not layers:
            try:
                is_first = get_pp_group().is_first_rank
            except Exception:
                is_first = True
            if not is_first:
                return {}
            raise RuntimeError(
                "VLLM_PLE_CPU_OFFLOAD is enabled, but the model has no PleOffloadLayer"
            )"""
if old_setup in t:
    t = t.replace(old_setup, new_setup)
    print("patched connector setup")

old_launch = """        # Inputs are replicated across TP ranks. One request per DP rank drives
        # the CPU result fan-out to every registered TP output buffer.
        if self.tp_rank != 0:
            return"""
new_launch = """        # Inputs are replicated across TP ranks. One request per DP rank drives
        # the CPU result fan-out to every registered TP output buffer.
        if not getattr(self, 'is_first_pp', True):
            return
        if self.tp_rank != 0:
            return"""
if old_launch in t:
    t = t.replace(old_launch, new_launch)
    print("patched connector launch")
else:
    print("WARNING connector launch pattern not found")
p.write_text(t)

# 8) kv_cache_utils.py (_project_kv_cache_groups_to_worker)
p = pathlib.Path("vllm/v1/core/kv_cache_utils.py")
t = p.read_text()
old_project = """        if worker_layer_names and isinstance(group_spec, UniformTypeKVCacheSpecs):
            group_spec = UniformTypeKVCacheSpecs(
                block_size=group_spec.block_size,
                kv_cache_specs={
                    layer_name: group_spec.kv_cache_specs[layer_name]
                    for layer_name in worker_layer_names
                },
            )"""
new_project = """        if isinstance(group_spec, UniformTypeKVCacheSpecs):
            group_spec = UniformTypeKVCacheSpecs(
                block_size=group_spec.block_size,
                kv_cache_specs={
                    layer_name: group_spec.kv_cache_specs[layer_name]
                    for layer_name in worker_layer_names
                    if layer_name in group_spec.kv_cache_specs
                },
            )"""
if old_project in t:
    t = t.replace(old_project, new_project)
    print("patched kv_cache_utils.py _project_kv_cache_groups_to_worker")
else:
    print("WARNING kv_cache_utils pattern not found")

old_bytes = """def _get_kv_cache_bytes_per_block(
    kv_cache_groups: list[KVCacheGroupSpec],
) -> int:
    \"\"\"Return the largest cache group's bytes per block.\"\"\"
    bytes_per_block = max(
        sum(
            _get_per_layer_spec(group, layer_name).page_size_bytes
            for layer_name in group.layer_names
        )
        for group in kv_cache_groups
    )
    assert bytes_per_block > 0
    return bytes_per_block"""
new_bytes = """def _get_kv_cache_bytes_per_block(
    kv_cache_groups: list[KVCacheGroupSpec],
) -> int:
    \"\"\"Return the largest cache group's bytes per block.\"\"\"
    active_groups = [g for g in kv_cache_groups if g.layer_names]
    if not active_groups:
        return 1
    bytes_per_block = max(
        sum(
            _get_per_layer_spec(group, layer_name).page_size_bytes
            for layer_name in group.layer_names
        )
        for group in active_groups
    )
    assert bytes_per_block > 0
    return bytes_per_block"""
if old_bytes in t:
    t = t.replace(old_bytes, new_bytes)
    print("patched kv_cache_utils.py _get_kv_cache_bytes_per_block")
else:
    print("WARNING kv_cache_utils.py _get_kv_cache_bytes_per_block pattern not found")

old_mem = """def _max_memory_usage_bytes_from_groups(
    vllm_config: VllmConfig,
    kv_cache_groups: list[KVCacheGroupSpec],
) -> int:
    \"\"\"
    Calculate maximum memory usage in bytes from KV cache groups.

    This correctly accounts for padding in hybrid models. For example, if a
    model has 8 full attention layers and 9 sliding window layers, they will
    be padded to 9 full + 9 sliding window for uniform group sizes.

    Each group independently claims blocks from the shared pool, so a request consumes
    the sum of the per-group block counts, i.e. ``bytes_per_block * total_blocks``.
    \"\"\"
    if not kv_cache_groups:
        return 0

    bytes_per_block = _pool_bytes_per_block(kv_cache_groups)
    total_blocks = 0
    for group in kv_cache_groups:
        spec = group.kv_cache_spec
        if isinstance(spec, UniformTypeKVCacheSpecs):
            total_blocks += spec.max_memory_usage_pages(vllm_config)
        else:
            total_blocks += cdiv(
                spec.max_memory_usage_bytes(vllm_config),
                spec.page_size_bytes,
            )

    return bytes_per_block * total_blocks"""
new_mem = """def _max_memory_usage_bytes_from_groups(
    vllm_config: VllmConfig,
    kv_cache_groups: list[KVCacheGroupSpec],
) -> int:
    active_groups = [g for g in kv_cache_groups if g.layer_names]
    if not active_groups:
        return 0

    bytes_per_block = _pool_bytes_per_block(active_groups)
    total_blocks = 0
    for group in active_groups:
        spec = group.kv_cache_spec
        if isinstance(spec, UniformTypeKVCacheSpecs):
            total_blocks += spec.max_memory_usage_pages(vllm_config)
        else:
            total_blocks += cdiv(
                spec.max_memory_usage_bytes(vllm_config),
                spec.page_size_bytes,
            )

    return bytes_per_block * total_blocks"""
if old_mem in t:
    t = t.replace(old_mem, new_mem)
    print("patched kv_cache_utils.py _max_memory_usage_bytes_from_groups")
else:
    print("WARNING kv_cache_utils.py _max_memory_usage_bytes_from_groups pattern not found")

p.write_text(t)

# 9) worker/utils.py (allocate_kv_cache safe lookup & prepare_kernel_block_sizes)
p = pathlib.Path("vllm/v1/worker/utils.py")
t = p.read_text()
old_alloc = """    for tensor in kv_cache_config.kv_cache_tensors:
        layer_name = tensor.layers[0]
        group_id, group = next(
            (group_id, group)
            for group_id, group in enumerate(kv_cache_config.kv_cache_groups)
            if layer_name in group.layer_names
        )"""
new_alloc = """    for tensor in kv_cache_config.kv_cache_tensors:
        layer_name = tensor.layers[0]
        matched_groups = [
            (group_id, group)
            for group_id, group in enumerate(kv_cache_config.kv_cache_groups)
            if layer_name in group.layer_names
        ]
        if not matched_groups:
            continue
        group_id, group = matched_groups[0]"""
if old_alloc in t:
    t = t.replace(old_alloc, new_alloc)
    print("patched worker/utils.py allocate_kv_cache")
else:
    print("WARNING worker/utils.py allocate_kv_cache pattern not found")

old_prep = """    for kv_cache_gid, kv_cache_group in enumerate(kv_cache_config.kv_cache_groups):
        kv_cache_spec = kv_cache_group.kv_cache_spec
        if isinstance(kv_cache_spec, UniformTypeKVCacheSpecs):
            # All layers in the UniformTypeKVCacheSpecs have the same type,
            # pick an arbitrary one to dispatch.
            kv_cache_spec = next(iter(kv_cache_spec.kv_cache_specs.values()))
        if isinstance(kv_cache_spec, EncoderOnlyAttentionSpec):
            continue
        if isinstance(kv_cache_spec, AttentionSpec):
            # This is an attention backend that supports virtual block splitting.
            kv_manager_block_size = kv_cache_group.kv_cache_spec.block_size
            group_backends = [g.backend for g in attn_groups[kv_cache_gid]]
            selected_kernel_size = select_common_block_size(
                kv_manager_block_size, group_backends
            )
            kernel_block_sizes.append(selected_kernel_size)"""
new_prep = """    for kv_cache_gid, kv_cache_group in enumerate(kv_cache_config.kv_cache_groups):
        kv_cache_spec = kv_cache_group.kv_cache_spec
        if isinstance(kv_cache_spec, UniformTypeKVCacheSpecs):
            # All layers in the UniformTypeKVCacheSpecs have the same type,
            # pick an arbitrary one to dispatch.
            if not kv_cache_spec.kv_cache_specs:
                kernel_block_sizes.append(kv_cache_spec.block_size)
                continue
            kv_cache_spec = next(iter(kv_cache_spec.kv_cache_specs.values()))
        if isinstance(kv_cache_spec, EncoderOnlyAttentionSpec):
            continue
        if isinstance(kv_cache_spec, AttentionSpec):
            # This is an attention backend that supports virtual block splitting.
            kv_manager_block_size = kv_cache_group.kv_cache_spec.block_size
            group_backends = [g.backend for g in attn_groups[kv_cache_gid]]
            if not group_backends:
                kernel_block_sizes.append(kv_manager_block_size)
                continue
            selected_kernel_size = select_common_block_size(
                kv_manager_block_size, group_backends
            )
            kernel_block_sizes.append(selected_kernel_size)"""
if old_prep in t:
    t = t.replace(old_prep, new_prep)
    print("patched worker/utils.py prepare_kernel_block_sizes")
else:
    print("WARNING worker/utils.py prepare_kernel_block_sizes pattern not found")

p.write_text(t)

# 10) kv_cache_interface.py (UniformTypeKVCacheSpecs guards & first_spec fallback)
p = pathlib.Path("vllm/v1/kv_cache_interface.py")
t = p.read_text()
old_first_spec = """    @property
    def first_spec(self) -> KVCacheSpec:
        \"\"\"Return the first spec in the group.\"\"\"
        return next(iter(self.kv_cache_specs.values()))"""
new_first_spec = """    @property
    def first_spec(self) -> KVCacheSpec | None:
        \"\"\"Return the first spec in the group.\"\"\"
        return next(iter(self.kv_cache_specs.values()), None)"""
if old_first_spec in t:
    t = t.replace(old_first_spec, new_first_spec)
    print("patched kv_cache_interface.py first_spec")
else:
    print("WARNING kv_cache_interface.py first_spec pattern not found")

old_iface = """    def get_max_layers_per_page_size(self) -> int:
        \"\"\"Max number of layers sharing a page size. For a balanced bucket
        this equals the number of repetitions of the layer pattern.\"\"\"
        return Counter(
            spec.page_size_bytes for spec in self.kv_cache_specs.values()
        ).most_common(1)[0][1]

    def max_memory_usage_pages(self, vllm_config: VllmConfig) -> int:
        return max(
            cdiv(spec.max_memory_usage_bytes(vllm_config), spec.page_size_bytes)
            for spec in self.kv_cache_specs.values()
        )"""
new_iface = """    def get_max_layers_per_page_size(self) -> int:
        \"\"\"Max number of layers sharing a page size. For a balanced bucket
        this equals the number of repetitions of the layer pattern.\"\"\"
        if not self.kv_cache_specs:
            return 0
        return Counter(
            spec.page_size_bytes for spec in self.kv_cache_specs.values()
        ).most_common(1)[0][1]

    def max_memory_usage_pages(self, vllm_config: VllmConfig) -> int:
        if not self.kv_cache_specs:
            return 0
        return max(
            cdiv(spec.max_memory_usage_bytes(vllm_config), spec.page_size_bytes)
            for spec in self.kv_cache_specs.values()
        )"""
if old_iface in t:
    t = t.replace(old_iface, new_iface)
    print("patched kv_cache_interface.py UniformTypeKVCacheSpecs")
else:
    print("WARNING kv_cache_interface.py UniformTypeKVCacheSpecs pattern not found")

old_max_mem = """    def max_memory_usage_bytes(self, vllm_config: VllmConfig) -> int:
        max_num_pages = max(
            cdiv(spec.max_memory_usage_bytes(vllm_config), spec.page_size_bytes)
            for spec in self.kv_cache_specs.values()
        )
        return max_num_pages * self.page_size_bytes"""
new_max_mem = """    def max_memory_usage_bytes(self, vllm_config: VllmConfig) -> int:
        if not self.kv_cache_specs:
            return 0
        max_num_pages = max(
            cdiv(spec.max_memory_usage_bytes(vllm_config), spec.page_size_bytes)
            for spec in self.kv_cache_specs.values()
        )
        return max_num_pages * self.page_size_bytes"""
if old_max_mem in t:
    t = t.replace(old_max_mem, new_max_mem)
    print("patched kv_cache_interface.py max_memory_usage_bytes")
else:
    print("WARNING kv_cache_interface.py max_memory_usage_bytes pattern not found")

old_uniform = """    @classmethod
    def is_uniform_type(cls, kv_cache_specs: dict[str, KVCacheSpec]) -> bool:
        \"\"\"
        Whether all layers have the same type of KV cache spec.

        Uses the registry to determine grouping base classes, so custom specs
        that inherit from FullAttentionSpec are treated as full attention.
        \"\"\"
        block_sizes = set(spec.block_size for spec in kv_cache_specs.values())
        if len(block_sizes) > 1:
            # Different block sizes, not uniform.
            return False
        first_spec = next(iter(kv_cache_specs.values()))
        return first_spec.is_uniform_with_collection(kv_cache_specs)"""
new_uniform = """    @classmethod
    def is_uniform_type(cls, kv_cache_specs: dict[str, KVCacheSpec]) -> bool:
        if not kv_cache_specs:
            return True
        block_sizes = set(spec.block_size for spec in kv_cache_specs.values())
        if len(block_sizes) > 1:
            # Different block sizes, not uniform.
            return False
        first_spec = next(iter(kv_cache_specs.values()), None)
        if first_spec is None:
            return True
        return first_spec.is_uniform_with_collection(kv_cache_specs)"""
if old_uniform in t:
    t = t.replace(old_uniform, new_uniform)
    print("patched kv_cache_interface.py is_uniform_type")
else:
    print("WARNING kv_cache_interface.py is_uniform_type pattern not found")

p.write_text(t)

# 11) warmup.py
p = pathlib.Path("vllm/v1/worker/gpu/warmup.py")
t = p.read_text()
old_warmup = """    if isinstance(kvcache_spec, UniformTypeKVCacheSpecs):
        kvcache_spec = kvcache_spec.first_spec
    if isinstance(kvcache_spec, CircularBufferSpec):"""
new_warmup = """    if isinstance(kvcache_spec, UniformTypeKVCacheSpecs):
        kvcache_spec = kvcache_spec.first_spec
    if kvcache_spec is None:
        return 0
    if isinstance(kvcache_spec, CircularBufferSpec):"""
if old_warmup in t:
    t = t.replace(old_warmup, new_warmup)
    print("patched warmup.py first_spec")
else:
    print("WARNING warmup.py pattern not found")
p.write_text(t)

print("all patched")
