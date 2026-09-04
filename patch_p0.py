import pathlib, re, os, sys

def patch_mamba_utils(root="/opt/vllm"):
    p = pathlib.Path(root) / "vllm/v1/worker/mamba_utils.py"
    t = p.read_text()
    # First hunk: bt_row_idx = batch_idx if HAS_IDX_MAPPING else req_idx  -> bt_row_idx = req_idx
    old = """    # Skip no-op self-copy.
    if src_block_idx == dest_block_idx and accept_token_bias == 0:
        return

    bt_row_idx = batch_idx if HAS_IDX_MAPPING else req_idx"""
    new = """    # Skip no-op self-copy.
    if src_block_idx == dest_block_idx and accept_token_bias == 0:
        return

    # FIX (ours): the captured block tables are the SOURCE per-request-slot
    # tables, so always index them by req_idx. The previous batch-row indexing
    # read the CURRENT step's gathered tables at a (possibly stale) batch row
    # -- on a non-last PP rank the deferred postprocess runs pp_size steps
    # after its batch was gathered, so batch rows point at DIFFERENT requests
    # and the align state copy walks another request's block ids.
    bt_row_idx = req_idx"""
    if old in t:
        t = t.replace(old, new)
        print("patched mamba_utils bt_row_idx")
    else:
        print("WARNING mamba_utils bt_row_idx pattern not found")
        # fallback regex
        if "bt_row_idx = batch_idx if HAS_IDX_MAPPING else req_idx" in t:
            t = t.replace("bt_row_idx = batch_idx if HAS_IDX_MAPPING else req_idx", "bt_row_idx = req_idx  # FIX req_idx")
            print("fallback patched bt_row_idx")

    # Second hunk: in triton caller where batch_idx passed as req_idx
    # Search for the second occurrence: _copy_mamba_state_block with batch_idx
    # The stock second location is inside a function where token_bias = tl.load(token_bias_ptr + req_idx)
    # and then _copy_mamba_state_block(state_idx, batch_idx, ...)
    # We replace batch_idx with req_idx there
    # Let's do targeted replace: look for pattern around token_bias
    old2 = "    token_bias = tl.load(token_bias_ptr + req_idx)\n    _copy_mamba_state_block(\n        state_idx,\n        batch_idx,"
    new2 = "    token_bias = tl.load(token_bias_ptr + req_idx)\n    _copy_mamba_state_block(\n        state_idx,\n        req_idx,  # FIX (ours): source tables are req-indexed"
    if old2 in t:
        t = t.replace(old2, new2)
        print("patched mamba_utils batch_idx->req_idx second site")
    else:
        # try broader
        if "        batch_idx,\n        src_col," in t:
            # find the triton kernel call with batch_idx
            # replace the second occurrence after the first patch
            # Count occurrences and replace the second
            parts = t.split("        batch_idx,")
            if len(parts) >= 3:
                # first already patched? second is at index 1? Let's just replace all remaining batch_idx that are second arg of _copy_mamba_state_block
                # Use regex for the triton call
                t2 = re.sub(r"(_copy_mamba_state_block\(\s+state_idx,\s+)batch_idx,", r"\1req_idx,  # FIX req_idx", t)
                if t2 != t:
                    t = t2
                    print("regex patched second batch_idx")
                else:
                    print("WARNING second batch_idx pattern not found")
            else:
                print("WARNING second site not found via split")
        else:
            print("WARNING second batch_idx pattern not found")
    p.write_text(t)

def patch_block_table(root="/opt/vllm"):
    p = pathlib.Path(root) / "vllm/v1/worker/gpu/block_table.py"
    t = p.read_text()
    old = """        # Block tables used for model's forward pass.
        # num_kv_cache_groups x [max_num_reqs, max_num_blocks]
        self.input_block_tables: list[torch.Tensor] = [
            torch.zeros_like(b.gpu) for b in self.block_tables
        ]"""
    new = """        # Block tables used for model's forward pass.
        # num_kv_cache_groups x [max_num_reqs, max_num_blocks]
        # FIX (ours): a round-robin POOL of gathered-table sets. With one set,
        # a later step's gather mutates the buffers while an earlier in-flight
        # step's kernels may still read them (PP keeps up to
        # max_concurrent_batches steps in flight). Rotating over
        # VLLM_BT_POOL >= max_concurrent_batches + 1 sets gives each step an
        # immutable view for its whole in-flight lifetime, with stable
        # per-slot data_ptrs (unlike per-step clones, which dangle under the
        # raw-pointer captures downstream).
        import os as _os
        self._num_table_sets = max(1, int(_os.environ.get("VLLM_BT_POOL", "1")))
        self._table_set_idx = 0
        self.input_block_tables_pool: list[list[torch.Tensor]] = [
            [torch.zeros_like(b.gpu) for b in self.block_tables]
            for _ in range(self._num_table_sets)
        ]
        self.input_block_tables: list[torch.Tensor] = (
            self.input_block_tables_pool[0]
        )"""
    if old in t:
        t = t.replace(old, new)
        print("patched block_table init")
    else:
        print("WARNING block_table init pattern not found")
        # fallback
        if "self.input_block_tables: list[torch.Tensor] = [" in t:
            print("found alternative block_table init")
    old2 = "        self.input_block_table_ptrs = self._make_ptr_tensor(self.input_block_tables)"
    new2 = """        self.input_block_table_ptrs_pool = [
            self._make_ptr_tensor(s) for s in self.input_block_tables_pool
        ]
        self.input_block_table_ptrs = self.input_block_table_ptrs_pool[
            self._table_set_idx
        ]"""
    if old2 in t:
        t = t.replace(old2, new2)
        print("patched block_table ptrs pool")
    else:
        print("WARNING block_table ptrs pattern not found")

    old3 = """        if self.num_kv_cache_groups == 0:
            return ()
        if out is None:
            out = tuple(self.input_block_tables)
            out_ptrs = self.input_block_table_ptrs"""
    # This may have different spacing, check local file
    # The actual stock in our copy is:
    # if out is None:
    #     out = tuple(self.input_block_tables)
    #     out_ptrs = self.input_block_table_ptrs
    # Let's handle with regex
    if old3 in t:
        new3 = """        if self.num_kv_cache_groups == 0:
            return ()
        if out is None:
            # Rotate to the next pooled set (see __init__).
            self._table_set_idx = (
                self._table_set_idx + 1
            ) % self._num_table_sets
            self.input_block_tables = self.input_block_tables_pool[
                self._table_set_idx
            ]
            self.input_block_table_ptrs = self.input_block_table_ptrs_pool[
                self._table_set_idx
            ]
            out = tuple(self.input_block_tables)
            out_ptrs = self.input_block_table_ptrs"""
        t = t.replace(old3, new3)
        print("patched block_table rotate")
    else:
        # try to find the gather function
        if "if out is None:" in t and "out = tuple(self.input_block_tables)" in t:
            # do regex replace
            import re
            pattern = r"if out is None:\s+out = tuple\(self\.input_block_tables\)\s+out_ptrs = self\.input_block_table_ptrs"
            repl = """if out is None:
            # Rotate to the next pooled set (see __init__).
            self._table_set_idx = (
                self._table_set_idx + 1
            ) % self._num_table_sets
            self.input_block_tables = self.input_block_tables_pool[
                self._table_set_idx
            ]
            self.input_block_table_ptrs = self.input_block_table_ptrs_pool[
                self._table_set_idx
            ]
            out = tuple(self.input_block_tables)
            out_ptrs = self.input_block_table_ptrs"""
            t2 = re.sub(pattern, repl, t)
            if t2 != t:
                t = t2
                print("regex patched block_table rotate")
            else:
                print("WARNING block_table rotate pattern not found")
                print(repr(t[t.find("if out is None:"):t.find("if out is None:")+500]))
        else:
            print("WARNING block_table rotate not found")

    # PP projection preserves global KV-cache group indices. Groups with no
    # layers on this worker are represented by zero-width block tables and
    # slot_mapping_enabled=False. tl.where does not predicate the later load,
    # so the stock kernel dereferences the null table pointer on that group.
    old4 = """        block_numbers = tl.load(
            block_table_ptr + req_state_idx * block_table_stride + block_indices,
            mask=is_local,
            other=0,
        )"""
    new4 = """        block_numbers = tl.load(
            block_table_ptr + req_state_idx * block_table_stride + block_indices,
            # PP: disabled/empty groups have a zero-width (null) block table.
            # Predicate the load itself; tl.where above does not prevent it.
            mask=(offset < end_idx) & is_local & mapping_enabled,
            other=0,
        )"""
    if old4 in t:
        t = t.replace(old4, new4)
        print("patched block_table disabled-group slot-mapping load")
    elif "mask=(offset < end_idx) & is_local & mapping_enabled" in t:
        print("block_table disabled-group load already patched")
    else:
        print("WARNING block_table slot-mapping load pattern not found")

    # The scheduler sends block IDs for every global KV-cache group to every
    # PP worker. A projected group with no layers on this stage deliberately
    # has a zero-width table. Do not stage global IDs into that absent table.
    old5 = """            row_capacity = self.block_tables[i].gpu.shape[1]
            if end > row_capacity:
                raise RuntimeError("""
    new5 = """            row_capacity = self.block_tables[i].gpu.shape[1]
            if row_capacity == 0:
                self.num_blocks.np[i, req_index] = 0
                continue
            if end > row_capacity:
                raise RuntimeError("""
    if old5 in t:
        t = t.replace(old5, new5)
        print("patched block_table zero-capacity PP group append")
    elif "if row_capacity == 0:" in t:
        print("block_table zero-capacity group append already patched")
    else:
        print("WARNING block_table zero-capacity append pattern not found")
    p.write_text(t)

def patch_single_type(root="/opt/vllm"):
    p = pathlib.Path(root) / "vllm/v1/core/single_type_kv_cache_manager.py"
    t = p.read_text()
    old = """        assert dcp_world_size == 1, "DCP not support mamba now."
        assert pcp_world_size == 1, "PCP not support mamba now."
        block_hashes = resolve_block_hashes("""
    new = """        assert dcp_world_size == 1, "DCP not support mamba now."
        assert pcp_world_size == 1, "PCP not support mamba now."
        # FIX (port of unmerged vllm#48375): EAGLE/MTP requires the final
        # matched page of a cache hit be dropped -- its recurrent-state
        # snapshot may include draft tokens that verification later rejects,
        # so resuming from it corrupts the GDN/mamba state. Upstream lowers
        # max_num_blocks by one; lowering max_length by one mamba block is
        # arithmetically identical for the coarse loop ((x-b)//b == x//b - 1)
        # and ALSO bounds the fine-grained partial-unit branch below, which
        # upstream's diff predates.
        if drop_eagle_block:
            max_length = max(0, max_length - kv_cache_spec.block_size)
        block_hashes = resolve_block_hashes("""
    if old in t:
        # Only patch MambaManager.find_longest_cache_hit (the one at line 1385)
        # There are multiple identical asserts, so replace only the Mamba one
        # Check if we are patching the right place: MambaManager is at class def search
        # The first occurrence of this pattern in MambaManager is at 1385
        # We'll replace the first occurrence after "class MambaManager"
        # Simple: count occurrences and replace the last one which is MambaManager
        # The file has multiple classes with same pattern; we need to ensure we patch MambaManager
        # MambaManager is the last class with that assert pattern before line 1395
        # Let's replace all occurrences where drop_eagle_block is param but we only should patch MambaManager
        # Safer: replace only when followed by resolve_block_hashes with kv_cache_spec.block_size
        # Check if drop_eagle_block already patched
        if "if drop_eagle_block:" in t and "max_length - kv_cache_spec.block_size" in t:
            print("single_type already patched (skip)")
        else:
            # Replace the MambaManager occurrence: we can use a more specific old that includes the comment
            # Use split and replace only the MambaManager section
            # Find MambaManager class
            idx = t.find("class MambaManager")
            if idx != -1:
                before = t[:idx]
                after = t[idx:]
                if old in after:
                    # replace first occurrence in after
                    after = after.replace(old, new, 1)
                    t = before + after
                    print("patched single_type MambaManager drop_eagle_block")
                else:
                    print("WARNING single_type old not found in MambaManager section")
            else:
                t = t.replace(old, new, 1)
                print("patched single_type (fallback)")
    else:
        print("WARNING single_type pattern not found")
    p.write_text(t)

def patch_mamba_hybrid(root="/opt/vllm"):
    p = pathlib.Path(root) / "vllm/v1/worker/gpu/model_states/mamba_hybrid.py"
    t = p.read_text()
    old = """    def add_request(self, req_index: int, new_req_data: NewRequestData) -> None:
        super().add_request(req_index, new_req_data)
        # Must reset the speculative acceptance count in this idx which could be stale.
        self.num_accepted_tokens_gpu[req_index].fill_(1)
        if self._align_mode:
            # Seed the running state block from the resumed/prefilled position.
            self._mamba_state_idx_gpu[req_index].fill_(
                (new_req_data.num_computed_tokens - 1) // self.cache_config.block_size
            )"""
    new = """    def add_request(self, req_index: int, new_req_data: NewRequestData) -> None:
        super().add_request(req_index, new_req_data)
        # Must reset the speculative acceptance count in this idx which could be stale.
        self.num_accepted_tokens_gpu[req_index].fill_(1)
        if self._align_mode:
            # Seed the running state block from the resumed/prefilled position.
            # FIX (vllm#53142, no upstream PR): the divisor must be the MAMBA
            # group's block size, not the attention/CLI block size -- on hybrids
            # they differ, and a resume over a cached prefix then seeds an
            # out-of-range block_table column, so the fused align precopy reads
            # a garbage block id (IMA on sm_121; silently wrong state where the
            # read stays mapped). _mamba_spec is populated by the first batch's
            # preprocess; fresh requests seed idx from num_computed_tokens=0
            # either way, so the fallback is safe.
            mamba_bs = (
                self._mamba_spec.block_size
                if self._mamba_spec is not None
                else self.cache_config.block_size
            )
            self._mamba_state_idx_gpu[req_index].fill_(
                (new_req_data.num_computed_tokens - 1) // mamba_bs
            )"""
    if old in t:
        t = t.replace(old, new)
        print("patched mamba_hybrid divisor")
    else:
        print("WARNING mamba_hybrid pattern not found")
        if "self._mamba_state_idx_gpu[req_index].fill" in t:
            print("found alternative mamba_hybrid line")
            # try regex
            import re
            pattern = r"self\._mamba_state_idx_gpu\[req_index\]\.fill_\(\s+\(new_req_data\.num_computed_tokens - 1\) // self\.cache_config\.block_size\s+\)"
            repl = """            mamba_bs = (
                self._mamba_spec.block_size
                if self._mamba_spec is not None
                else self.cache_config.block_size
            )
            self._mamba_state_idx_gpu[req_index].fill_(
                (new_req_data.num_computed_tokens - 1) // mamba_bs
            )"""
            t2 = re.sub(pattern, repl, t)
            if t2 != t:
                t = t2
                print("regex patched mamba_hybrid")
    p.write_text(t)

def patch_ple_layer(root="/opt/vllm"):
    for sub in ["nvidia", "amd"]:
        p = pathlib.Path(root) / f"vllm/models/qwen4_exp/{sub}/ple_layer.py"
        if not p.exists():
            print(f"skip ple_layer {sub} not found")
            continue
        t = p.read_text()
        old = """            existing_state[..., : self.conv_state_len] = safe_next_state
            conv_state.index_copy_(0, state_indices, existing_state)
        return output

    def _short_conv_dilated_spec_batched("""
        # The actual file has this exact sequence in _short_conv_dilated_causal (not spec)
        # Let's check zeb's target: it was around line 872 in nvidia/ple_layer.py
        # That file's function is _short_conv_causal or _short_conv_dilated?
        # We need to find the correct insertion point: after safe_next_state, before index_copy
        # There are two places; we want the first (non-spec) causal one that handles prefill
        # The zeb patch is in ple_layer.py around existing_state assignment
        # Let's patch both occurrences if found
        # Try to find the pattern with update_mask
        if old in t:
            new = """            existing_state[..., : self.conv_state_len] = safe_next_state
            # HARDENING (ours): initialize the speculative-extension columns
            # [conv_state_len:] whenever prefill (re)initializes a state
            # block. Prefill/decode only write [0:conv_state_len]; on a fresh
            # cache block the tail is uninitialized VRAM (NaN patterns), and
            # the spec-decode rollback gather can reach into it before the
            # first full spec write heals it. Zero is the correct
            # "no history" value.
            if existing_state.shape[-1] > self.conv_state_len:
                tail = existing_state[..., self.conv_state_len :]
                existing_state[..., self.conv_state_len :] = torch.where(
                    update_mask.view(num_prefills, 1, 1).expand_as(tail),
                    torch.zeros_like(tail),
                    tail,
                )
            conv_state.index_copy_(0, state_indices, existing_state)
        return output

    def _short_conv_dilated_spec_batched("""
            # Check if already patched
            if "HARDENING" in t:
                print(f"ple_layer {sub} already patched skip")
            else:
                t = t.replace(old, new)
                print(f"patched ple_layer {sub}")
                p.write_text(t)
        else:
            # try alternative: the file may have spec version with update_mask defined
            if "existing_state[..., : self.conv_state_len] = safe_next_state" in t:
                # patch all occurrences with tail zero init if update_mask exists nearby
                print(f"ple_layer {sub} old2 not found, trying fallback for spec tail")
                # Find the block and add after safe_next_state where update_mask is in scope
                # For now try to replace any safe_next_state assignment that is followed by index_copy
                pattern = r"existing_state\[\.\.\., : self\.conv_state_len\] = safe_next_state\s+conv_state\.index_copy_"
                if re.search(pattern, t):
                    # Need to ensure we don't double-patch
                    if "HARDENING" not in t:
                        # Insert hardening for the prefill path that has update_mask
                        # We'll do a more specific replace for the first occurrence that has update_mask nearby
                        # Look for update_mask definition
                        # Simple: replace safe_next_state + index_copy with hardened version if shape check
                        # Let's do manual
                        old2 = "            existing_state[..., : self.conv_state_len] = safe_next_state\n            conv_state.index_copy_(0, state_indices, existing_state)"
                        new2 = """            existing_state[..., : self.conv_state_len] = safe_next_state
            # HARDENING (spec tail zero-init)
            if existing_state.shape[-1] > self.conv_state_len:
                tail = existing_state[..., self.conv_state_len :]
                # update_mask may not be defined in spec path; guard
                try:
                    existing_state[..., self.conv_state_len :] = torch.where(
                        update_mask.view(num_prefills, 1, 1).expand_as(tail),
                        torch.zeros_like(tail),
                        tail,
                    )
                except NameError:
                    existing_state[..., self.conv_state_len :] = torch.zeros_like(existing_state[..., self.conv_state_len :])
            conv_state.index_copy_(0, state_indices, existing_state)"""
                        if old2 in t:
                            t = t.replace(old2, new2, 1)
                            print(f"fallback patched ple_layer {sub}")
                            p.write_text(t)
                        else:
                            print(f"WARNING ple_layer {sub} fallback old not found")
                    else:
                        print(f"ple_layer {sub} already has hardening")
                else:
                    print(f"WARNING ple_layer {sub} pattern not found")
                    # debug
                    idx = t.find("existing_state[..., : self.conv_state_len] = safe_next_state")
                    print(t[idx-200:idx+400][:500])
            else:
                print(f"WARNING ple_layer {sub} no safe_next_state found")

def patch_connector(root="/opt/vllm"):
    p = pathlib.Path(root) / "vllm/v1/ple_offload/connector.py"
    t = p.read_text()
    old_init = """        self._request_queue: queue.Queue[PleOffloadRequest | None] = queue.Queue(
            maxsize=1
        )
        self._request_thread: threading.Thread | None = None"""
    new_init = """        self._request_queue: queue.Queue[PleOffloadRequest | None] = queue.Queue(
            maxsize=1
        )
        # PP fix: _process_request stages inputs by copying from the runner's
        # LIVE input buffers, which the next forward overwrites. At PP=1 the
        # natural cadence keeps model thread and sender in step. Under pipeline
        # parallelism rank 0 issues the next forward before the sender has
        # staged the previous one, which either overflows the queue
        # (queue.Full, killing the worker) or silently stages the wrong inputs.
        # Gate each launch on staging of the previous request having completed.
        # Only the small input copy is serialised, not the forward.
        self._staged = threading.Event()
        self._staged.set()
        self._request_thread: threading.Thread | None = None"""
    if old_init in t:
        t = t.replace(old_init, new_init)
        print("patched connector init _staged")
    else:
        print("WARNING connector init pattern not found")
        if "_staged" in t:
            print("already has _staged")
        else:
            # try alternative spacing
            if "self._request_queue: queue.Queue" in t:
                print("found alternative connector queue")
    old_process = """        except Exception:
            logger.exception("PLE request thread failed")
            os._exit(1)"""
    new_process = """        except Exception:
            logger.exception("PLE request thread failed")
            self._staged.set()  # do not strand a model thread waiting on us
            os._exit(1)"""
    if old_process in t:
        t = t.replace(old_process, new_process)
        print("patched connector exception _staged.set")
    else:
        print("WARNING connector exception pattern not found")
    old_copy = """        else:
            self._copy_cpu_inputs(request)

        with torch.cuda.nvtx.range("ple_offload.send_request"):"""
    new_copy = """        else:
            self._copy_cpu_inputs(request)

        # Inputs are now copied out of the runner's live buffers; the model
        # thread may safely produce the next forward's inputs.
        self._staged.set()
        with torch.cuda.nvtx.range("ple_offload.send_request"):"""
    if old_copy in t:
        t = t.replace(old_copy, new_copy)
        print("patched connector _copy staged.set")
    else:
        print("WARNING connector copy pattern not found")
    old_put = """        self._request_queue.put_nowait(request)

    def prepare_forward("""
    new_put = """        # Block until the sender finished staging the previous request, so the
        # runner cannot overwrite the input source mid-copy. Times out rather
        # than hanging the engine if the sender thread has died.
        if not self._staged.wait(timeout=60.0):
            raise RuntimeError(
                "PLE offload: previous request was not staged within 60s; "
                "the request thread may have stalled."
            )
        self._staged.clear()
        self._request_queue.put(request, timeout=60.0)

    def prepare_forward("""
    if old_put in t:
        t = t.replace(old_put, new_put)
        print("patched connector put handshake")
    else:
        print("WARNING connector put pattern not found")
        if "_staged.wait" in t:
            print("already has handshake")
    p.write_text(t)

def patch_pp_warmup_broadcast(root="/opt/vllm"):
    p = pathlib.Path(root) / "vllm/v1/worker/gpu/pp_utils.py"
    t = p.read_text()

    # Dedicated subgroup for the sampled-token broadcast with eager init
    sibling_old = """        self.broadcast_group = get_pp_group().make_sibling_device_group(
            group_desc="pp_broadcast"
        )"""
    sibling_new = """        self.broadcast_group = get_pp_group().make_sibling_device_group(
            group_desc="pp_broadcast"
        )
        self.warmup_sync = False
        # Initialize this NCCL communicator before request-time activation P2P
        # can initialize a different communicator concurrently on another
        # stream. Concurrent lazy NCCL initialization deadlocks on PP=2.
        comm_init = torch.zeros(1, dtype=torch.int32, device=self.device)
        torch.distributed.broadcast(
            comm_init, src=self.last_rank, group=self.broadcast_group
        )
        self.main_stream.synchronize()"""
    if sibling_old in t:
        t = t.replace(sibling_old, sibling_new, 1)
        print("patched PP sampled-token communicator eager init")
    elif "Concurrent lazy NCCL initialization deadlocks" in t:
        print("PP communicator eager init already patched")
    else:
        print("WARNING PP communicator eager-init pattern not found")

    # Replace receive method to be rank-invariant and synchronize event
    old_recv_full = """    def receive(self, input_batch: InputBatch) -> bool:
        \"\"\"Returns True iff sampled tokens need to be gathered from *all*
        requests in the batch.\"\"\"
        assert not self.is_last_rank
        need_sampled_mask = compute_need_sampled_mask(input_batch)
        if need_sampled_mask is None:
            # Leave this step's reserved slot as None.
            return False

        # Snapshot the per-slot generation counter so a later free of any of
        # these RequestStates request indices is detectable at consume time.
        gen_at_receive_np = self.req_idx_gen_np[input_batch.idx_mapping_np]

        num_reqs = input_batch.num_reqs
        with torch.cuda.stream(self.broadcast_stream):
            self.broadcast_stream.wait_stream(self.main_stream)
            sampled_tokens = torch.empty(
                num_reqs, self.max_sample_len, dtype=torch.int64, device=self.device
            )
            combined = torch.empty(2, num_reqs, dtype=torch.int32, device=self.device)
            torch.distributed.broadcast(
                sampled_tokens, src=self.last_rank, group=self.broadcast_group
            )
            torch.distributed.broadcast(
                combined, src=self.last_rank, group=self.broadcast_group
            )
            event = self.broadcast_stream.record_event()
            num_sampled, num_rejected = combined.unbind(dim=0)
            # Must record_stream since these were allocated on broadcast stream but
            # later used on the main stream.
            sampled_tokens.record_stream(self.main_stream)
            combined.record_stream(self.main_stream)
        self.queue[-1] = PendingRecv(
            event,
            sampled_tokens,
            num_sampled,
            num_rejected,
            input_batch.idx_mapping,
            input_batch.idx_mapping_np,
            need_sampled_mask,
            gen_at_receive_np,
        )
        return bool(need_sampled_mask.all())"""

    new_recv_full = """    def receive(self, input_batch: InputBatch) -> bool:
        \"\"\"Returns True iff sampled tokens need to be gathered from *all*
        requests in the batch.\"\"\"
        assert not self.is_last_rank
        need_sampled_mask = compute_need_sampled_mask(input_batch)
        if need_sampled_mask is None:
            need_sampled_mask = np.full(
                input_batch.num_reqs, self.warmup_sync, dtype=np.bool_
            )

        # Snapshot the per-slot generation counter so a later free of any of
        # these RequestStates request indices is detectable at consume time.
        gen_at_receive_np = self.req_idx_gen_np[input_batch.idx_mapping_np]

        num_reqs = input_batch.num_reqs
        with torch.cuda.stream(self.broadcast_stream):
            self.broadcast_stream.wait_stream(self.main_stream)
            sampled_tokens = torch.empty(
                num_reqs, self.max_sample_len, dtype=torch.int64, device=self.device
            )
            combined = torch.empty(2, num_reqs, dtype=torch.int32, device=self.device)
            torch.distributed.broadcast(
                sampled_tokens, src=self.last_rank, group=self.broadcast_group
            )
            torch.distributed.broadcast(
                combined, src=self.last_rank, group=self.broadcast_group
            )
            event = self.broadcast_stream.record_event()
            num_sampled, num_rejected = combined.unbind(dim=0)
            # Must record_stream since these were allocated on broadcast stream but
            # later used on the main stream.
            sampled_tokens.record_stream(self.main_stream)
            combined.record_stream(self.main_stream)
        # Do not overlap the sampled-token collective with the next PP P2P.
        event.synchronize()
        self.queue[-1] = PendingRecv(
            event,
            sampled_tokens,
            num_sampled,
            num_rejected,
            input_batch.idx_mapping,
            input_batch.idx_mapping_np,
            need_sampled_mask,
            gen_at_receive_np,
        )
        return bool(need_sampled_mask.all())"""

    if old_recv_full in t:
        t = t.replace(old_recv_full, new_recv_full, 1)
        print("patched PP receive method")
    else:
        print("WARNING PP old_recv_full pattern not found; trying fallback")

    # Replace broadcast method to pad to max_sample_len
    old_broadcast_full = """    def broadcast(
        self,
        sampled_token_ids: torch.Tensor,
        num_sampled: torch.Tensor,
        num_rejected: torch.Tensor,
        input_batch: InputBatch,
    ) -> None:
        assert self.is_last_rank
        if compute_need_sampled_mask(input_batch) is None:
            # No request needs sampled outputs for a subsequent decode step.
            return

        assert sampled_token_ids.dtype == torch.int64

        if current_platform.is_xpu():
            self.main_stream.synchronize()

        with torch.cuda.stream(self.broadcast_stream):
            self.broadcast_stream.wait_stream(self.main_stream)
            torch.distributed.broadcast(
                sampled_token_ids.contiguous(),
                src=self.last_rank,
                group=self.broadcast_group,
            )
            combined = torch.stack((num_sampled, num_rejected), dim=0)
            torch.distributed.broadcast(
                combined, src=self.last_rank, group=self.broadcast_group
            )
            for tensor in (sampled_token_ids, num_sampled, num_rejected):
                tensor.record_stream(self.broadcast_stream)"""

    new_broadcast_full = """    def broadcast(
        self,
        sampled_token_ids: torch.Tensor,
        num_sampled: torch.Tensor,
        num_rejected: torch.Tensor,
        input_batch: InputBatch,
    ) -> None:
        assert self.is_last_rank
        # Always match the non-last rank's collectives. Whether these outputs
        # are retained is a receiver-local policy represented by its mask.
        assert sampled_token_ids.dtype == torch.int64

        num_reqs = input_batch.num_reqs
        if sampled_token_ids.shape[1] < self.max_sample_len:
            padded = torch.zeros(
                (num_reqs, self.max_sample_len),
                dtype=torch.int64,
                device=self.device,
            )
            padded[:, : sampled_token_ids.shape[1]] = sampled_token_ids
            to_broadcast = padded
        else:
            to_broadcast = sampled_token_ids.contiguous()

        if current_platform.is_xpu():
            self.main_stream.synchronize()

        with torch.cuda.stream(self.broadcast_stream):
            self.broadcast_stream.wait_stream(self.main_stream)
            torch.distributed.broadcast(
                to_broadcast,
                src=self.last_rank,
                group=self.broadcast_group,
            )
            combined = torch.stack((num_sampled, num_rejected), dim=0)
            torch.distributed.broadcast(
                combined, src=self.last_rank, group=self.broadcast_group
            )
            for tensor in (to_broadcast, num_sampled, num_rejected):
                tensor.record_stream(self.broadcast_stream)
        # Match the receiver before either rank advances to the next P2P.
        self.broadcast_stream.synchronize()"""

    if old_broadcast_full in t:
        t = t.replace(old_broadcast_full, new_broadcast_full, 1)
        print("patched PP broadcast method")
    else:
        print("WARNING PP old_broadcast_full pattern not found")

    p.write_text(t)

def patch_warmup_pp_mode(root="/opt/vllm"):
    p = pathlib.Path(root) / "vllm/v1/worker/gpu/warmup.py"
    t = p.read_text()
    old = """    # Disable KV connector for warmup run.
    model_runner.kv_connector.set_disabled(True)
    worker_execute_model(prefill_output)"""
    new = """    # Disable KV connector for warmup run.
    model_runner.kv_connector.set_disabled(True)
    # PP stage-local synthetic metadata can disagree about whether sampling is
    # needed. Force matched, synchronous side-stream collectives for warmup.
    if model_runner.pp_handler is not None:
        model_runner.pp_handler.warmup_sync = True
    worker_execute_model(prefill_output)"""
    if old in t:
        t = t.replace(old, new, 1)
        print("patched warmup PP mode enable")
    old_end = """    model_runner.kv_connector.set_disabled(False)
    torch.accelerator.synchronize()"""
    new_end = """    model_runner.kv_connector.set_disabled(False)
    if model_runner.pp_handler is not None:
        model_runner.pp_handler.warmup_sync = False
    torch.accelerator.synchronize()"""
    if old_end in t:
        t = t.replace(old_end, new_end, 1)
        print("patched warmup PP mode disable")
    p.write_text(t)

def patch_skip_v2_pp_synthetic_warmup(root="/opt/vllm"):
    p = pathlib.Path(root) / "vllm/v1/worker/gpu_worker.py"
    t = p.read_text()
    old = """        if self.use_v2_model_runner:
            # V2: Run full execute_model + sample_tokens to JIT compile triton kernels.
            warmup_kernels(self.model_runner, self.execute_model, self.sample_tokens)
        elif get_pp_group().is_last_rank:"""
    new = """        if self.use_v2_model_runner:
            # The synthetic V2 warmup constructs stage-local request metadata.
            # Under PP this can make the sampled-token side collective diverge
            # from the next activation P2P ordering and deadlock initialization.
            # CUDA graphs/model compilation are already complete; defer these
            # small sampler/state kernels to the first real scheduled request.
            if self.parallel_config.pipeline_parallel_size > 1:
                logger.warning(
                    "Skipping V2 synthetic kernel warmup under pipeline "
                    "parallelism; kernels will JIT on the first request."
                )
            else:
                warmup_kernels(
                    self.model_runner, self.execute_model, self.sample_tokens
                )
        elif get_pp_group().is_last_rank:"""
    if old in t:
        t = t.replace(old, new, 1)
        print("patched skip V2 PP synthetic warmup")
    else:
        print("WARNING V2 PP synthetic warmup pattern not found")
    p.write_text(t)

def patch_model_runner(root="/opt/vllm"):
    p = pathlib.Path(root) / "vllm/v1/worker/gpu/model_runner.py"
    t = p.read_text()
    old = """            self.model_state.preprocess_state(
                input_batch,
                block_tables,
                self.kv_cache_config,
                self.req_states.num_computed_tokens.gpu,
            )"""
    new = """            # FIX (ours): give the mamba spec-decode ctx the SOURCE
            # per-request-slot block tables (stable data_ptr, req-indexed,
            # stream-ordered staged writes) instead of the per-step gathered
            # views; the ctx captures raw pointers once, and gathered views
            # rotate/mutate under PP. Kernels index rows by req_idx to match
            # (mamba_utils.py).
            self.model_state.preprocess_state(
                input_batch,
                tuple(bt.gpu for bt in self.block_tables.block_tables),
                self.kv_cache_config,
                self.req_states.num_computed_tokens.gpu,
            )"""
    if old in t:
        t = t.replace(old, new, 1)
        print("patched model_runner master block_tables for preprocess_state")
        p.write_text(t)
    elif "tuple(bt.gpu for bt in self.block_tables.block_tables)" in t:
        print("model_runner already has master block_tables fix")
    else:
        print("WARNING model_runner preprocess_state pattern not found")

if __name__ == "__main__":
    root = sys.argv[1] if len(sys.argv) > 1 else "/opt/vllm"
    patch_mamba_utils(root)
    patch_block_table(root)
    patch_single_type(root)
    patch_mamba_hybrid(root)
    patch_ple_layer(root)
    patch_connector(root)
    patch_pp_warmup_broadcast(root)
    patch_warmup_pp_mode(root)
    patch_skip_v2_pp_synthetic_warmup(root)
    patch_model_runner(root)
    print("all P0+ done")

