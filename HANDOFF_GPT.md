# Comprehensive Handoff Document: Qwen 3.8 Flash Next PP=2 Deployment & Llama-Swap Integration

---

## 1. Executive Summary & Goal
The objective is to deploy the hybrid Transformer–Mamba MoE model **`vektorprime/qwen38-flash-next-pp2`** (quantized as `cyankiwi/Qwen3.8-Flash-Next-AWQ-INT4`, ~175 GiB safetensors across 38 shards) on a dual-GPU system and tuck it behind **`llama-swap`** as an OpenAI-compatible endpoint.

### Core Challenge & Architecture
1. **Model Architecture**: Hybrid architecture featuring standard attention, Mamba SSM linear attention layers, Mixture of Experts (MoE), Multi-Token Prediction (MTP) draft speculative heads, and N-gram PLE (Per-Layer Embedding) offloading.
2. **Hardware Constraints**:
   - 2x **NVIDIA CMP 170HX** GPUs (GPU 0 and GPU 1, ~70 GiB HBM2e VRAM each, capped at 150W power limit).
   - ~130 GiB host System RAM (used to offload ~95 GiB of PLE embedding tables via CPU worker).
3. **Parallelism Strategy**:
   - **Pipeline Parallelism (PP=2)**: GPU 0 (PP Stage 0, layers 0..31 + drives Gloo IPC to PLE CPU worker) + GPU 1 (PP Stage 1, layers 32..63 + MTP speculator).
   - **Tensor Parallelism (TP=1)**.

---

## 2. Infrastructure & System Specifications

| Component | Details |
| :--- | :--- |
| **Host / VM** | Proxmox VM 101 (`ollama.homelab`), IP: `192.168.1.30` |
| **SSH Access** | `ssh root@192.168.1.30` (or `ssh jonathan@192.168.1.30`) |
| **GPUs** | 2x NVIDIA CMP 170HX (Bus IDs `01:00.0` and `02:00.0`), 150W Power Cap |
| **NVIDIA Driver / CUDA** | Driver `610.57.04`, CUDA UMD `13.3`, PyTorch/NCCL `2.29.7` |
| **Host System Memory** | 130 GiB RAM allocated for PLE Offload Worker |
| **Projects Directory** | `/home/jonathan/Documents/projects/qwen38-flash-next-pp2` |
| **Llama-Swap Config** | `/home/jonathan/Documents/server-configs/docker/llama-swap/configs/qwen38-pp2.yaml` |
| **Llama-Swap Proxy Port** | `http://192.168.1.30:9292` |
| **Container Service Port** | `http://192.168.1.30:8001` (mapped to container port 8000) |

---

## 3. Deployment Artifacts & File Structure

### Key Files on Remote Host (`192.168.1.30`):
* **Project Directory**: `/home/jonathan/Documents/projects/qwen38-flash-next-pp2`
  * `Dockerfile`: Builds custom vLLM nightly with PR `#53899` (`qwen38` branch) and applies PP/P0 patch scripts.
  * `docker-compose.yml`: Deploys `qwen38-flash-next-api` container with GPU reservations, IPC shared memory (`shm_size: 32gb`), host networking / port forwarding (`8001:8000`), and model cache volume mappings.
  * `patch_pp.py`: Master Python script patching upstream vLLM codebase for pipeline parallelism compatibility.
  * `patch_p0.py`: P0 / Mamba buffer indexing and queue fix script.
* **Llama-Swap Configuration**:
  * `/home/jonathan/Documents/server-configs/docker/llama-swap/configs/qwen38-pp2.yaml`:
    ```yaml
    models:
      - name: "qwen38-flash-next-awq"
        type: "proxy"
        proxy_url: "http://qwen38-flash-next-api:8000"
        health_check: "http://qwen38-flash-next-api:8000/health"
    ```

---

## 4. Patches Implemented So Far (`patch_pp.py`)

Upstream vLLM (specifically PR #53899 for Qwen3.8 / Qwen4Exp) was designed under the assumption of single-GPU or TP-only execution without PP support for PLE offloading and hybrid KV caches. The following 12 critical patches were developed and injected during Docker build:

1. **`vllm/models/qwen4_exp/{nvidia,amd}/model_state.py`**:
   - Guarded `uses_ngram_embedding` to only activate on `get_pp_group().is_first_rank`.
   - Prevented non-first PP ranks (which receive intermediate hidden states instead of token IDs) from failing initialization.
2. **`vllm/model_executor/models/config.py`**:
   - Disabled the hard-coded `NotImplementedError` blocking PLE with `pipeline_parallel_size > 1`.
3. **`vllm/v1/worker/gpu_worker.py`**:
   - Removed the rejection of `parallel_config.pipeline_parallel_size != 1`.
4. **`vllm/models/qwen4_exp/{nvidia,amd}/model.py`**:
   - In Hyper-Connections (`skip_substrs`), added `hyper_connection_mixer.` to skip lists on non-last PP ranks so intermediate weights aren't expected where they don't exist.
5. **`vllm/v1/worker/gpu/model_runner.py` (`setup_ple_offload`)**:
   - Added `is_first_pp` guard in `setup_ple_offload` so non-first ranks cleanly bypass `query_start_loc_source` requirements.
6. **`vllm/models/qwen4_exp/{nvidia,amd}/mtp.py`**:
   - Patched assertion `assert hidden_states is not None` to only trigger if first PP rank or `hidden_states is not None`.
7. **`vllm/v1/ple_offload/connector.py`**:
   - Prevented non-first PP ranks from attempting to register non-existent PLE layers or launch Gloo IPC receivers.
8. **`vllm/v1/core/kv_cache_utils.py` (`_project_kv_cache_groups_to_worker`)**:
   - Filtered projected KV cache layer dictionaries per PP worker rank so each worker only tracks layers physically residing on its GPU.
9. **`vllm/v1/core/kv_cache_utils.py` (`_get_kv_cache_bytes_per_block` & `_max_memory_usage_bytes_from_groups`)**:
   - Filtered active groups (`[g for g in kv_cache_groups if g.layer_names]`) to avoid division-by-zero or empty-list max reductions.
10. **`vllm/v1/kv_cache_interface.py` (`UniformTypeKVCacheSpecs`)**:
    - Guarded `first_spec` (`next(iter(self.kv_cache_specs.values()), None)`), `max_memory_usage_pages`, `max_memory_usage_bytes`, and `is_uniform_type` to return default/safe values when `kv_cache_specs` is empty on a given PP stage.
11. **`vllm/v1/worker/gpu/model_runner.py` & `warmup.py`**:
    - Handled `layer_spec is None` in KV cache initialization loop (`slot_mapping_enabled.append(False)`, `max_num_blocks_per_group.append(0)`).
12. **`vllm/v1/worker/utils.py` (`allocate_kv_cache` & `prepare_kernel_block_sizes`)**:
    - In `allocate_kv_cache`, safely matched layers against group names.
    - In `prepare_kernel_block_sizes`, added empty guard for `kv_cache_spec.kv_cache_specs` and `group_backends` to prevent `StopIteration`.

---

## 5. Current State & Exact Error Encountered

### Current Behavior:
* **Shards & Weights**: All 38 safetensors shards load cleanly on Worker PP0 (GPU 0), Worker PP1 (GPU 1), and PleOffloadWorker (CPU RAM).
* **PLE Offload**: Successfully registers 1 PLE layer on GPU 0, enters busy-loop with Gloo IPC.
* **AOT Torch Compile Cache**: Successfully reconstructed from standalone compile artifacts in ~0.37s.
* **Failure Point**: During `compile_or_warm_up_model` -> `warmup_kernels` in `gpu_worker.py`:

```text
(Worker_PP1 pid=297) ERROR: File "/opt/vllm/vllm/v1/worker/gpu/warmup.py", line 332, in warmup_kernels
(Worker_PP1 pid=297) ERROR:   worker_execute_model(prefill_output)
(Worker_PP1 pid=297) ERROR: File "/opt/vllm/vllm/v1/worker/gpu_worker.py", line 1279, in execute_model
(Worker_PP1 pid=297) ERROR:   output = self.model_runner.execute_model(...)
(Worker_PP1 pid=297) ERROR: File "/opt/vllm/vllm/v1/worker/gpu/model_runner.py", line 1638, in execute_model
(Worker_PP1 pid=297) ERROR:   self.model_state.preprocess_state(...)
(Worker_PP1 pid=297) ERROR: File "/opt/vllm/vllm/v1/worker/gpu/model_states/mamba_hybrid.py", line 215, in preprocess_state
(Worker_PP1 pid=297) ERROR:   ctx = self._ensure_align_ctx(kv_cache_config, mamba_group_ids, block_tables)
(Worker_PP1 pid=297) ERROR: File "/opt/vllm/vllm/v1/worker/gpu/model_states/mamba_hybrid.py", line 188, in _ensure_align_ctx
(Worker_PP1 pid=297) ERROR:   ctx.initialize_from_forward_context(...)
(Worker_PP1 pid=297) ERROR: File "/opt/vllm/vllm/v1/worker/mamba_utils.py", line 974, in initialize_from_forward_context
(Worker_PP1 pid=297) ERROR:   self._populate_metadata(...)
(Worker_PP1 pid=297) ERROR: File "/opt/vllm/vllm/v1/worker/mamba_utils.py", line 1005, in _populate_metadata
(Worker_PP1 pid=297) ERROR:   self.state_base_addrs[idx] = _reinterpret_u64_as_i64(...)
(Worker_PP1 pid=297) ERROR: torch.AcceleratorError: CUDA error: an illegal memory access was encountered
```

---

## 6. Investigation & Diagnostic Insights for Next Agent

1. **Root Cause Analysis of `cudaErrorIllegalAddress` in Warmup**:
   - The traceback points to `self.state_base_addrs[idx] = _reinterpret_u64_as_i64(...)` inside `mamba_utils.py::_populate_metadata`.
   - In PyTorch CUDA runtime, `CUDA error: an illegal memory access` is often reported asynchronously on the subsequent CUDA memory write/copy after a prior CUDA kernel (e.g. earlier Triton kernel, Mamba state allocation, or P2P pipeline send/receive buffer) faulted.
   - **Key Suspicion 1**: In `warmup.py::warmup_kernels`, dummy inputs are passed to `execute_model`. Under Pipeline Parallelism (PP=2), `Worker_PP1` expects intermediate activations sent from `Worker_PP0` via NCCL P2P. If `warmup_kernels` executes workers independently or with mismatched dummy batch structures, PP rank 1 may read uninitialized/out-of-bounds peer memory or P2P buffers.
   - **Key Suspicion 2**: In `mamba_hybrid.py::_ensure_align_ctx`, `block_tables` on PP1 may reference KV cache block IDs or Mamba state pools that were sized differently on GPU 1 than GPU 0.
   - **Key Suspicion 3**: Inspect `CUDA_LAUNCH_BLOCKING=1` to find the exact synchronous line triggering the illegal memory access.

2. **Useful Verification Commands**:
   - Run interactive test with `CUDA_LAUNCH_BLOCKING=1`:
     ```bash
     ssh root@192.168.1.30 "docker exec -it qwen38-flash-next-api bash"
     ```
   - Check container logs in real time:
     ```bash
     ssh root@192.168.1.30 "docker logs -f qwen38-flash-next-api"
     ```
   - Test OpenAI-compatible endpoint once server finishes warmup:
     ```bash
     curl -s http://192.168.1.30:8001/v1/models
     curl -s http://192.168.1.30:8001/v1/chat/completions \
       -H "Content-Type: application/json" \
       -d '{"model":"qwen38-flash-next-awq","messages":[{"role":"user","content":"Hello!"}],"max_tokens":32}'
     ```
   - Test Llama-Swap routing:
     ```bash
     curl -s http://192.168.1.30:9292/v1/chat/completions \
       -H "Content-Type: application/json" \
       -d '{"model":"qwen38-flash-next-awq","messages":[{"role":"user","content":"Ping"}],"max_tokens":16}'
     ```
