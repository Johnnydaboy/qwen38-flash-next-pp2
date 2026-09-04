# Final Handoff & Deployment Record: Qwen 3.8 Flash Next (PP=2) on Dual CMP 170HX with Llama-Swap

## 1. Executive Summary
The deployment of [`vektorprime/qwen38-flash-next-pp2`](https://github.com/vektorprime/qwen38-flash-next-pp2) (quantized AWQ checkpoint `cyankiwi/Qwen3.8-Flash-Next-AWQ-INT4`, 175.36 GiB across 38 shards) is **fully operational, robust, and verified across long sequence generations**.

The model runs under Pipeline Parallelism (**PP=2, TP=1**) on dual NVIDIA CMP 170HX GPUs with CPU-offloaded Per-Layer Embeddings (PLE) across ~130 GiB host RAM. It is integrated behind [`llama-swap`](https://github.com/mostlygeek/llama-swap) on port `9292` with automated on-demand start, health checking, and idle TTL management.

Crucially, the long-sequence token repetition collapse (`!` or repeating token loops occurring when decoding past token length ~1,568) has been **diagnosed to the root cause, permanently patched, and validated up to 3,425+ tokens with zero degradation**.

---

## 2. Hardware & Environment Topology

* **Target Host**: Proxmox VM 101 (`ollama.homelab`), IP: `192.168.1.30`
* **GPUs**: Dual **NVIDIA CMP 170HX** (~70 GiB HBM2e VRAM each, 150W power cap, Bus IDs `01:00.0` and `02:00.0`).
* **Host System RAM**: 130 GiB allocated to VM (used by `PleOffloadWorker` to store and compute ~95 GiB of embedding tables via Gloo IPC).
* **Network**: Docker bridge network `ai-net` (shared between `qwen38-flash-next-api` and `llama-swap`).
* **Direct Inference Port**: `http://192.168.1.30:8001` (`/v1/chat/completions`, `/v1/models`)
* **Llama-Swap Gateway Port**: `http://192.168.1.30:9292` (`/v1/chat/completions`, `/v1/models`)

```
                                  +-----------------------------------------------------+
                                  |                     llama-swap                      |
                                  |                    (Port 9292)                      |
                                  +--------------------------+--------------------------+
                                                             |
                                           Proxy & Lifecycle (Auto-Start/Stop)
                                                             |
                                  +--------------------------v--------------------------+
                                  |            qwen38-flash-next-api (Port 8001)         |
                                  |                vLLM V2 Engine Core                  |
                                  +--------------------------+--------------------------+
                                                             |
                        +------------------------------------+------------------------------------+
                        |                                                                         |
            +-----------v-----------+                                                 +-----------v-----------+
            |      Worker PP0       |                                                 |      Worker PP1       |
            |     (GPU 0, 47°C)     | <=========== Inter-Stage Activation P2P ======> |     (GPU 1, 40°C)     |
            |   Layers 0..31 + Vit  | <=========== Sampled Token Broadcast =========> |   Layers 32..63 + MTP |
            +-----------+-----------+                                                 +-----------------------+
                        |
                 Gloo IPC Socket
                        |
            +-----------v-----------+
            |   PleOffloadWorker    |
            |     (Host RAM)        |
            |   95 GB PLE Tables    |
            +-----------------------+
```

---

## 3. Key Issues Diagnosed & Patches Applied

### A. Long-Sequence Collapse at Token 1,568 (Root Cause & Resolution)
1. **The Phenomenon**:
   - Autoregressive generation ran at ~58 tok/s until sequence position 1,568, where generation immediately degenerated into continuous repeating tokens (`!` or repeating phrases like `ductductduct...`).
2. **The Structural Trigger**:
   - In hybrid models with Mamba and full attention layers, vLLM sets attention block size to match or exceed Mamba page size (`attn_block_size = 1568`).
   - In `--mamba-cache-mode align`, `mamba_block_size = 1568`.
   - Block 0 covers tokens `0..1567`. The very first token crossing into Block 1 is **token 1,568**.
3. **The Root Cause**:
   - At token 1,568, `preprocess_mamba_align_fused_kernel` advances `state_idx` from 0 to 1 and launches `precopy_mamba_align_fused_kernel` to migrate recurrent SSM/conv state from Block 0 to Block 1.
   - The kernel indexes into `ctx.block_table_ptrs` with row index `req_idx` to load `dest_block_id = block_table[req_idx, 1]`.
   - However, in `model_runner.py`:
     ```python
     block_tables, slot_mappings = self.prepare_attn(input_batch)
     self.model_state.preprocess_state(input_batch, block_tables, ...)
     ```
     `block_tables` passed here was `self.input_block_tables` (the gathered, rotated pool buffer), **not** the master persistent request-indexed block tables.
   - Furthermore, `VLLM_BT_POOL=5` rotated `input_block_tables` across 5 buffers each step. `ctx.block_table_ptrs` was captured once on step 0 pointing to `pool[1]`.
   - When Block 1 was allocated, column 1 was gathered into `pool[0]`. `pool[1][0, 1]` remained `0`.
   - The Triton align precopy kernel read `dest_block_id = 0`, copying state into Block 0 instead of the newly allocated physical block.
   - GDN attention then read the uninitialized physical block (full of zeros), emitting NaNs on PP0 at pos 1,568. These propagated to PP1 and collapsed generation into repetition loops.
4. **The Fix**:
   - **Master Table Binding**: In `model_runner.py`, `preprocess_state` now receives the static, persistent, request-indexed master GPU block tables:
     ```python
     master_block_tables = tuple(bt.gpu for bt in self.block_tables.block_tables)
     self.model_state.preprocess_state(input_batch, master_block_tables, ...)
     ```
   - **Single Buffering**: Set `VLLM_BT_POOL=1` in `docker-compose.yml`, eliminating pool rotation disconnects.
   - **Verification**: Verified zero NaNs and seamless decode past 1,600, 2,048, and 3,425 tokens across multiple block transitions (Block 0 $\rightarrow$ 1 at 1,568; Block 1 $\rightarrow$ 2 at 3,136).

---

### B. Pipeline Parallelism Layer & Cache Projection
5. **N-Gram Embedding Input Guard (`model_state.py`)**: Guarded `uses_ngram_embedding` to execute only on `get_pp_group().is_first_rank` (PP0 receives token IDs; PP1 receives hidden states).
6. **PLE & PP Check Removal (`config.py` & `gpu_worker.py`)**: Removed hard-coded `NotImplementedError` and worker validations rejecting PLE when `pipeline_parallel_size > 1`.
7. **Hyper-Connection Mixer Substring Filter (`model.py`)**: Added `hyper_connection_mixer.` to `skip_substrs` on non-last PP ranks so intermediate weights aren't expected on PP0.
8. **PLE Offload Rank Guard (`model_runner.py` & `connector.py`)**: Bypassed `setup_ple_offload`, socket registration, and worker threads on non-first PP ranks.
9. **Per-Worker KV Cache Projection (`kv_cache_utils.py` & `kv_cache_interface.py`)**: Filtered projected KV cache layer dictionaries per PP worker rank so workers without specific layers don't encounter `StopIteration` or unmapped page errors.

---

### C. Mamba Memory Safety & Buffer Alignment
10. **Zero-Width Block Table Bypass (`block_table.py` / `mamba_utils.py`)**: Schedulers distribute global block IDs to every worker. Added `row_capacity == 0` guards so PP workers without local layers ignore empty group appends rather than throwing capacity overflow exceptions.
11. **Mamba State Initialization (`mamba_utils.py`)**: Added predicate mask `(offset < end_idx) & is_local & mapping_enabled` during metadata generation to eliminate invalid GPU pointer dereferences (Xid 31).

---

### D. NCCL Communicator & Broadcast Synchronization (The Request Hang Root Cause)
12. **Broadcast Tensor Shape Mismatch in `PPHandler` (`pp_utils.py`)**:
    * *The Problem*: During prefill or standard decoding (`num_draft_tokens == 0`), `SamplerOutput.sampled_token_ids` has shape `(num_reqs, 1)`. When MTP speculative decoding was enabled (`num_speculative_tokens = 3`), PP0 allocated receiving buffer sized to `(num_reqs, 4)`. Sender (PP1) broadcasted 1 int64, while receiver (PP0) waited for 4 int64s. NCCL deadlocked on the communicators, causing PP0 to hang indefinitely in `event.synchronize()`.
    * *The Fix*: Patched `PPHandler.broadcast` to automatically pad `sampled_token_ids` up to `self.max_sample_len` so both sender and receiver always exchange identical tensor dimensions (`[num_reqs, max_sample_len]`).
13. **Eager Sibling Communicator Init (`pp_utils.py`)**: Initialized the `pp_broadcast` subgroup eagerly in `PPHandler.__init__` with a 1-element dummy broadcast before request-time activation P2P, preventing concurrent lazy NCCL init race conditions.
14. **V2 Synthetic Kernel Warmup Bypass (`gpu_worker.py`)**: Bypassed synthetic `warmup_kernels` under PP>1 to prevent stage-local synthetic metadata divergence; small Triton kernels safely JIT on the first real request.

---

## 4. Llama-Swap Configuration & Lifecycle

* **Configuration File**: `/home/jonathan/Documents/server-configs/docker/llama-swap/configs/qwen38-pp2.yaml`
* **Merged Active Config**: `/home/jonathan/Documents/server-configs/docker/llama-swap/run/config.yaml`
* **Compose File**: `/home/jonathan/Documents/projects/qwen38-flash-next-pp2/docker-compose.yml`

```yaml
healthCheckTimeout: 300  # Root-level timeout (5 minutes for 175GB weight loading)

models:
  qwen38-flash-next-pp2:
    name: "qwen38-flash-next-pp2"
    useModelName: "qwen38-flash-next-awq"
    
    cmd: >
      docker compose
      -p qwen38-flash-next-pp2
      -f /qwen38-pp2/docker-compose.yml
      up
      qwen38-flash-next-pp2

    cmdStop: >
      docker compose
      -p qwen38-flash-next-pp2
      -f /qwen38-pp2/docker-compose.yml
      stop
      qwen38-flash-next-pp2

    proxy: "http://qwen38-flash-next-api:8000"
    checkEndpoint: "/health"
    ttl: 600

    aliases:
      - "qwen38-flash-next-awq"
      - "Qwen3.8-Flash-Next-AWQ-INT4"
      - "qwen3.8-pp2"
```

---

## 5. Verification & Benchmark Results

| Test Scenario | Sequence Length | Status | Result / Latency / Throughput |
| :--- | :--- | :--- | :--- |
| **Model Listing** | N/A | **PASS** | Returns `qwen38-flash-next-pp2` & aliases |
| **Cold-Start via Llama-Swap** | Short | **PASS** | Container auto-spun up in ~115s, completion returned |
| **Warm Direct Completion** | 32 tokens | **PASS** | ~0.60s (~75 tokens/sec) |
| **Single Block Boundary Cross** | 1,614 tokens (Block 0 $\rightarrow$ 1) | **PASS** | 20.85s (57.5 tok/s), 0 trailing `!`, 0 repetition |
| **Multi-Block Deep Stress Test** | 3,425 tokens (Block 0 $\rightarrow$ 1 $\rightarrow$ 2) | **PASS** | 51.27s (58.5 tok/s), 0 trailing `!`, 13,927 chars clean story |
| **Thermal & Memory Health** | Extended Decode | **PASS** | GPU 0: 47°C (44W), GPU 1: 40°C (41W), 0B Swap used |

---

## 6. Operational Runbook & Commands

### A. Quick Inference Test via Llama-Swap
```bash
curl -s http://192.168.1.30:9292/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "qwen38-flash-next-awq",
    "messages": [{"role": "user", "content": "Explain pipeline parallelism in one short sentence."}],
    "max_tokens": 32,
    "temperature": 0.7
  }'
```

### B. Long-Sequence Multi-Block Boundary Test
```bash
python3 -c '
import urllib.request, json, time
payload = {
    "model": "qwen38-flash-next-awq",
    "prompt": "hello world " * 200 + "\nWrite a detailed sci-fi story.",
    "max_tokens": 2000,
    "temperature": 0.0
}
req = urllib.request.Request("http://192.168.1.30:8001/v1/completions",
    headers={"Content-Type": "application/json"},
    data=json.dumps(payload).encode("utf-8"))
with urllib.request.urlopen(req, timeout=300) as resp:
    res = json.loads(resp.read().decode("utf-8"))
    print(res["choices"][0]["text"][-200:])
'
```

### C. Checking Container and Hardware Logs
```bash
# Check container logs
ssh jonathan@192.168.1.30 "docker logs qwen38-flash-next-api 2>&1 | tail -n 30"

# Check GPU utilization and temperatures
ssh jonathan@192.168.1.30 "nvidia-smi"
```

### D. Rebuilding Docker Image (If Source Patches Are Modified)
```bash
ssh jonathan@192.168.1.30
cd /home/jonathan/Documents/projects/qwen38-flash-next-pp2
sudo docker build -t qwen38-flash-next:pp2 .
sudo docker compose up -d --force-recreate
```
