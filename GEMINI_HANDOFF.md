# Qwen 3.8 Flash Next PP=2 / llama-swap Handoff

Last updated: 2026-09-03 22:04 PDT

## Objective

Serve `cyankiwi/Qwen3.8-Flash-Next-AWQ-INT4` through a custom vLLM build on two NVIDIA CMP 170HX GPUs using TP=1 and PP=2, with CPU-offloaded PLE tables, then expose it through llama-swap under the served model name `qwen38-flash-next-awq`.

The model is a hybrid Transformer/Mamba MoE model with MTP speculative heads and N-gram PLE. The relevant upstream Qwen 3.8 work is vLLM PR `#53899`; that code did not support this PP+PLE combination without local patches.

## Access and paths

- VM: Proxmox VM 101, `ollama.homelab`, `192.168.1.30`
- SSH: `ssh root@192.168.1.30`
- Project: `/home/jonathan/Documents/projects/qwen38-flash-next-pp2`
- Main patches: `patch_pp.py`, `patch_p0.py`
- Compose: `/home/jonathan/Documents/projects/qwen38-flash-next-pp2/docker-compose.yml`
- vLLM image: `qwen38-flash-next:pp2`
- Container: `qwen38-flash-next-api`
- Direct API: `http://192.168.1.30:8001`
- llama-swap API: `http://192.168.1.30:9292`
- llama-swap config: `/home/jonathan/Documents/server-configs/docker/llama-swap/configs/qwen38-pp2.yaml`
- Local diagnostic copies: `/home/jonathan/qwen-debug`

Do not print or paste the Hugging Face token from the compose environment. It should be rotated later because it appeared in prior diagnostic output.

## Hardware and memory

- 2x NVIDIA CMP 170HX, about 63.4 GiB usable VRAM each, capped at 150 W
- Driver `610.57.04`; CUDA UMD `13.3`; pinned NCCL/PyTorch NCCL `2.29.7`
- Guest currently reports about 125 GiB RAM after the VM RAM reduction
- Persistent swap:
  - existing `/swapfile`, 2 GiB
  - added `/swapfile-qwen`, 16 GiB
  - `/etc/fstab`: `/swapfile-qwen none swap sw,pri=-3 0 0`

Loading normally fills nearly all 18 GiB swap while leaving about 32–38 GiB `MemAvailable`. This has kept the VM responsive. A full model reload takes roughly 3.5–4 minutes.

## Current state — important

The newest image is:

```text
sha256:15877eb826c8b23f5bdc1c69dd50794d06b8e30e7e6b33b50be21e30065ea756
```

The container is currently running its first load of this image. At 22:04 PDT it was `health=starting`, had finished all PP0 shards, and had about 37 GiB guest memory available. This newest image has **not yet been tested with a completion request**.

The live container restart policy is deliberately `no` during diagnosis, although compose declares `unless-stopped`. Restore `unless-stopped` only after direct and llama-swap completions pass.

The compose health-check `start_period` was increased persistently from 180 seconds to 360 seconds because normal initialization exceeds three minutes.

Check the current run before changing anything:

```bash
ssh root@192.168.1.30 \
  "docker inspect qwen38-flash-next-api --format 'status={{.State.Status}} health={{if .State.Health}}{{.State.Health.Status}}{{else}}none{{end}} restart={{.HostConfig.RestartPolicy.Name}}'; free -h"

ssh root@192.168.1.30 \
  "docker logs -f qwen38-flash-next-api"
```

## Existing PP/PLE compatibility patches

`patch_pp.py` already makes these upstream areas PP-aware:

1. PLE/N-gram embedding only activates on the first PP rank.
2. Removes the explicit PP+PLE rejection in model config and GPU worker validation.
3. Prevents non-owning PP ranks from expecting Hyper-Connection mixer weights.
4. Guards `setup_ple_offload` and MTP hidden-state assumptions by PP rank.
5. Prevents non-first PP ranks from registering PLE layers or launching PLE IPC receivers.
6. Projects KV-cache layer dictionaries to the physical layers owned by each PP rank.
7. Handles empty projected KV groups in cache sizing and `UniformTypeKVCacheSpecs`.
8. Handles `layer_spec is None` and empty backend groups during worker cache allocation/warmup.

`patch_p0.py` also contains earlier Mamba indexing, block-table pooling, speculative-state hardening, PLE-layer, and PP warmup work. The build logs show several connector patterns no longer match the newer connector implementation:

```text
WARNING connector init pattern not found
found alternative connector queue
patched connector exception _staged.set
WARNING connector copy pattern not found
WARNING connector put pattern not found
```

The current upstream connector already uses a queue of `_PendingPleOffloadRequest` objects and CUDA D2H completion events. The unmatched legacy handshake patches are probably obsolete. One bad residual change remains: its exception handler calls `self._staged.set()` even though `_staged` is not created in this connector revision. That only executes if the request thread throws, but it should be cleaned up or adapted before finalizing.

## Failure chain and fixes developed in this session

### 1. Original CUDA illegal memory access — fixed

Original warmup failure:

```text
mamba_utils.py::_populate_metadata
self.state_base_addrs[idx] = _reinterpret_u64_as_i64(...)
torch.AcceleratorError: CUDA error: an illegal memory access was encountered
```

With `CUDA_LAUNCH_BLOCKING=1`, the real fault was identified earlier in `block_table.py::_compute_slot_mappings_kernel` on PP1. Stage-local KV projection preserves global cache group indices. Groups absent from a rank are represented by zero-width tensors and `slot_mapping_enabled=False`, but Triton still dereferenced their null block-table pointer because the load mask was only `is_local`; `tl.where()` did not predicate the load.

Persistent fix in `patch_p0.py`:

```python
mask=(offset < end_idx) & is_local & mapping_enabled
```

This eliminated the illegal address and allowed startup to reach healthy.

### 2. First real request: zero-width table CPU write — fixed

After startup, the first direct completion failed cleanly with:

```text
RuntimeError: Block table write for request 7, group 3 exceeds row capacity (4 > 0)
```

The scheduler sends global block IDs to every PP worker. PP1 group 3 had no local layers and therefore row capacity zero. The persistent append guard now does:

```python
row_capacity = self.block_tables[i].gpu.shape[1]
if row_capacity == 0:
    self.num_blocks.np[i, req_index] = 0
    continue
```

This is intentionally limited to zero-capacity projected groups; real local groups retain the overflow check.

### 3. Synthetic V2 warmup deadlock — bypassed

vLLM V2's synthetic `warmup_kernels()` built inconsistent stage-local request metadata and deadlocked the sampled-token PP path. The current patch skips only that optional synthetic warmup when `pipeline_parallel_size > 1`; model compilation and CUDA graph capture still run, and the remaining small Triton kernels JIT on the first real request.

Expected log:

```text
Skipping V2 synthetic kernel warmup under pipeline parallelism; kernels will JIT on the first request.
```

### 4. Real-request sampled-token PP deadlock — latest work, not yet validated

After the zero-width append fix, a real request advanced through both model stages but hung after sampling. `py-spy` showed:

- PP0 main thread: Triton `_init_handles` / `cuModuleLoadData` while launching `_post_update_num_computed_tokens_kernel` from `sample_tokens()`.
- PP1 main thread: already waiting in `irecv_tensor_dict()` for the next activation P2P.
- GPU0: 100% utilization; GPU1: 0%.
- PLE request thread and CPU PLE worker: idle, proving the PLE result had completed and was not the blocker.

The sampled-token side collective was rank-dependent. First, `patch_p0.py` was changed so both PP ranks always participate in the two small broadcasts. If the receiver does not need the result, it uses an all-false mask and later discards it. This preserves non-final-prefill state semantics while making collective existence rank-invariant.

That alone still hung. An eager one-element broadcast was then added in `PPHandler.__init__` to initialize the sibling sampled-token NCCL communicator before activation P2P initializes its separate communicator. That also did not eliminate the request-time hang.

The **newest, currently loading image** adds the remaining synchronization:

- receiver calls `event.synchronize()` after the two sampled-token broadcasts;
- sender calls `self.broadcast_stream.synchronize()` after them;
- neither rank advances to the next activation P2P while sampled-token NCCL work is outstanding.

This costs a small amount of PP overlap but should remove the cross-stream/cross-rank dependency cycle. It is not yet validated because the user requested this handoff during the reload.

## Exact next steps

1. Let the current container finish. Do not recreate it merely because health is still `starting`.

2. Once healthy, test the direct endpoint. The first request may take longer due to deferred JIT:

```bash
curl -sS --max-time 180 http://192.168.1.30:8001/v1/models

curl -sS --max-time 180 http://192.168.1.30:8001/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen38-flash-next-awq","messages":[{"role":"user","content":"Reply with exactly: OK"}],"max_tokens":16,"temperature":0}'
```

3. Watch the logs and GPU state during the request:

```bash
ssh root@192.168.1.30 "docker logs -f qwen38-flash-next-api"

ssh root@192.168.1.30 \
  "nvidia-smi --query-gpu=index,utilization.gpu,memory.used,power.draw --format=csv,noheader"
```

4. If it hangs again, install `py-spy` only in the disposable running container and capture both workers:

```bash
docker exec qwen38-flash-next-api pip install -q py-spy
docker exec qwen38-flash-next-api ps -eo pid,comm,args | grep Worker_PP
docker exec qwen38-flash-next-api py-spy dump --pid <PP0_PID> --native
docker exec qwen38-flash-next-api py-spy dump --pid <PP1_PID> --native
```

If PP0 is still in `post_update_num_computed_tokens` while PP1 is in the next P2P receive even with unconditional synchronization, add temporary logging immediately before/after each `torch.distributed.broadcast` in `PPHandler.receive()` and `PPHandler.broadcast()`, including tensor shape, dtype, global rank, and sequence number. A tensor-count or shape mismatch is then more likely than communicator ordering. Do not assume the 100% GPU indication means a model compute kernel; earlier it was an outstanding NCCL operation exposed by CUDA module-load synchronization.

5. If direct generation succeeds, immediately send a second direct request. The second should avoid first-request JIT and verifies decode/state reuse.

6. Test llama-swap:

```bash
curl -sS --max-time 180 http://192.168.1.30:9292/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen38-flash-next-awq","messages":[{"role":"user","content":"Reply with exactly: SWAP_OK"}],"max_tokens":16,"temperature":0}'
```

The intended llama-swap proxy config is:

```yaml
models:
  - name: "qwen38-flash-next-awq"
    type: "proxy"
    proxy_url: "http://qwen38-flash-next-api:8000"
    health_check: "http://qwen38-flash-next-api:8000/health"
```

Both containers must be attached to external Docker network `ai-net`.

7. After direct and llama-swap tests pass, restore automatic restart:

```bash
ssh root@192.168.1.30 \
  "docker update --restart=unless-stopped qwen38-flash-next-api"
```

8. Then run several sequential requests and a small two-request concurrency test. Pay special attention to PLE queue reuse, Mamba state reuse, and the block-table pool (`VLLM_BT_POOL=5`).

## Rebuild and restart procedure

Compose has an `image:` entry but no `build:` entry, so `docker compose build` does nothing. Build explicitly:

```bash
ssh root@192.168.1.30
cd /home/jonathan/Documents/projects/qwen38-flash-next-pp2
docker build -t qwen38-flash-next:pp2 .
docker compose up -d --force-recreate
docker update --restart=no qwen38-flash-next-api
```

Every restart reloads approximately 175 GiB across 38 shards into two GPUs and the CPU PLE worker. Avoid repeated speculative rebuilds; collect live stacks/logging first.

## Safety and recovery notes

- The earlier CUDA illegal access generated NVIDIA Xid 31. A VM reset while the passed-through GPUs were faulted hard-locked the Proxmox node and required a physical reset. Do not reset VM 101 as a routine recovery method.
- Prefer `docker stop -t 10 qwen38-flash-next-api`; if a GPU/NCCL hang prevents exit, use `docker kill qwen38-flash-next-api` and verify `nvidia-smi` before restarting.
- Do not hard-reset the host merely because fans are at 100%. First check temperature, process state, SSH responsiveness, and `nvidia-smi`.
- The allocator sometimes logs a 20 MiB expandable-segment mapping OOM near the end of weight loading, while startup subsequently succeeds. Treat it as fatal only if followed by an exception or process exit.
- Keep the 16 GiB swap unless the guest RAM allocation is raised substantially. Full swap use during load is expected in the present configuration.

## Success criteria

The deployment is complete only when all of these pass:

- container remains `healthy` after initialization;
- `/v1/models` works directly;
- first and second direct chat completions return without engine death or hang;
- llama-swap completion returns through port 9292;
- two sequential and two modest concurrent requests succeed;
- no Xid/illegal address appears;
- live restart policy is restored to `unless-stopped`.
