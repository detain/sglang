# sglang_missing.md — Improvement & Gap Findings

**Deployment:** `Qwen/Qwen3.8-Flash-Next-FP8` on 4× RTX PRO 6000 Blackwell **Max-Q Workstation Edition** (SM120, 188 SM, 97.9 GiB each, PCIe-only, no NVLink, no P2P), branch `stack-36644-core` (tip c018ceb4cb), launched by `/root/run_sglang_qwenflashnext6.sh` (TP4/EP4, FP8-KV, page 128, YaRN-1M, NEXTN 3/1/4 + linear ReplaySSM, PLE pinned offload, HiCache L2+L3/file, `extra_buffer`, mem 0.88).

**Compiled:** 2026-09-17 by four research agents: (1) full discussion mining of PRs #36787/#37275/#38144/#38209/#36644; (2) PR/issue mining of #39807/#39862/#39809/#39893/#39830/#39721-24/#37500/#36497; (3) reference extraction (HF bf16 + FP8 cards, SGLang cookbook, vLLM recipe, Qwen blog, 28-page tech report PDF); (4) static codebase analysis of this tree + environment audit. Cross-contradictions resolved against code where noted. This is findings only — nothing here has been executed on the GPUs.

---

## 0. TL;DR — top actions, roughly by expected payoff

1. **The GDN-state/KV pool split is badly inverted for long context.** `--mamba-full-memory-ratio` is unset → default **0.9**, parking ~25 GiB/rank in a state pool that needs ~5 GiB, and starving KV. For 8K avg ctx use ≈0.64, 16K ≈0.33, 32K ≈0.16; or pin `--max-mamba-cache-size = 5 × target_concurrency`. **Biggest pure-throughput lever.** (Appendix A has the math.)
2. **Fused-MoE runs on untuned default configs on this box.** `#36787`'s tuned FP8 configs shipped as JSON keyed on the exact device name `NVIDIA_RTX_PRO_6000_Blackwell_Server_Edition`; this GPU reports `...Max-Q Workstation Edition` → no match. Copy/retarget the three Server-Edition fp8_w8a8 block-128 files (E=128/256/64, N=640 — exactly your per-rank shapes) or set `SGLANG_MOE_CONFIG_DIR`. Boot log should currently show "Using default MoE kernel config" — check.
3. **DeepGEMM is installed and has a purpose-built SM120 MoE path, but `auto` refuses to select it with `a2a=none`.** Try `--moe-runner-backend deep_gemm` (deep_gemm_sm120.py exists for exactly StandardDispatcher's contiguous layout) and separately `--fp8-gemm-backend deep_gemm` (SM120 auto deliberately short-circuits to CUTLASS). A/B both.
4. **`SGLANG_QSA_DECODE_BACKEND=trtllm` is reachable and was numerically validated on SM120** (not SM121), and it avoids the Triton fp8-widen path. A/B decode latency against the current `auto`→triton.
5. **HiCache on this exact surface (FP8-KV + QSA + GDN + PLE side-states, TP4/EP4, file L3) has zero GPU validation anywhere — upstream or here.** #39893's 15/15 table is B200 TP1 NVFP4; #39862's is 1× RTX PRO 6000 TP1. Run `test_qsa_hicache.py` + a host-restore probe when the GPUs free before trusting production.
6. **Chunked-prefill-with-cached-prefix re-gathers the whole sequence's KV per layer in Python** (`qwen_sparse_attn_backend.py:1441-1552` `index_select`+`cat`): O(ctx) copies per chunk per full-attn layer. The most suspicious perf item on a 1M-context box. Profile before tuning anything smaller.
7. **Static YaRN factor 4.0 degrades short-context quality by design** (HF card + vLLM recipe: use factor 2.0 if typical ctx is 524K). All traffic on your server pays the short-text tax. Consider a second instance or dynamic YaRN if short quality matters.
8. **The AOT `topk.cu` (kernels/aot/csrc/elementwise/topk.cu) still has the same threshold-bin overflow bug #38144 fixed only in the JIT path** — silent wrong top-k at ≥~30K contexts. Unfixed upstream and here.
9. **Every PR in the stack is unreviewed, never CI-run (no `run-ci` label, pr-gate fails, GPU jobs skipped), and #38209/#36644 still target the stale `qwen4-main-squashed` base.** Your box is effectively the only SM120 TP4/EP4 validation that exists anywhere — your own e2e results are the missing evidence.

---

## 1. Where this deployment sits relative to every published recipe

Nothing in the cookbook, vLLM recipe, HF cards, or blog covers your exact shape. Specifically:

- **SGLang cookbook verified cells for RTX PRO 6000 (SM120) are TP=1 NVFP4 only.** There is *no* FP8-checkpoint cell for SM120 at any TP size; datacenter FP8 cells are TP4/EP4 on H200/B200+ with NVLink.
- **No source anywhere sets `--kv-cache-dtype fp8_e4m3` for this model.** (SGLang cells leave it unset; AMD uses auto.) Your FP8-KV + QSA-compressed-K path is off-matrix; #36644's own quality data: GSM8K 95.97 vs 96.27 BF16 (p=0.45, not significant), and on real checkpoints **no KV scales ship** → unit descale + startup warning (non-unit scales only ever tested synthetically).
- **`--page-size 128`** — verified NVIDIA cells leave the default; Spark-like cells use 64. Legal here (pool only demands a multiple of compress-ratio 4), but unvalidated territory; #39893's sidecar is page-aligned at your page size — covered by its layout-combo tests (4 combos), so fine in principle, unproven on SM120.
- **`--ple-offload-embedding` + 1M YaRN + HiCache(file) + NEXTN+ReplaySSM** appears in *no* verified cell of any framework.
- **Cookbook SM120 note: pinned PLE requires ≥64 GiB free host RAM + `--ulimit memlock=-1`** — the table is 47.68 GiB FP8 split 4 ways pinned. Confirm the host has headroom (the #36787 GB10 story: 78 GiB weights + 47.7 GiB table > 121 GiB unified → OOM at weight load).
- **`--linear-attn-prefill-backend triton --linear-attn-decode-backend flashinfer`**: every verified CUDA cell uses flashinfer/flashinfer; SM120 auto-resolves everything to triton; the cookbook's own SM120 note says pinning flashinfer decode measured **identical TPOT (±0.3 ms)** on SM120 — your decode pin is validated-neutral (see B8 for the prefill-side experiment).
- **`--max-running-requests` is pinned in every verified cell** (16–96); you leave it unset. With the ratio fix (#0.1) the state pool caps concurrency anyway, but pin it deliberately and read the *effective* value from the boot log (the cookbook warns `/get_server_info` lies under MTP).
- **mem-fraction**: datacenter FP8 cells use 0.85; SM120 TP1 cells use 0.93–0.96 *after* loader temporaries free. Your 0.88 is uncharted middle; cookbook documents OOM modes in **GDN short-conv during prefill** when <4 GiB free post-graph-capture — verify `avail mem` boot lines before pushing higher.
- **#36787 caveat: pure TP4-without-EP is *impossible* on FP8** (per-rank N=160 not divisible by the 128 quant block) — so the "does EP even help on PCIe" A/B is moot for this checkpoint; EP4 with StandardDispatcher is the only shape, and its MoE combine is a plain per-layer all-reduce (see C2).

---

## 2. Confirmed performance gaps on this hardware

### A-class (found in code, active today)

| # | Finding | Evidence | Fix |
|---|---------|----------|-----|
| A2 | Untuned fused-MoE configs (device-name miss) | `moe_runner/triton_utils/fused_moe_triton_config.py:36-51` keys on exact device name; tuned JSONs exist only for `Server_Edition` | Retarget JSONs / `SGLANG_MOE_CONFIG_DIR`; ultimately `benchmark/kernels/fused_moe_triton` |
| A3 | DeepGEMM MoE path reachable but unselected | `quantization/fp8.py:1233-1254`: auto→deep_gemm only with a2a ∈ {deepep,mooncake,nixl}; `moe_runner/deep_gemm_sm120.py` exists; package installed w/ SM120 entry | `--moe-runner-backend deep_gemm` (A/B) |
| A8 | Dense FP8 GEMM short-circuits deep_gemm on SM120 | `fp8_utils.py:856-888`: auto forces cutlass | `--fp8-gemm-backend deep_gemm` (A/B) |
| A6 | trtllm-gen QSA sparse decode (validated SM100 **and SM120**) never auto-picked | `qwen_sparse_attn_backend.py:55-140`; flashinfer 0.6.18 installed; internal page-64 scratch, independent of your `--page-size 128` | `SGLANG_QSA_DECODE_BACKEND=trtllm` (A/B; watch accept-rate/needle probes — see C-risk below re sm121 corruption class) |
| A5 | FP8 KV on Triton QSA = **bandwidth only** (no fp8 MMA; widen-on-load per #36644) + `_store_kv` **clones K/V every write** for non-unit block scales | `qsa/sparse_attn.py:291,681,826`; `qwen_sparse_attn_backend.py:285-299` | trtllm decode (A6) avoids the widen path; clone is a kernel-level TODO upstream |
| A4 | Custom allreduce structurally unavailable (PCIe, no P2P) → every TP/MoE combine is NCCL | `custom_all_reduce_utils.py:442-515` hard-gates on NVLink+P2P | None via flags; **do not** force `SGLANG_SKIP_P2P_CHECK` (silent-corruption risk class). Profile NCCL share of decode step first (C3 below) |
| A1 | FlashInfer GDN cannot do spec **verify** on SM120 (`SUPPORTS_TARGET_VERIFY = sm_major in (9,10)`) → verify forced to Triton | `linear/gdn_flashinfer.py:213`; `attention_hook.py:240-340` | Already mitigated by `--enable-linear-replayssm-spec` (chain-only topk≤1 — your shape is legal; falls back to slow per-step recurrent verify for topk>1) |
| B5 | Default `mamba-full-memory-ratio` 0.9 → state pool ~25 GiB/rank vs ~5 needed at 8K ctx; costs ~14%/~40% of achievable concurrency at 8K/32K avg | `arg_groups/fields/schedule.py:203-210` + Appendix A math | Set ratio per workload table or `--max-mamba-cache-size` pin |
| B6 | `extra_buffer` = 5 mamba slots/req; `extra_buffer_lazy` = 4 | `kv_cache_configurator.py:163-172`; cookbook SM120 cells use `_lazy`; `SGLANG_OPT_MAMBA_SKIP_DECODE_LOCK=1` same lever | Switch strategy (overlap scheduler is on — required for lazy) |

### B-class (untested-but-allowed backends worth A/B on this box)
- **B8:** `--linear-attn-prefill-backend flashinfer` on SM120 (allowed by code, never auto-picked; fp32-convert per prefill exists; 36 GDN layers × 8K chunks makes this a credible prefill swing).
- **B4/B7/B9:** trtllm QSA decode; HiCache host sizing (below); mem-fraction 0.90-0.92 after headroom check.

### C-class (profile before fixing)
- **C1 (top suspect):** prefix-hit chunked prefill = full per-layer Python re-gather (`qwen_sparse_attn_backend.py:1441-1552`): with 1M-resident caches and 8K chunks this is O(ctx) `index_select`+`cat` per chunk per QSA layer. Fix direction: a paged chunked-prefill kernel (none exists on this path). #37275 itself notes the chunk wrapper still "derives its launch bound with a device read" (per-forward host sync) — related unfixed item.
- **C3:** NCCL tail on PCIe TP4: 48 layers × 2+ all-reduces per token, small batches under topk-1 spec → un-amortized latency. Measure the NCCL fraction of a decode step (torch profiler) before micro-tuning kernels; it may bound returns from everything above.
- **C5/C6:** ReplaySSM ring accounting under HiCache load-back has no GPU test anywhere — run one staging boot with `SGLANG_VALIDATE_MAMBA_REPLAY_STATE_INDICES=1`; check spec-acceptance metrics (published: accept 3.3/4 on GSM8K SM120 NVFP4, 2.9 long ShareGPT, model report ideal 4.07; vLLM measured MTP *hurting* throughput on H100 — don't assume).

---

## 3. Correctness / stability risks (this stack and its neighbors)

1. **#39830-class corruption is fixed *on paper* only for your config.** Both PRs' validation is TP1 (B200 / single RTX PRO 6000, NVFP4). Issue #39830 stays open, zero maintainer replies; the reporter's proposed **restore-position assertion** ("a restored Mamba checkpoint taken at the same token position as the last restored KV page — today nothing checks it") is still unimplemented upstream *and locally*. Your two residual fixes (e39d62115b bad-page degradation, 54b41dc9e7 compressed-K retraction backup) close the known residuals but only via CPU tests.
2. **`fast_topk` overflow:** #38144's own author states the **AOT sibling kernel (`kernels/aot/csrc/elementwise/topk.cu`) shares the bug and is handled "separately" — no such PR exists** as of 2026-09-17. If anything routes through the AOT path on ≥30K contexts you get wrong-but-plausible top-k with no error. Audit which callers resolve to AOT on SM120.
3. **trtllm-gen vs consumer Blackwell decode:** history says the sm_121 (GB10) trtllm/XQA sparse-decode route **silently corrupts long-context decode** (hashd1ve's retraction on #36497: token-0 runs at 120k+). #36787's l.91-93 comment claims SM120 *is* validated for trtllm-gen sparse decode — trust it only with needle/accept-rate probes when A/B-ing B4; keep SM121 excluded.
4. **Upstream still-open defect families adjacent to your config** (from #36497 comments): mixed-chunk + QSA device-side assert (#38180 = fail-fast guard only; deeper unaligned-write-plan fix unimplemented; #39342 still open); invalid-probability rank crashes #37052 and QSA+NEXTN **graph silent corruption** #37111 (GB10 TP2, production-warning withdrawn from the ecosystem, not resolved); PD+mooncake `sequence_lengths.tolist()` IMA — only observed with `SGLANG_ENABLE_HEALTH_ENDPOINT_GENERATION` on (you already set it false); `--load-format dummy` OOMs under PLE offload; NVFP4 checkpoints need `--speculative-draft-model-quantization unquant`.
5. **#38209 removes per-layer `positions.max().item()` capacity checks** in favor of a ModelRunner reservation invariant — no SM120 validation in the PR at all (GB200-only, single-round numbers, dev env macOS without CUDA). Your combined evidence is detain's #37275 comment ("runs pretty awesome"), which predates the 09-16 rebase.
6. **Non-determinism:** the engine is not bit-deterministic at temp 0 even on device-cache hits (#39830 measurement: cold reproduced 3-4/10). Don't judge HiCache fixes by byte-equality; use the reporter's marker/needle protocol.
7. **Quality flags to re-check after any change:** #36787's unresolved RealWorldQA gap (86.4-87.2 vs published 88.5, speculation excluded, "being investigated"); #37275 GSM8K delta within noise; #36644 GSM8K −0.3 (p=0.45). None is *caused* by your config, all are open upstream questions to attach your own numbers to.

---

## 4. What the model/tech-report describes that SGLang lacks (or doesn't use here)

1. **FlashQLA** — Qwen's own TileLang GDN kernels, **2–3× forward vs Triton** (github.com/QwenLM/FlashQLA). Not an SGLang backend today. Your 36/48 GDN layers make this the single biggest *missing-kernel* item; also note sm100 QSA scoring still uses TileLang while sm120 uses the Triton port.
2. **Gated-Residual (GR) FP8 state storage** — report describes FP8-storage of the 4-branch residual as the adopted design point (halves the biggest per-token traffic); SGLang carries hidden states in bf16. No flag, no path.
3. **FlashInfer paged index-K scorer** (flashinfer-ai/flashinfer#4899) — #38209's author names it as the thing that *removes the need for compressed-key packing* entirely; deliberately not built on.
4. **Paged chunked-prefill attention** — see C1; and the chunk wrapper's device-read launch bound (#37275 "not covered").
5. **Cross-step QSA index sharing (GLM-5 style) is present** — `QSAMTPSharedSparseIndices` (`qwen_sparse_attn_backend.py:149-205`) — good news; spec draft reuses top-k as the report describes.
6. **`--enable-prefill-cp` (context parallel)** exists upstream only as the unmerged #39721-24 series and is *correctly excluded*: rejects EP>1, speculation, radix cache, and chunked-prefill simultaneously (your whole config), validation 4×B200-only, zero discussion — and its one transferable warning: **FlashInfer autotune's decode-shaped dummy forward "poisons the CUDA context" under CP topologies**.
7. **No SM120 CI runner exists at all** — every sm120-gated test in the stack skips on registered runners. Upstream cannot catch regressions in your paths; treat local per-file test runs as the real CI.
8. **SM120/SM121 QSA decode gating upstream is *still* open as #36556**; #36787 (in your stack) is its SM120 superset — unmerged, so main still can't serve this model on your cards as of 2026-09-17.
9. **Metrics/diagnostics #39830 asked for are unmerged**: #39436 (per-pool eviction metric + host-coverage boot line). Useful for exactly the pool-rebalance work in §0.1.
10. **Multimodal:** the model is a VLM (27-layer ViT, mrope). Your launch passes none of the video/image flags; if you serve vision, note the card's long-video `longest_edge` raise and per-request `mm_processor_kwargs.fps` being vLLM-only today. `enable_thinking=False` works per HF card but the cookbook claims "thinking cannot be turned off" — verify against the shipped template before relying on non-thinking mode.

---

## 5. Config hygiene (cheap, immediate)

- **HiCache:** `--hicache-size 10` = **10 GiB per rank**, split across KV/Mamba and now the QSA-indexer sidecar slice — small once B5 rebalances. Set `SGLANG_HICACHE_FILE_BACKEND_STORAGE_DIR` (default `/tmp/hicache` — possibly tmpfs, i.e. silently eating RAM) and `SGLANG_HICACHE_FILE_BACKEND_MAX_SIZE` (default **unbounded** disk growth). Consider `--hicache-write-policy write_through_selective` and tuning `prefetch_timeout_base/per_ki_token/threshold` via `--hicache-storage-backend-extra-config` for long-context prefetch.
- **YaRN:** keep factor 4.0 only if you actually serve >262K; HF card explicitly recommends 2.0 (524K) otherwise — short-text quality tax is real and permanent for all traffic.
- **PLE rebase obsolescence (post-09-16 rebase, per Dev-Jahn on #37275):** main now does sm12x fp32 prefill-state conversion itself and FlashInfer 0.6.18 ships the pooled bf16 GDN decode kernel → local commits **b5169fd30e and 5bf8ba3576 are no longer needed**; the exec-bag graph-config prewarm fix (5521147e04) was folded upstream. Drop on next rebase (verify first — your stack tip predates some of these).
- **`SGLANG_QSA_PREFILL_GEOMETRY=tuned`:** was *required* on Workstation when #37275 gated by device name; post-rebase the gate is **SM-count ∈ {188}** (`sparse_attn.py:71,196-207`) and your card has 188 SM → `auto` already picks tuned. Redundant but harmless; keep until you re-test without it. Open upstream ask: **nobody has posted the `bench_qsa_sparse_prefill.py` Workstation table yet** (Dev-Jahn requested 09-16) — you're the only box that can.
- Sampling per card: thinking `temp=1.0, top_p=0.95, top_k=20`; non-thinking `temp=0.7, top_p=0.8, presence_penalty=1.5`; presence_penalty 0–2 is the anti-repetition knob.
- `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` is in every SM120 cookbook cell and absent from your script.
- Run tests **per file** (CI parity — your CAVEATS already say this).

## 6. Community data you owe/should post (it is also free validation)
1. `bench_qsa_sparse_prefill.py` table for Max-Q Workstation → #37275 (open request, unblocks removing the env override).
2. The **TP4/EP4 HiCache+FP8-KV+file-L3 e2e** result on SM120 → #39862/#39893/#39830 (your 09-17 comment on #39862 is the *only* TP4 data point anywhere; nobody has validated this shape upstream).
3. `run-ci` label is missing on all 9 stack PRs → every head has never executed GPU CI. Ping for labels (or `@tag-and-rerun-ci`) after posting SM120 evidence; #38209/#36644 additionally need retargeting off the frozen `qwen4-main-squashed` base.

---

## Appendix A — pool math (TP4/EP4, per GPU)

- Card budget @0.88: 84.1 GiB (of 97,887 MiB) · weights/rank ≈ 31.3 GiB ((172.76 − 47.68 PLE)/4; experts 28.7 EP-sharded) · pools ≈ 52.8 GiB
- `state_bytes_per_slot` ≈ 14.7 MB (12×48×128² bf16 ×36 layers + conv + PLE side + rope/ring) · `kv_bytes_per_token` ≈ 13.8 KiB (13 attn layers incl. MTP: 2KV×256 fp8 = 13.3 KiB + **bf16 compressed-K** 0.8 KiB — compressed K is *not* fp8) · token-equivalent ≈ 1,050
- S=5 (`extra_buffer`+overlap) → r\* = 5×1050/L: **L=8K→0.64, 16K→0.33, 32K→0.16, 2K→2.6 (clamp ≤ pool max)**. Current r=0.9 ⇒ ~1,730 slots (clamp 346 req) but only ~2.0M KV tokens.
- Sanity anchor: one 1M-token request alone reserves ~13 GiB KV — KV (not state) must be the governor for your advertised context. Bias r up if concurrency is spiky (state-pool misses are fail-loud re-prefill; KV starvation is silent churn). Validate against boot-log lines `Mamba Cache is allocated…` / `KV Cache is allocated. #tokens:` after any change, per the compute-mamba-ratio skill.

## Appendix B — unset SM120-relevant knobs (from `environ.py` + code audit)
`SGLANG_QSA_DECODE_BACKEND` (B4) · `SGLANG_MOE_CONFIG_DIR` (A2) · `SGLANG_ENABLE_JIT_DEEPGEMM` (True — prerequisite B1/B3) · `SGLANG_OPT_MAMBA_SKIP_DECODE_LOCK` (B6 alt) · `SGLANG_VALIDATE_MAMBA_REPLAY_STATE_INDICES` (C5) · `SGLANG_HICACHE_FILE_BACKEND_STORAGE_DIR/_MAX_SIZE/_EVICTION_RATIO/_MIN_FREE_SPACE` (§5) · `SGLANG_HICACHE_HOST_REGISTER_CHUNK_GB` · `SGLANG_HICACHE_DECODE_OFFLOAD_STRIDE` · `SGLANG_QWEN4_PLE_FILE_*` (dead on this HW — file PLE backend needs the GB10 C2C pageable-memory attribute, `qwen4_exp_ple_table.py:321-343`). Already effectively on: PLE fusion, GDN fused proj-conv, NUMA interleave, MQA triton auto.

## Appendix C — suggested experimental matrix (each row = one boot, GPUs currently busy — schedule when free)
| Run | Change vs current | Watch |
|---|---|---|
| E0 | baseline (today) | tok/s c1/c16/c64, boot `avail mem`, "default MoE kernel config" line |
| E1 | `--mamba-full-memory-ratio 0.33` (+ pin max-running-requests) | KV #tokens, concurrency clamp, TTFT/TBT |
| E2 | E1 + Max-Q MoE JSONs + deep_gemm runner | MoE kernel time in profiler, GSM8K spot |
| E3 | E2 + `SGLANG_QSA_DECODE_BACKEND=trtllm` | decode TPOT, needle@120k, token-0 watch |
| E4 | E3 + `extra_buffer_lazy` + hicache dir/size hygiene | slots/req, L3 disk use |
| E5 | E4 + HiCache e2e probe (host-restore markers after eviction, incl. restart = L3) | correctness table à la #39830 protocol |
| E6 | flashinfer GDN prefill | prefill tok/s on 8K/64K chunks |

*Sources: gh API pulls of all listed PRs/issues (bodies, all issue+review comments, CI check-runs) on 2026-09-17; static audit of this tree (file:line refs inline); tech_report.pdf full text; HF/recipes.vllm/cookbook cards as fetched.*
