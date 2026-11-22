# Experiment reproducibility

This audit reads existing results. It does not authorize training, CUDA inference, benchmark reruns or mesh extraction. Full inventories and checkpoint files remain at their original paths.

## Evidence regeneration without training

Run from the repository root using the existing audit-compatible interpreter. No packages are installed by these commands. CPU checkpoint inspection and plotting are auxiliary work, not a neural CPU fallback.

```powershell
$env:PYTHONUTF8='1'
$env:PYTHONDONTWRITEBYTECODE='1'
.\.tools\pgsr_env\Scripts\python.exe scripts/research/discover.py --hash-artifacts
.\.tools\pgsr_env\Scripts\python.exe scripts/research/analyze.py
.\.tools\pgsr_env\Scripts\python.exe scripts/research/checkpoints.py
.\.tools\pgsr_env\Scripts\python.exe scripts/research/assets.py
.\.tools\pgsr_env\Scripts\python.exe scripts/research/figures.py
```

`discover.py` inventories the entire root without following symlink directories; installed environments/external code are preserved. Selected scientific artifacts are SHA256-hashed; large raw datasets/environment packages are indexed without wholesale hashing. Caches require size/mtime agreement. `analyze.py` deduplicates source copies using exact structured-log content, never merges distinct runs just because their configurations match, and retains recorded inactive zeros. Missing cells remain empty/null. The source file and original line accompany each observation.

`checkpoints.py` uses `weights_only=True`, CPU mapping and mmap for modern Torch archives. A fixed allowlist supports project NumPy RNG and Torch version metadata; it never imports checkpoint-controlled globals or falls back to unrestricted pickle. Hash-identical content is inspected once. State-shape/finite/Adam/RNG checks establish structural readability, not a new CUDA resume test. Legacy LPIPS weights use restricted non-mmap loading. Preserve the two locked LPIPS weight files for metric reproducibility.

## Existing training/inference contracts

Canonical B0 uses `src.gaussian_splatting.runner`; native source is the pinned 2023 Graphdeco revision. B2 uses `src.pgsr.runner` with its separate environment and pinned PGSR plane rasterizer. Never mix these native namespaces or resume B2 from B0/PLY. Existing [B0 instructions](tech/b0_training_usage.md) and [PGSR contracts](tech/pgsr_training_usage.md) describe authorized execution; this audit did not run those commands.

The PGSR research import is structurally complete at iteration21298 but **not directly compatible with the current resume CLI layout**: its root config/identity and training-result files were moved into `configs/` and `logs/` by an earlier import. Do not silently rewrite sealed artifacts. The original flat `phase03_results/recovery/run` remains available, with checkpoint SHA matching its published pointer and original recovery manifest. An explicitly authorized future Kaggle continuation can use that original recovery through the existing `PREVIOUS` handoff. CUDA restore and full 30k completion still need actual execution. The original bundle must not be deleted as a presumed disposable duplicate.

Prepared legacy data is `datasets/tanks_and_temples/Ignatius/prepared/legacy_v1/ignatius-legacy-v1-6a3308c22df11122`. Metadata SHA256: `7636dd218be5cdb1bfa10b76b59a40b2b72cfb6129b0bf2d8fac4283a767913b`; frozen split: `fbd4eb563ba6b5418165ce4d402e04ec123117f56b2fe0c601a35efaad3ba2b4`. World coordinates and K/Tcw remain unchanged; arbitrary monocular scale and transductive pose estimation are disclosed. Legacy data does not unblock the separate video GPU-BA gate.

## Metrics and selection limits

B0 test scores use rounded uint8 PNGs. PGSR smoke has float-render diagnostics; local PNG scores, where present, are the matching quantization protocol. Separate L1/SSIM histories for PGSR were not logged. Exposure changes its RGB branch and has no additive exposure-loss series. Highest sampled training loss improvement cannot identify a best validation checkpoint. Periodic B0 validation uses its recorded subset, not the final test set; validation LPIPS histories are absent.

Timing is reported with scope: a recorded elapsed-segment sum is a lower bound. In particular, B0 low-memory's final `training_seconds_total` describes only its last resumed segment. GPU memory sources distinguish Torch reserved/allocated peaks from sampled device-wide usage. Different GPUs, resolutions, schedules, view subsets and objective weights prevent direct performance/objective rankings. No dense GT accuracy or final predicted mesh metric was recovered.

## Cleanup and verification

Generate `scripts/research/cleanup.py`'s plan only after the evidence and dependency audit. `--apply-disposable` acts solely on individually enumerated, unchanged bytecode/tool-cache files inside the root and outside installed/protected trees. It never recursively deletes experiment folders. Checkpoint/PLY/log/config/Kaggle/dataset/unknown-dependency deletions require explicit approval; see `research/cleanup/approval_required.csv`.

After cleanup, use `scripts/research/verify.py` to verify retained hashes, report links, metrics/registry consistency and entry-point syntax without running models. Keep original reports as historical evidence; the current audit status is [RESEARCH_STATUS.md](RESEARCH_STATUS.md).
