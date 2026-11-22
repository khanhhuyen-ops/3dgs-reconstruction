# Project instructions

## Objective and language
Build a reproducible monocular-video reconstruction pipeline with Gaussian rendering and surface extraction. Prioritize verified geometry, provenance and fair comparisons, then memory/runtime and maintainability.
All conversation, progress updates, code comments/docstrings, commit messages, developer documentation and new agent instructions must be in English. Only formal academic research reports are written in professional Vietnamese. Preserve historical documents in their original language.

## Start here; avoid unnecessary context
Read this file, then [RESEARCH_STATUS](docs/RESEARCH_STATUS.md) and only the task-specific specification/skill. Use the indexed [project map](docs/PROJECT_STRUCTURE.md) and [reproducibility guide](docs/EXPERIMENT_REPRODUCIBILITY.md).
Do not read raw datasets, PDFs, binary checkpoints, entire logs, environments or external repositories unless the task needs them. Use `rg` and cached inventories; never print binary files or large loss histories. Plans and installed libraries are not proof of execution.
Current audit does not authorize training, inference, new benchmarks, meshing or resuming jobs. Obtain explicit training authorization before any such run. The measured state lives in `research/metadata/experiment_registry.json`, not this instruction file.

## Data and research integrity
- Video is the default source. Its SfM remains blocked unless a real GPU BA backend is verified. Keep `BLOCKED_GPU_BA`; never silently use a CPU mapper or replace SfM with VGGT/SLAM/custom mapping to bypass the gate.
- Legacy `raw/image_set/` is a separately authorized experiment with historical CPU BA. Do not merge it with video or call it GPU-strict video reconstruction.
- Never modify, overwrite or delete `datasets/**/raw/`. Preserve original COLMAP databases, reconstructions, poses, PLYs, logs, SfM source/scripts/tests/configs and historical reports/skills. Downstream transformations need derived copies plus provenance and validation.
- GT is evaluation-only: no training, pose optimization, frame selection, confidence tuning or implicit alignment feedback. Preserve prepared image IDs, K/Tcw, world normalization and split. Resize/crop must update K and record the image transform.
- Declare camera W2C/C2W conventions, axial/ray depth, normal frame, monocular scale ambiguity and evaluation alignment. Video manifests retain true timestamps, source hashes, resize/crop and selection reasons.
- Baselines: B0 Original3DGS, B1 2DGS, B2 PGSR, optional B3 GOF, proposed geometry-aware PGSR. Ours remains a hypothesis until evaluated; sparse anchors/confidence/edge supervision are not automatically novel.
- Compare identical frames/resolution/split/poses/masks/color space/metric definitions except the declared experimental variable. No test exposure fitting, training-view scores labeled as test, fake metrics, checkpoint interchange or mocked benchmark success. B0 has no native mesh; B0+TSDF is a declared extension.
- Pin external source revisions and pretrained hashes; keep third-party code outside `src`. Cite each paper's own section/page/equation when changing algorithms. Separate paper results, proposals, synthetic checks, smoke tests and measured full runs.

## Runtime and implementation
Windows/PowerShell; CUDA target `cuda:0`, RTX 3050 Laptop about 6 GiB locally. Verify actual tools, versions/compiler and CUDA operations. `nvidia-smi` alone does not verify backend execution.
Deep-learning and GPU-required operations use strict CUDA with no CPU fallback. CPU is allowed for orchestration, data I/O, plotting and safe checkpoint inspection. OOM preserves state; any resolution/batch/sample/count reduction needs a declared profile and logged changes. Do not load several large models together.
Use repository-relative `pathlib.Path`, versioned configs, deterministic seeds, logging, type hints, checked subprocess exit codes, atomic output publication and separate method environments. Preserve native optimizer/ADC order and full optimizer/ADC/exposure/RNG/sampler state. PLY is inference state, not a training-resume checkpoint.
Inspect existing changes/callers before editing; preserve user work. Separate preprocessing, geometry, training, meshing, evaluation and visualization; `src/pipeline.py` only orchestrates implemented stages. Do not create empty future-stage trees. CLI options should include config/stage/dry-run/resume where applicable.
Run focused tests before expensive work; distinguish tests executed from historical/skipped evidence. Actual CUDA subset/VRAM checks precede authorized large runs. Report changed files, commands, validation, backend, blockers and remaining work. Success requires nonempty parseable artifacts and relevant quality checks.

## Locations and task routing
- `src/data`, `configs/data`: prepared COLMAP data; `.agents/skills/colmap-data/SKILL.md`.
- `src/gaussian_splatting`, `.tools/b0_env`: B0; `src/pgsr`, `.tools/pgsr_env`: B2. Never mix their native flat namespaces in one process. Read `.agents/skills/gaussian-training/SKILL.md` and geometry skill for losses.
- `outputs/tanks_and_temples/Ignatius/research/`: canonical run folders; checkpoints inside each run. `phase03_results/` is an original downloaded recovery bundle; preserve its sealed layout.
- `scripts/kaggle`, `scripts/kaggle_pgsr`: existing Kaggle handoff. `scripts/visualization`: existing Blender importers; preserve objects/coordinates, use a separate scene and verify after editing. Do not reinstall addons without need.
- `scripts/research`: read-only evidence recovery, plots, cleanup planning and verification. `research/metrics/<experiment_id>/`: complete recovered CSV histories and loss definitions/summaries. Formal reports: `research/reports/`.
- Evaluation/comparison: `.agents/skills/experiment-evaluation/SKILL.md`; literature: `paper-research`; extraction: `mesh-reconstruction`; geometric losses: `geometry-regularization` under `.agents/skills/`.
- SfM instructions are currently `.agents/skills/preprocessing/{SKILL-video-sfm.md,plan-video-sfm.md,tech-video-sfm.md}` plus `docs/video_sfm.md`. Preserve these files and do not create a duplicate skill tree.
- Detailed research/architecture/tech/phase guidance is indexed by `docs/README.md`. Read in that order when relevant; executable source and logs establish actual state. Record conflicts instead of silently changing the protocol.

## Deletion, Git and audit safety
Never use `git reset --hard`, `git clean -fd`, blanket experiment deletion, force-push or overwrite unrelated runs. Keep secrets, datasets, environments and large generated artifacts out of commits; retain configs and summary metrics. Do not remove installed packages or files outside this repository.
Before any cleanup, inventory paths/sizes/hashes/dependencies and publish an exact deletion plan. Only individually verified regenerable bytecode/tool cache files outside protected/installed trees may be auto-deleted. Check resolved containment, unchanged size/mtime/hash, then unlink one file; no recursive deletion of experiment directories.
Checkpoints, PLY/meshes, original Kaggle outputs, logs, datasets, COLMAP files, configs, failed/interrupted evidence and unknown dependencies require explicit user approval even when copies have equal hashes. Preserve final model, validation-best model if objectively identified, required resume state and all reported-result evidence. Never delete the only copy. Record any approved path migration and update/verify references reversibly.
Use safe restricted checkpoint loading on CPU and lightweight structure/hash checks first. Never use unrestricted pickle for unknown serialized objects. Retained optimizer state is essential for meaningful resume.
