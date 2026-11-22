# Project structure

Start with [research status](RESEARCH_STATUS.md); do not infer execution from old plans.

| Location | Purpose / reading policy |
| --- | --- |
| `AGENTS.md` | Stable English language, provenance, CUDA and deletion rules |
| `src/preprocessing`, `src/geometry/sfm`, `src/pipeline.py` | Protected video/SfM implementation and implemented orchestration |
| `src/data`, `configs/data` | Prepared COLMAP adapter, qualification, frozen camera/data/split contracts |
| `src/gaussian_splatting` | B0 Original3DGS native adapter; no native mesh output |
| `src/pgsr` | B2 PGSR native plane-renderer adapter, geometry losses and full-state checkpoints |
| `src/evaluation` | Existing B0 RGB evaluator and shared metric definitions/weights |
| `scripts/kaggle`, `scripts/kaggle_pgsr` | Existing separate-method handoff, recovery and import helpers |
| `scripts/visualization` | Existing Blender review/import helpers; viewport previews are not metric renders |
| `scripts/research` | Audit discovery, evidence extraction, restricted checkpoint inspection, figures and safe cleanup |
| `configs/training` | Versioned B0/B2 dependency locks and declared resource profiles |
| `outputs/tanks_and_temples/Ignatius/research/B0_3DGS` | Canonical B0 runs, logs, configs, checkpoints and Gaussian PLYs |
| `outputs/tanks_and_temples/Ignatius/research/B2_PGSR` | Canonical B2 runs; the imported research folder has a documented non-native layout |
| `phase03_results/recovery/run` | Original flat PGSR recovery bundle; preserve its sealed contents |
| `outputs/kaggle_download`, `outputs/kaggle_handoff` | Original local Kaggle transport copies, with exact duplicates indexed rather than removed |
| `datasets/tanks_and_temples/Ignatius/raw` | Immutable video, legacy images and evaluation-only GT |
| `datasets/.../prepared/legacy_v1/...` | Frozen downstream legacy scene; 197 train / 33 val / 33 test |
| `research/metadata` | Experiment registry, all scientific artifact paths, checkpoint audit and inventory summary |
| `research/metrics/<experiment_id>` | Full recovered CSV loss histories, definitions, summary statistics and periodic validation |
| `research/comparisons` | Recorded metric table and pairwise comparability decisions |
| `research/figures` | PNG/SVG/PDF plots and source index; missing-data omissions are explicit |
| `research/cleanup` | Complete inventory, dependencies, approval list, plan and executed-cleanup receipt |
| `research/reports` | Formal Vietnamese scientific report |
| `docs/research`, `docs/architecture`, `docs/tech`, `docs/plans` | Existing detailed references; dated historical claims are not current evidence |
| `.tools`, `external`, `.git`, `paper` | Managed environments/external sources/history/papers; avoid routine broad reads and preserve contents |

No datasets, checkpoints or experiment folders were moved for this audit. Relative native paths and sealed bundle layouts were preserved. B1 2DGS and a project mesh extraction implementation were not found; corresponding roadmap diagrams are planned work.

Use `research/cleanup/file_inventory.csv` before repeating a full filesystem walk. Hash-cache reuse requires unchanged byte size and nanosecond mtime. Recompute hashes when integrity is in doubt. Inventory timestamps and scope are recorded in `research/metadata/discovery.json`; figures and reports may have been generated after that snapshot.

Context controls: the installed CLI reports `codex-cli 0.167.0`. The project uses concise `AGENTS.md`, `.gitignore`, targeted `rg` reads and cached indexes. No `.codexignore` was added because support was not established. Official [AGENTS.md discovery guidance](https://developers.openai.com/codex/guides/agents-md) and [configuration reference](https://developers.openai.com/codex/config-reference) document project instructions and their size limit; ignoring a file is not a security boundary.
