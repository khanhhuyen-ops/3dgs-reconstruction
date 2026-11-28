# Research status

Verified audit date: 2026-11-22. This compact status supersedes execution claims in older plans; historical reports remain unchanged.

| Experiment ID | Model | Observed iteration / target | Status | Loss rows |
| --- | --- | --- | --- | --- |
| b0-kaggle-c55f001d26 | B0_Original3DGS | 30000/30000 | COMPLETED | 231 |
| ignatius-legacy-b0-low-memory_20261115T141109Z_8ed1c6 | B0_Original3DGS | 30000/30000 | COMPLETED | 302 |
| ignatius-legacy-b0-research-quality_20261115T222504Z_9ddaaa | B0_Original3DGS | 7400/30000 | INTERRUPTED | 76 |
| ignatius-legacy-b0-smoke_20261115T133608Z_4ef4c4 | B0_Original3DGS | 0/1200 | UNKNOWN | 0 |
| ignatius-legacy-b0-smoke_20261115T134942Z_346824 | B0_Original3DGS | 1200/1200 | COMPLETED | 26 |
| kaggle-pgsr-geometry-smoke | B2_PGSR | 300/300 | COMPLETED | 30 |
| local-smoke-v1 | B2_PGSR | 300/300 | COMPLETED | 30 |
| pgsr-kaggle-c269454133 | B2_PGSR | 21298/30000 | INTERRUPTED | 212 |

**Implemented/trained methods:** B0 Original3DGS and B2 PGSR. **2DGS has not been used in a project training run.** GOF, Ours and project mesh extraction are not implemented/verified. Completed smoke schedules are not completed research baselines.

**PGSR:** original Kaggle notebook reports session COMPLETE, but its saved model is PARTIAL_SESSION_BUDGET at21298/30000,3733679 Gaussians. Full-state CPU integrity is verified. No audit training/resume/inference was launched. The canonical reorganized folder is not directly resume-compatible; preserve the original flat `phase03_results/recovery/run` and its seal. A future continuation needs explicit authorization.

**B0:** full-resolution Kaggle30k has33 test views (PSNR20.913033, SSIM0.695186, LPIPS0.269434). Low-memory30k has33 test views at resolution divisor4 (23.375393/0.839926/0.115017). Those resolutions differ. Local full-resolution training stopped after logged7400, with published checkpoint7000.

**Evidence:** 907 recovered loss rows;58 checkpoint/weight files,53 unique hashes;56 restricted-load project-format full-state checks (including unit fixtures) and2 pretrained/other weight reads. No TensorBoard events found. Phase04 update2026-11-24: Kaggle version2 reconstructed a PGSR CUDA-TSDF triangle mesh from checkpoint21298 with1,131,692 vertices,2,184,436 faces and197 integrated train views. The PLY SHA256 and ZIP integrity passed locally and Blender5.2.3 imported the mesh at identity transform. No GT accuracy/Chamfer/F-score was measured.

**Limits:** PGSR standalone L1/SSIM and additive exposure-loss histories were not recorded; validation LPIPS histories are absent. The historical Ignatius raw-video run remains GPU-BA blocked; successful legacy image-set experiments do not change that. Single-seed, single-scene, transductive fixed-pose experiments do not support SOTA claims.

**Next steps, not executed:** approve a PGSR continuation if desired; finish30k and repeat the frozen extraction variant; integrate B1 2DGS only under a separate training request; establish mesh/GT protocol before geometry claims.

Indexes: [experiments](../research/metadata/experiment_registry.json), [loss histories](../research/metrics/), [comparison](../research/comparisons/model_comparison.md), [cleanup approval list](../research/cleanup/approval_required.csv), [Vietnamese report](../research/reports/bao_cao_phan_tich_huan_luyen.md).

## Museum video · 2026-11-26

The independent GPU-strict pipeline extracted 1,142 candidates and selected 160 keyframes. The pinned COLMAP 4.2.1 CASPAR build registered 159 views. It passed the GPU probe and the mapper log check after structure-less registration was disabled. A derived copy dropped 289 tracks with duplicate image observations, leaving 31,447 qualified sparse points with mean reprojection error 0.528 px. The original mapper output and earlier attempts remain available.

The frozen scene `museum-video-v1-04702c7617547b96` has 159 undistorted 1280×720 views, a 119/20/20 split and 27,435 train-only initialization points. Both local CUDA smoke runs completed with full-state resume, PLY reload, and held-out RGB evaluation. The smoke configurations used different resolutions and training subsets. PGSR exported depth, normals and validity masks for its eight train views.

| Method | Completed smoke schedule | Sampled loss rows | Held-out PNG PSNR / SSIM / LPIPS |
| --- | ---: | ---: | --- |
| B0 Original3DGS | 1,200/1,200 | 26 | 12.7081 / 0.3185 / 0.4757 |
| B2 PGSR | 300/300 | 30 | 8.8862 / 0.0843 / 0.7375 |

[Museum commands](tech/museum_usage.md) · [Museum report](../research/reports/museum_bao_cao_tai_tao.md) · [Kaggle prepared scene](https://www.kaggle.com/datasets/vthkhnhhuyn/museum-prepared-video-v1).
