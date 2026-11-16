# Hướng dẫn tài liệu nghiên cứu

Mốc16/11/2026: **Phase02 B0 low-memory training/evaluation hoàn tất**, full-resolution dừng bởi resource guard; [Kaggle usage](tech/kaggle_phase02_usage.md) hướng dẫn tiếp tục; Phase03–07 chưa triển khai. Đọc [AGENTS](../AGENTS.md), [repository audit lịch sử](repository_audit.md), [master plan](plans/master_plan.md) rồi tài liệu theo task. Video SfM vẫn `BLOCKED_GPU_BA`; prepared legacy là thí nghiệm riêng được user cho phép. [Phase01 report](phase_01_implementation_report.md) giữ evidence lịch sử; [Phase02 report](phase_02_implementation_report.md) ghi actual results/artifacts, [B0 usage](tech/b0_training_usage.md) mô tả commands/contracts.

## Research — sources, phương pháp và tính mới

| File | Nội dung |
| --- | --- |
| [paper_inventory.md](research/paper_inventory.md) | Toàn bộ14 PDF/209pages, hash, vai trò, version và official implementations/access limits |
| [literature_review.md](research/literature_review.md) | 20 fields cho từng paper, sources theo trang/mục/equation/table, paper numbers tách khỏi local results |
| [method_comparison.md](research/method_comparison.md) | B0/B1/B2/B3/Ours, native/common meshing, checkpoint compatibility |
| [mathematical_foundations.md](research/mathematical_foundations.md) | Camera/Gaussian/depth/normal/homography/fusion/field math và điều kiện áp dụng |
| [research_gaps.md](research/research_gaps.md) | Novelty overlap, counterexamples và falsification criteria |
| [proposed_method.md](research/proposed_method.md) | COLMAP confidence/sparse anchors, proposed equations/schedules/ablations, chưa scientific validation |
| [pgsr_implementation_notes.md](research/pgsr_implementation_notes.md) | Pinned PGSR source so với paper: weights/schedule/patches/edge and gradient behavior |

## Architecture — boundaries và data flow

| File | Nội dung |
| --- | --- |
| [system_architecture.md](architecture/system_architecture.md) | Pipeline tổng thể, module ownership và quality gates |
| [data_flow.md](architecture/data_flow.md) | Artifact contracts/lineage, source IDs, read/write boundaries và invalidation |
| [training_architecture.md](architecture/training_architecture.md) | Parameter/render/loss/optimizer/ADC/checkpoint loop và data isolation |
| [mesh_architecture.md](architecture/mesh_architecture.md) | Depth/native-field routes, chunking, cleaning/export boundaries |

## Technical specifications — hợp đồng triển khai tương lai

| File | Nội dung |
| --- | --- |
| [tech_stack.md](tech/tech_stack.md) | Libraries/backend choices, GPU/CPU roles và compatibility gates |
| [development_environment.md](tech/development_environment.md) | Live metadata audit, Windows/Python/CUDA build risks, chưa cài hoặc benchmark |
| [data_specification.md](tech/data_specification.md) | Source qualification, schema, intrinsics/poses/undistortion, train-only observations, split và normalization |
| [gaussian_representation.md](tech/gaussian_representation.md) | Per-method tensors/activations, initialization và checkpoint incompatibility |
| [training_specification.md](tech/training_specification.md) | Loop/Adam/ADC/resume, Development+Research Quality presets và resource/OOM policy |
| [loss_specification.md](tech/loss_specification.md) | Formulas, tensor shapes/masks, gradient flow, schedules, numerics và required tests |
| [meshing_specification.md](tech/meshing_specification.md) | TSDF/Poisson/GOF/MT decisions, units/backends, filtering/cleaning/export/tests |
| [evaluation_specification.md](tech/evaluation_specification.md) | Fair rendering/mesh metrics, scale/alignment/masks/thresholds, budgets và reporting |
| [reproducibility.md](tech/reproducibility.md) | Hashes, run manifests, version locks, seeds, cache keys, status/evidence requirements |
| [phase_01_usage.md](tech/phase_01_usage.md) | CLI/config/schema thực thi, derived filters, train-only init, transactional resume và compatibility limits |
| [b0_training_usage.md](tech/b0_training_usage.md) | Phase02 actual native integration/CUDA build, runnable profiles, training/resume/render/metrics và deviations |

## Plans — mỗi phase có18 mục triển khai

| File | Nội dung |
| --- | --- |
| [master_plan.md](plans/master_plan.md) | Current checkpoint, dependency graph, decisions, gates và handoff |
| [phase_01_data_preparation.md](plans/phase_01_data_preparation.md) | Existing COLMAP output → qualified prepared dataset |
| [phase_02_3dgs_baseline.md](plans/phase_02_3dgs_baseline.md) | Original3DGS B0, CUDA renderer/optimizer/ADC/state và RGB validator |
| [phase_03_pgsr_training.md](plans/phase_03_pgsr_training.md) | B2 plane rendering, single/multi-view, exposure và native variant |
| [phase_04_mesh_extraction.md](plans/phase_04_mesh_extraction.md) | Depth/normal → fusion/mesh → cleaning/export/Blender smoke |
| [phase_05_proposed_method.md](plans/phase_05_proposed_method.md) | Candidate Ours, confidence cache/loss và ablation preparation |
| [phase_06_baselines.md](plans/phase_06_baselines.md) | B1 integration, optional B3, comparable runs/seeds/failure matrix |
| [phase_07_evaluation.md](plans/phase_07_evaluation.md) | Frozen predictions/GT protocol → evaluator/tables/figures/report |

## Agent skills

| Skill | Kích hoạt |
| --- | --- |
| [colmap-data](../.agents/skills/colmap-data/SKILL.md) | COLMAP output qualification, undistortion, loader, splits |
| [gaussian-training](../.agents/skills/gaussian-training/SKILL.md) | Gaussian training/renderer/ADC/checkpoint |
| [geometry-regularization](../.agents/skills/geometry-regularization/SKILL.md) | Depth/normal/multi-view/confidence objectives |
| [mesh-reconstruction](../.agents/skills/mesh-reconstruction/SKILL.md) | Surface extraction, fusion, cleaning/export |
| [experiment-evaluation](../.agents/skills/experiment-evaluation/SKILL.md) | Protocol/baselines/ablations/metrics/reporting |
| [paper-research](../.agents/skills/paper-research/SKILL.md) | Cited PDF review, code comparison và novelty assessment |

## Audit và tài liệu cũ được bảo toàn

| File | Nội dung / trạng thái |
| --- | --- |
| [repository_audit.md](repository_audit.md) | Repository, dataset/SfM, runtime metadata, qualification diagnostics và blockers ngày14/11 |
| [documentation_audit.md](documentation_audit.md) | Coverage/link/metadata/scope validation của task tài liệu |
| [video_sfm.md](video_sfm.md) | Existing SfM operations; không sửa |
| [implementation_report.md](implementation_report.md) | Evidence lịch sử10/11; `.git` note là trạng thái lịch sử, audit mới có Git |
| [implementation_metrics.json](implementation_metrics.json) | Metrics lịch sử; không đổi thành new training results |
| [tool_versions.md](tool_versions.md) | Tool provenance cũ, không version lock của future trainer |
| [SKILL-video-sfm.md](../.agent/skills/preprocessing/SKILL-video-sfm.md) | Legacy SfM skill giữ nguyên |
| [plan-video-sfm.md](../.agent/skills/preprocessing/plan-video-sfm.md) | Legacy SfM plan; phase numbering riêng với roadmap mới |
| [tech-video-sfm.md](../.agent/skills/preprocessing/tech-video-sfm.md) | Legacy SfM tech; không tự thay backend/policy |

Phase01 có `src/data/`/`configs/data/`; Phase02 hiện có `src/gaussian_splatting/`, minimal RGB `src/evaluation/` và `configs/training/`. Các thành phần Phase03–07 vẫn proposed/planned, không suy diễn B0 execution thành PGSR/mesh/evaluation benchmark hoàn chỉnh. Nguồn video chưa đủ sparse model; legacy không vượt gate video. Không tự sửa BA hoặc bắt đầu phase tiếp theo.
