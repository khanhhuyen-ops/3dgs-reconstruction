# Master plan — research roadmap

**Read-only audit, 2026-11-22:** [current evidence and remaining gates](../RESEARCH_STATUS.md) supersede the dated status paragraphs below. PGSR saved a resumable research state at21298/30000; notebook session COMPLETE does not mean training complete. 2DGS has not been trained and no project predicted triangle mesh was found. This audit does not authorize continuing training or advancing phases. [Recovered experiments](../../research/metadata/experiment_registry.json), [Vietnamese report](../../research/reports/bao_cao_phan_tich_huan_luyen.md).

**Current Phase03 checkpoint17/11/2026:** user đã yêu cầu actual PGSR implementation/training. `src/pgsr` và isolated native CUDA environment đã có; numerical tests, real smoke80→resume300, full-state/RNG restore và PLY reload đã pass. Kaggle research job đã submit/RUNNING, final30000/evaluation/Blender PGSR pending. [B2 usage](../tech/pgsr_training_usage.md), [local evidence](../phase_03_local_validation.json). Phase04–07 chưa thực thi; các status PGSR planned ở checkpoint cũ bên dưới là historical.

Checkpoint tài liệu ngày14/11/2026 được giữ tại [audit](../repository_audit.md) và [documentation audit](../documentation_audit.md). [Phase01 report](../phase_01_implementation_report.md) là evidence lịch sử. Request tiếp theo ngày15/11 đã cho phép **Phase02 Original3DGS implementation, environment build và actual training** trên prepared legacy experiment; user xác nhận tiếp tục Phase02. Phase03–07 chưa được triển khai/chạy. CLI/config/source pin hiện tại: [B0 usage](../tech/b0_training_usage.md).

## Current checkpoint

**Phase02 B0 ngày17/11:** full-resolution Kaggle đã hoàn tất30000 iterations,3.381.717 Gaussians,33 test views PSNR20,9130 /SSIM0,6952 /LPIPS0,2694; final bundle đã tải và `IMPORTED_FINAL`. Actual native PLY CUDA render trên Windows khớp PNG Kaggle ở test view1. Blender review giữ full XYZ và bounded Gaussian previews trong scene riêng. [Evidence nhận kết quả](../phase_02_kaggle_result_import.json), [Phase03 entry readiness](../phase_03_entry_readiness.md) ghi shared data/profile và gates còn mở. PGSR chưa triển khai/train; B0 checkpoint không dùng làm PGSR resume state.

**Phase02 B0 ngày16/11:** smoke1.200 và low-memory30.000 iterations hoàn tất, actual CUDA/native loader/backward/ADC/full checkpoint recovery/PLY roundtrip/evaluation đã kiểm chứng. Low-memory197 train views,76.223→249.213 Gaussians,33 test views PSNR23,3754 /SSIM0,8399 /LPIPS0,1150. Full-resolution dừng `BLOCKED_RESOURCE` tại7401; checkpoint7000 được giữ cho [Kaggle migration](../tech/kaggle_phase02_usage.md). [Phase02 report](../phase_02_implementation_report.md) và [validation JSON](../phase_02_validation_summary.json) ghi artifacts/tests/resources/giới hạn; không gọi local resource variant là paper benchmark reproduction.

Preprocessing/CUDA features+matching có evidence cũ. Video reconstruction `BLOCKED_GPU_BA`, Phase01 video `BLOCKED_INPUT`; không sửa SfM/thay input. Legacy riêng (`ba_use_gpu=false`) đã preparation/readiness/resume pass:263 images, split197/33/33,76.223 train-init points. Phase02 actual unmodified B0 SceneInfo loader với frozen-split adapter đã pass; canonical data/poses/normalization giữ nguyên. Phase01 PGSR diagnostic tự lọc thêm19.249 points và reader overwrite PLY vẫn là vấn đề cần method-specific adapter khi Phase03 được yêu cầu. [Phase01 usage](../tech/phase_01_usage.md), [historical report](../phase_01_implementation_report.md).

## Dependency roadmap

| Phase | Plan / trách nhiệm duy nhất | Entry gate | Exit gate |
| --- | --- | --- | --- |
| 01 | [Data preparation](phase_01_data_preparation.md) | User yêu cầu implementation; selected source/model | Prepared dataset/split/normalization/qualification, no raw writes |
| 02 | [Original3DGS](phase_02_3dgs_baseline.md) | G1–G2 data; compatible CUDA build | B0 forward/backward/resume; RGB validation tools |
| 03 | [PGSR](phase_03_pgsr_training.md) | Phase02 shared infrastructure, qualified data | B2 geometry/variant/exposure/numerics checked |
| 04 | [Surface/mesh](phase_04_mesh_extraction.md) | Valid checkpoint+depth contracts, CUDA mesher | Raw/clean meshes parse, export/coverage checks |
| 05 | [Proposed method](phase_05_proposed_method.md) | B2 + meshing validation + novelty review | Anchor variant/parity/tests, ablation matrix frozen |
| 06 | [Baselines](phase_06_baselines.md) | B0/B2/Ours stable; B1 build; resource/protocol lock | Comparable saved runs/renders/meshes; failures retained |
| 07 | [Evaluation/reporting](phase_07_evaluation.md) | Frozen predictions + verified evaluator/GT protocol | Reproducible final tables/figures/limitations |

Phase06 owns B1 adapter and optional B3 branch; Phase02/03 sở hữu B0/B2 implementations, không triển khai lại trong Phase06. Phase02 tạo shared RGB metric validator tối thiểu để dùng validation; Phase07 mở rộng/freeze final evaluator và báo cáo, không đổi metric definition dựa test. Phase04 chỉ synthetic/consistency geometry validation; GT mesh evaluation ở Phase07. Phase05 là hypothesis implementation/pilot; comparative experiment matrix chạy ở Phase06.

Baseline protocol phải được thiết kế/freeze trước research runs dù execution nằm Phase06/07. Không chờ xem test scores mới quyết định split/voxel/threshold/budget. Nếu cần thay protocol phải tạo version/group mới và đánh giá lại nhóm tương ứng, giữ kết quả cũ.

## Decision registry

| ID | Quyết định thiết kế | Nguồn / hệ quả |
| --- | --- | --- |
| D01 | Giữ SfM/raw/config/output read-only | User task; mọi derived data có provenance riêng |
| D02 | Default video; legacy image_set separate | Audit `project.ini` và video state |
| D03 | PGSR official-code variant làm B2 | [Paper/code notes](../research/pgsr_implementation_notes.md), pinSHA |
| D04 | B0 rendering native; B0+TSDF là diagnostic | P02 không native mesh, không đổi baseline identity |
| D05 | B1/B2/Ours main TSDF, B3 opacity/MT separate | [Method comparison](../research/method_comparison.md) |
| D06 | Ours v0 chỉ sparse confidence anchors | [Proposal](../research/proposed_method.md), overlap P03/P05/P09/CDGS |
| D07 | CUDA strict ở required workloads | Không CPU fallback; missing backend/resource thì blocker |
| D08 | Fixed-pose transductive SfM disclosure | [Data spec](../tech/data_specification.md), holdout isolation downstream |
| D09 | Historical14/11: hai profiles chỉ trong Markdown | Phase02 B0 profiles được triển khai riêng theo D14; không biến presets nhánh khác thành measured feasibility |
| D10 | Existing `.agent/` giữ; new modular skills `.agents/` | Không duplicate/move skill SfM hoặc tạo source placeholders |
| D11 | Legacy Phase01 riêng; drop toàn duplicate/outside-image track chỉ trên derived export | User cho phép source riêng ngày15/11; [report](../phase_01_implementation_report.md) ghi IDs/counts/hashes; default video/reject giữ nguyên |
| D12 | Native dataset layout và split phải có adapter theo pinned source | PGSR đọc sparse/, 3DGS/2DGS sparse/0; [registry](../../configs/data/loader_references.json), [usage](../tech/phase_01_usage.md); native COLMAP reader pass không đồng nghĩa full scene loader pass |
| D13 | B0 dùng official2023 commit414b553 và native CUDA rasterizer/KNN submodules | [Dependency lock](../../configs/training/dependency_lock.json); không depth/exposure/AA additions2024, không PGSR development |
| D14 | Ba B0 profiles tách smoke/low-memory/uncapped research | [B0 usage](../tech/b0_training_usage.md); cap có logged deferred ADC events, không gắn low-memory score thành paper benchmark |
| D15 | RGB evaluator theo frozen project split và AlexNet LPIPS v0.1 với weights SHA-256 | sRGB rounded PNG, SSIM native, per-view mean; khác VGG choice của native metrics.py được công bố |
| D16 | Giữ native ADC→Adam order và final-iteration skip; checkpoint thêm RNG/sampler | Native capture/restore giữ nguyên, save immutable files, actual CUDA row-state/RNG tests và smoke resume pass |
| D17 | B2 dùng pinned PGSR plane renderer và official TNT guarded variant | [Usage](../tech/pgsr_training_usage.md): read-only prepared adapter, matched N/delta, native reductions và invalid-domain masks; không Ours |
| D18 | B2 research cùng full-resolution/frozen split B0, ncc_scale1 | Không upscale gray3840; source/data/config identity strict, PGSR full state riêng và train-only exposure |

## Risks và quality gates

Video chưa có poses/points; legacy B0 riêng đã qualify/build/train/evaluate thực. Risk còn lại gồm uncapped densification trên6GB, camera/depth/normal conventions và method-specific loader differences của Phase03, transductive SfM disclosure, multiseed/fair-comparison protocol. Official mesh GT alignment/masks/threshold vẫn cần xác minh ở phase evaluation, không dùng RGB metrics để thay geometry metrics.

G1 nguồn/quality → G2 data projection/split → G3 CUDA render/backward → G4 state/resume/density → G5 depth/mesh → G6 evaluator/protocol → G7 claims/evidence. Gate không đạt thì checkpoint/traces giữ nguyên; không progress bằng dummy artifact.

## Session handoff

Phase02 request đã được xác nhận và B0 low-memory đã hoàn tất; [report](../phase_02_implementation_report.md), [commands](../tech/b0_training_usage.md) và frozen run artifacts là handoff hiện tại. Video branch vẫn thiếu qualified sparse model; legacy không thay nguồn video. Phase03 cần scope riêng, pin/build/verify PGSR-specific contracts; không giả native B0 checkpoint tương thích PGSR. Không mesh hoặc method khác được chạy trong Phase02.
