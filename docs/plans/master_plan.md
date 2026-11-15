# Master plan — research roadmap

Checkpoint tài liệu ngày14/11/2026 được giữ tại [audit](../repository_audit.md) và [documentation audit](../documentation_audit.md). Request ngày15/11 chỉ cho phép **Phase01 implementation** và thí nghiệm legacy riêng; Phase02–07 vẫn `NOT_IMPLEMENTED/NOT_RUN`. [Phase01 report](../phase_01_implementation_report.md) ghi evidence mới, không sửa reports cũ.

## Current checkpoint

Preprocessing/CUDA features+matching có evidence cũ. Video reconstruction `BLOCKED_GPU_BA`, Phase01 video `BLOCKED_INPUT`; không tự sửa SfM/thay input. `src/data/` đã triển khai BIN/TXT qualification, derived undistortion/filters, frozen splits, train-only initialization, lazy loader và readiness. Legacy riêng (`ba_use_gpu=false`) đã preparation/readiness/resume pass: 263 images, split197/33/33, 76.223 train-init points. Native readers 3DGS/2DGS pass; PGSR tự lọc thêm19.249 points và full scene reader luôn overwrite PLY, cần adapter để giữ common initialization. Full scene loaders chưa chạy. Xem [usage](../tech/phase_01_usage.md) và [report](../phase_01_implementation_report.md) trước khi handoff Phase02.

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
| D09 | Hai profiles chỉ trong Markdown | Chưa runnable config, chưa GPU feasibility claim |
| D10 | Existing `.agent/` giữ; new modular skills `.agents/` | Không duplicate/move skill SfM hoặc tạo source placeholders |
| D11 | Legacy Phase01 riêng; drop toàn duplicate/outside-image track chỉ trên derived export | User cho phép source riêng ngày15/11; [report](../phase_01_implementation_report.md) ghi IDs/counts/hashes; default video/reject giữ nguyên |
| D12 | Native dataset layout và split phải có adapter theo pinned source | PGSR đọc sparse/, 3DGS/2DGS sparse/0; [registry](../../configs/data/loader_references.json), [usage](../tech/phase_01_usage.md); native COLMAP reader pass không đồng nghĩa full scene loader pass |

## Risks và quality gates

Ưu tiên giải quyết qualified source trước build/training: hiện video chưa có poses/points. Kế tiếp là Windows toolchain/Python-CUDA mismatch, 6GB densification peaks, depth/normal conventions, duplicate tracks và leakage. Official GT alignment/masks/threshold thiếu sẽ block final metric, không block việc soạn architecture.

G1 nguồn/quality → G2 data projection/split → G3 CUDA render/backward → G4 state/resume/density → G5 depth/mesh → G6 evaluator/protocol → G7 claims/evidence. Gate không đạt thì checkpoint/traces giữ nguyên; không progress bằng dummy artifact.

## Session handoff

Phase01 implementation và report mới: [usage](../tech/phase_01_usage.md), [results](../phase_01_implementation_report.md). Không có training/mesh result mới. Chạy real video branch vẫn thiếu qualified sparse model; legacy không thay nguồn video. Handoff Phase02 cần user request mới, kiểm thử actual native SceneInfo/split adapter và môi trường CUDA riêng; không bắt đầu training trong Phase01.
