# Phase 01 — Data preparation

Trạng thái ngày15/11/2026: **IMPLEMENTED / acceptance PARTIAL**. User đã yêu cầu Phase01 và cho phép thí nghiệm legacy riêng. Real legacy preparation/readiness/resume pass; video `BLOCKED_INPUT`; full native scene loaders chưa chạy; PGSR native filter/PLY overwrite cần adapter. Không gọi toàn Phase01 complete. [Report](../phase_01_implementation_report.md) và [CLI/contract](../tech/phase_01_usage.md) ghi kết quả thực tế; numerical thresholds vẫn là project heuristics.

## 1. Phase overview

Chuyển **qualified COLMAP output** sang prepared dataset cho Gaussian training bằng `src/data/`. Mặc định video_v0 hiện thiếu sparse model; legacy dùng YAML/experiment riêng. Không reconstruct SfM trong phase này.

## 2. Research objective

Giữ projection/geometry/provenance đúng khi undistort, resize, normalize và split. Chuẩn bị observations đủ để nghiên cứu anchors mà không rò test RGB.

## 3. Scientific references

P02 §3–5; P14 §IV; P03 §4.3 p.6; [COLMAP format](https://colmap.github.io/format.html); [data spec](../tech/data_specification.md).

## 4. Current dependencies

[Audit lịch sử](../repository_audit.md): video branch BLOCKED_GPU_BA; legacy có duplicate tracks và ba_use_gpu=false. Source legacy đã được user chọn riêng cho integration; qualification/drop-track policy có report mới, không sửa model gốc. Full scene-loader gate chưa đạt do native dependencies thiếu.

## 5. Input requirements

Explicit source_mode/image_root/model_dir/manifest; complete camera/image/point files, IDs và images khớp; known BA history. Diagnostic synthetic_input không được dùng làm scene. GT không thuộc input.

## 6. Expected outputs

Qualification report; bản sao image/model đã undistort; dataset manifest v1; deterministic splits; normalization và inverse; train-point/observation tables; loader diagnostics. Mọi output mới có dataset ID, không ghi đè nguồn.

## 7. Detailed architecture

Read-only model reader → source qualification → camera/key point undistortion adapter → image transform → split/normalization → train-only point/appearance extraction → loader. Ownership thực tế `src/data/`; parser/SfM/config/tests cũ giữ nguyên. Các bước có transactional output, per-image progress hashes và verified resume.

## 8. Mathematical requirements

World-to-camera/Hamilton quaternion; SIMPLE_RADIAL projection; K/image transform; non-linear undistortion map; similarity transform với t'=s(t+Rc); axial depth; train-ray parallax. [Foundations](../research/mathematical_foundations.md) là canonical convention.

## 9. Implementation tasks

Đã triển khai parser BIN/TXT, explicit source/model selection và fail-closed checks, exact semantic comparison, poses/tracks/images validation, explicit duplicate/outside-image whole-track filters, inverse radial image warp/keypoint transform, frozen splits, train-only colors/observations, lazy loader và readiness. [Usage](../tech/phase_01_usage.md) liệt kê modules/commands/schema. Không mutate raw/original SfM; native full scene loaders chưa được gọi.

## 10. Ordered execution steps

1) Ghi source hashes và input status. 2) Parse/validate IDs/poses/tracks. 3) Resolve qualification issues trên derived copy. 4) Undistort và test projection parity. 5) Freeze split/normalization. 6) Build training subset/color/observations. 7) Validate loader shapes, hashes và write boundary. Nếu bước 1 thiếu model: report blocker, không bịa camera.

## 11. Technical dependencies

NumPy/Pillow/YAML/model_io hiện có dùng cho diagnostics. COLMAP undistortion hoặc phép mapping tương ứng phải được kiểm tra hỗ trợ camera model khi triển khai. Không cần Gaussian renderer cho stage này.

## 12. Hardware and resource constraints

CPU read/parse/decode/geometry diagnostics được phép; không neural inference. Chunk observations và stream images, không load tất cả pixels vào VRAM. CUDA backend chỉ cần khi bắt đầu một operation được chỉ định CUDA.

## 13. Validation strategy

Synthetic cameras/plane với known projections: resize/crop/undistort/similarity/inverse parity; IDs không contiguous; duplicate/broken reciprocity/missing images rejected; source hash unchanged; split disjoint/complete. Real source quality report tách với unit fixture evidence.

## 14. Evaluation metrics

Số images/points/observations được giữ và loại; unique camera count; residual mean/median/p95; positive-depth fraction; train/test counts; coverage và memory/runtime. Không báo dense depth/mesh accuracy.

## 15. Risks and mitigation

Camera flips, double half-pixel, double normalization, test-informed point colors và duplicate support. Mitigate bằng strict convention tests, train-only observation/color derivation và explicit transductive disclosure. Legacy anomalies chưa có root cause, không tự sửa original validator.

## 16. Acceptance criteria

Prepared images/model/observations projection nhất quán; synthetic FP64 parity ≤1e-4px; all refs/hashes resolve; no GT read as input; train/test isolation downstream; read-only source checks. Default duplicate-track policy reject, derived filtering chỉ theo config/log rõ.

## 17. Definition of Done

Artifacts tồn tại/parse, loader shape/color/depth conventions đúng, required tests chạy, report thực tế và blockers. Unit tests + real legacy integration là evidence riêng; không gọi full Phase01 Done khi video model thiếu và full native scene loaders chưa chạy. [Report](../phase_01_implementation_report.md) là acceptance checklist thực tế.

## 18. Next phase dependencies

Phase 02/B1/B2/Ours cùng dataset/split/normalization hash. Anchor cache sẽ dùng observation table; camera repair hoặc new SfM source là task riêng, không hidden prerequisite execution.

