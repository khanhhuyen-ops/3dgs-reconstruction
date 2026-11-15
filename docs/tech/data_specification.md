# Data specification — prepared dataset v1

Ngày15/11: loader/schema Phase01 đã triển khai ở `src/data/`; CLI/artifact/NPZ contract thực tế tại [usage](phase_01_usage.md), results/gates tại [report](../phase_01_implementation_report.md). [Audit14/11](../repository_audit.md) giữ evidence lịch sử. Mathematical convention: [foundations](../research/mathematical_foundations.md). New dataset luôn read-only đối với raw/model gốc; training/anchors vẫn thuộc phase sau.

## Source qualification

Input là **một** explicit `source_mode`, `image_root`, `model_dir`, source-manifest/reference và file hashes. Default `video`; không model trong `video_v0` → `BLOCKED_INPUT`, không đọc legacy `sparse/0` thay thế. Nếu user chọn image_set về sau, tạo dataset/experimentID riêng, ghi BA history `ba_use_gpu=false`, timestamp/video identity chưa biết là `null`.

Chấp nhận COLMAP TXT/BIN khi parser hợp lệ; BIN/TXT cùng tồn tại phải chọn authoritative format và so sánh counts/camera/pose/point checksums sau conversion trên derived copy. Không mutate original export. Model có `rigs/frames` không được bỏ qua nếu chứa nontrivial sensor transforms; legacy one-camera trivial rig có thể qualify bằng verified equivalence.

V1 training support undistorted PINHOLE/SIMPLE_PINHOLE. Distorted SIMPLE_RADIAL được inverse-warp sang ảnh/camera copy và lưu mapping; Phase01 export PINHOLE. Model khác bị reject kèm tên, không bỏ distortion coefficient để giả pinhole. Điểm/pose finite, rotations orthonormal/det+1, quaternion normalized, f>0, valid IDs/track reciprocity và positive observed depth. Keypoint ngoài transformed image: default reject; legacy explicit `outside_image_tracks: drop_track` loại toàn point track và lưu IDs/counts, không clamp tọa độ.

Track có duplicate observations cùng image: diagnostic trước; default qualification reject. Legacy config riêng triển khai **loại toàn track bị duplicate**, không sửa pose; log reason/IDs/counts và validate lại support/coverage. Chọn canonical observation là variant khác, không tự sửa gốc. 3.942 legacy tracks chỉ bị loại ở derived export, original model nguyên trạng; số đo mới trong [report](../phase_01_implementation_report.md).

## Schema và tensors

| Field | Type/shape | Invariant |
| --- | --- | --- |
| `schema_version`, `dataset_id` | string | v1 và content-addressed identity |
| `source` | object | mode, model/image/manifest hashes, BA provenance, no GT input |
| `images[]` | records | stable image/frame IDs, relative paths, checksum, width/height, timestamp nullable |
| `camera` mỗi image | K[3,3], Tcw[4,4], Twc[4,4] float64 metadata | inverse parity, right-handed convention, undistorted flag |
| `image_transform` | resize/crop/orientation + undistortion map metadata | áp dụng cùng lên keypoints; không dùng A tuyến tính thay nonlinear undistortion |
| `points` | IDs[N], XYZ[N,3], RGB[N,3] | accepted point mask, source provenance, RGB0..255 metadata |
| `tracks` | ragged observation table | imageID/keypointidx, observedXY[M,2], pointID; reciprocal |
| `normalization` | center[3], scale scalar, inverse | positive scale, same similarity across scene |
| `splits` | train/val/test IDs + policy/hash | disjoint, complete accepted set, deterministic |
| `qualification` | checks/stats/rejected IDs | errors không bị ẩn bằng SUCCESS |
| Loader output | RGB[3,H,W]float32 `[0,1]`, optional mask[1,H,W]bool | CPU decode; transfer CUDA ở compute boundary |
| Anchor batch | xy[M,2], z[M], weight[M], valid[M] | train only, no NaNs, same transformed pixel convention |

## Scene normalization

Thiết kế: center từ robust median **train camera centers**, scale là inverse90th-percentile train-camera radius; reject radius≈0. Không lẫn camera-center normalization với point-based transform native của method: adapter phải disable hoặc compose rõ, lưu cả ma trận. Test projection trước/sau similarity bằng cùng distortion convention. Depth units gắn scale, mesh export đảo transform để về SfM units.

Không dùng GT để estimate normalization hoặc chọn views. Poses freeze cho mọi baseline; pose refinement là protocol khác, không bật riêng Ours.

## Split v1 đề xuất và leakage boundary

Stable ordered accepted image list: video theo timestamp rồi frameID, image_set theo numeric-aware name rồi imageID. Zero-based index `k`: test khi `k mod8=0`, validation khi `k mod8=4`, còn lại train. Phase01 freeze accepted IDs và ghi actual counts trong `splits.json`; không gán trước96/263 thành training count. Tiny synthetic fixtures chỉ là unit tests, không benchmark.

Chỉ train RGB đi vào loss, density statistics, neighbor selection, appearance fitting, anchor/confidence và meshing input views. Initial points cần ≥2 unique train observations; RGB point colors từ all-view COLMAP là leakage risk, phải recompute từ train observations/images hoặc khai báo sử dụng. Gỡ test track entries không xóa việc original BA đã dùng test images.

**Primary downstream protocol: fixed-pose, transductive SfM.** Camera/point geometry có thể được COLMAP estimate từ toàn bộ image set; report rõ điều này. Không gọi đây là strict unseen-view reconstruction. Train-only points/appearance isolation chỉ ngăn thêm leakage downstream. Strict train-only-SfM protocol cần một SfM experiment khác, test-pose estimation policy riêng và request tương ứng; không thực hiện ở đây.

Uniform every8 split có near-duplicate temporal leakage risk trong video. Secondary blocked temporal holdout có thể preregister để stress-test generalization; không thay primary split khi biết test score. Không chia các baseline khác nhau hoặc đổi split để camera register tốt hơn.

## Validation acceptance

Image names/sizes/hash coverage100%; no missing IDs, no raw writes; reciprocal train tracks; no GT access; roundtrip/projective parity tolerance≤1e−4px trên synthetic FP64 fixture. Test exact expected pixel values với resize/crop/undistortion. Thực tế scene diagnostics phân phối residual/coverage phải review, không chỉ dùng minimum2cameras/1point của old SfM stage làm chất lượng đủ training.

Loader phải giới hạn CPU cache, lazy image loading, method-specific camera export và explicit split. Không tạo fake images/poses để vượt gate. Schema mismatch → fail with field/path, không defaults camera identity.
