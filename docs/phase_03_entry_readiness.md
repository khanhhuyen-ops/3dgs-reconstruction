# Phase03 entry — kết quả Kaggle và kiểm tra Blender

Ngày kiểm chứng: **17/11/2026**. Trạng thái: **READY_FOR_PHASE03_IMPLEMENTATION_GATES**. Phase02 B0 full-resolution đã hoàn tất, tải về và kiểm chứng; PGSR chưa được triển khai hoặc train trong task này.

## 1. Artifact Phase02 đã nhận

Notebook `vthkhnhhuyn/phase02-ignatius-run-all` báo `COMPLETE`. Bundle `phase02_results_final` có 145 file được xác minh checksum; importer ghi **IMPORTED_FINAL** tại:

```text
outputs/tanks_and_temples/Ignatius/research/B0_3DGS/b0-kaggle-c55f001d26/
```

| Artifact | Kết quả thực tế |
| --- | --- |
| Full checkpoint | `checkpoints/iteration_030000.pth`, 30000 iterations / 29999 Adam calls |
| Checkpoint SHA256 | `9c9538604d015403f355f41c4ac42ae989454b73d8bb55876843cd94ad4484d5` |
| Gaussian PLY | `point_cloud/iteration_030000/point_cloud.ply`, 3.381.717 Gaussians |
| PLY SHA256 | `72e396873d5355d7fea27394ca8c6e411cc29e5a09e2b9ad27bdca1a76a485fd` |
| Actual held-out evaluation | 33 test views; PSNR **20,913033**, SSIM **0,695186**, LPIPS **0,269434** |
| Resume state | Native parameters, Adam moments/steps, ADC, RNG và train-only camera sampler đã parse/validate |
| Kaggle training duration | 8908,29 giây cho segment từ checkpoint7000 đến30000; không phải tổng thời gian từ iteration1 |
| Kaggle resources | Tesla T4; peak Torch allocated 7.349.913.088 B, reserved 10.160.701.440 B; sampled device-wide peak 9549 MiB có scope riêng |

[Import/runtime evidence](phase_02_kaggle_result_import.json) lưu receipt, measured results, environment và cloud inspection. Low-memory local có249.213 Gaussians và profile/resolution khác; không xếp hạng hai run như một comparison group công bằng. Chưa đo mesh accuracy/Chamfer/F-score.

## 2. Kiểm chứng trên Windows và Blender

Đã reload **PLY đầy đủ**, render test camera1 bằng official rasterizer trên RTX3050, `cuda:0`, Torch2.4.1+cu121. PNG1920×1080 trùng từng kênh 8-bit với PNG Kaggle ở view này: mean/max difference0. Đây là one-view inference parity; không khẳng định float bitwise parity mọi view hoặc đã resume optimizer Linux trên Windows. Peak Torch reserved của bài kiểm tra là2.566.914.048 B. Evidence:

```text
outputs/visualization/phase02_20261117/native_cuda_validation/verification.json
outputs/visualization/phase02_20261117/native_cuda_validation/windows_cuda_render.png
```

Đã kết nối Blender5.2.3 LTS, dùng addon KIRI có sẵn, tạo scene riêng `Phase02_PointCloud_Review_20261117` và lưu:

```text
outputs/visualization/phase02_20261117/phase02_pointcloud_review.blend
```

| Object trong scene review | Số điểm gốc giữ đầy đủ | Native Gaussian preview |
| --- | ---: | ---: |
| `B0_Kaggle_FullResolution_ALL_XYZ` | 3.381.717 | 200000 điểm lấy mẫu đều, seed20261117 |
| `B0_Local_LowMemory_ALL_XYZ` | 249213 | 200000 điểm lấy mẫu đều, cùng seed |
| `Sparse_TrainInitialization_ALL_XYZ` | 76223 | XYZ/RGB nguồn sparse, không giả là Gaussian |

`ALL_XYZ` giữ mọi vị trí điểm với identity transform, được ẩn để giảm tải viewport. Có thể unhide, chọn object, bật overlays và dùng Edit Mode để xem các vertices. Sample PLY giữ nguyên các native Gaussian fields của điểm được chọn, không crop/prune theo opacity. Bản Kaggle preview chỉ dùng khoảng5,9% số Gaussians; các lỗ trong preview không là completeness của full model. KIRI SH-DC/material preview không thay renderer CUDA dùng tính metrics.

Đã xem tượng và cảnh xung quanh qua camera COLMAP, kiểm tra native fields, lưu diagnostic PNG cho hai preview. Camera Blender dùng `Twc @ diag(1,-1,-1,1)`; không biến đổi XYZ hoặc align theo GT. Image `CUDA_FULL_MODEL_3381717_POINTS` được pack vào `.blend` để xem full native render. Scene gốc và ba object `Camera`, `Cube`, `Light` được giữ nguyên. Các object là point cloud/Gaussian splats; chưa phải triangle mesh.

Tọa độ dùng arbitrary monocular/SfM units, chưa được metric scale alignment; độ dài hiển thị trong Blender không là phép đo mét thực tế.

Script tái lập: [phase02_pointcloud_review.py](../scripts/visualization/phase02_pointcloud_review.py). `prepare()` tạo inspection JSON, all-centers NPZ và sample PLY; `build_blender()` chạy trong live Blender, tạo scene mới; `verify_cuda_ply()` kiểm tra one-view native CUDA riêng. Không sửa PLY/checkpoint/dataset gốc. Logs và dữ liệu lớn nằm trong outputs, không commit.

## 3. Handoff cố định cho Phase03

Machine-readable handoff:

```text
outputs/phase03_handoff/ignatius_legacy_v1_20261117.json
```

- Prepared dataset: `datasets/tanks_and_temples/Ignatius/prepared/legacy_v1/ignatius-legacy-v1-6a3308c22df11122`.
- Metadata SHA256: `7636dd218be5cdb1bfa10b76b59a40b2b72cfb6129b0bf2d8fac4283a767913b`.
- Split hash: `fbd4eb563ba6b5418165ce4d402e04ec123117f56b2fe0c601a35efaad3ba2b4`.
- Shared reference: seed42, resolution1 ở1920×1080,197train/33val/33test, camera poses/normalization giữ nguyên; train neighbors và exposure chỉ dùng train views.
- PGSR initialization dùng76223 train-only SfM points trong prepared dataset. **Không resume checkpoint B0 thành PGSR**; B0 là baseline/checkpoint tham chiếu với representation và native optimizer khác.
- Nguồn là legacy `image_set` đã được cho phép làm experiment riêng, historical CPU BA và fixed-pose transductive protocol đã công bố. Video SfM vẫn `BLOCKED_GPU_BA`. GT geometry chỉ ở evaluator, không truyền vào training/neighbors/selection.

## 4. Gate còn phải thực hiện cho PGSR

Theo [Phase03 plan](plans/phase_03_pgsr_training.md) và [PGSR paper/code notes](research/pgsr_implementation_notes.md), candidate là official-code variant ở reviewed snapshot `de24f1a38b350387e8d8fe381b2cd70c1ae946e7`. Snapshot này chưa được build/chạy trong task nhận kết quả; không gọi là backend PGSR đã hoạt động.

1. Xác minh checkout, submodules, license và build environment PGSR riêng; giữ nguyên backend B0 đã kiểm chứng.
2. Method-specific camera/initialization adapter và projection parity trên frozen data; không tự đổi split hoặc point filtering để chạy được. Phase01 diagnostic đã ghi PGSR tự lọc thêm19.249 points và reader có thể overwrite PLY ([master plan](plans/master_plan.md)); cần xác định contract và derived copy phù hợp, không cho native reader ghi vào prepared/raw/SfM gốc. Đây là evidence lịch sử, chưa chạy lại PGSR trong task nhận kết quả.
3. Train-only neighbor graph, train exposure state và canonical held-out appearance.
4. Synthetic plane-depth/normal, signs, projection/homography, constant/empty masks và per-loss gradient tests trên CUDA.
5. Full optimizer/exposure/RNG/ADC interruption/resume và source/config identity.
6. Declared subset smoke, đo VRAM/quality trước research schedule; B0 fit trên T4 không chứng minh PGSR fit cùng GPU.

Chưa bật Ours hoặc trích xuất mesh. Handoff này xác nhận dữ liệu và baseline có thể dùng để bắt đầu các gate implementation Phase03.
