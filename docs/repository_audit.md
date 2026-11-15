# Repository audit — 14/11/2026

Trạng thái công việc: documentation-first. Không chạy training, cài dependency, sửa SfM hoặc tái tạo output. Kết luận dưới đây lấy từ file hiện tại và các phép đọc/diagnostic CPU; số đo GPU cũ là evidence trong logs ngày 10/11, không phải benchmark mới.

## 1. Inventory trước thay đổi

Git HEAD: `74875ab6b2e3142bc55e672b39c0bbfc9072e236`. Git hoạt động; `git status --short` ban đầu chỉ có `?? paper/`. Ghi chú “không có .git” trong báo cáo 10/11 là lịch sử, không còn mô tả checkout hiện tại.

| Thành phần | Thực tế | Quyết định |
| --- | --- | --- |
| `src/preprocessing/`, `src/geometry/sfm/`, `src/common/`, `src/pipeline.py` | Pipeline video/SfM V0 đã có | Bảo vệ nguyên trạng |
| `configs/sfm_ignatius.yaml`, `sfm_ignatius_smoke.yaml` | Video strict CUDA, CASPAR BA gate | Không thay config |
| `scripts/run_sfm.ps1`, `tests/` | Wrapper; 3 file test | Không sửa/chạy GPU tests trong phiên tài liệu |
| `.agent/skills/preprocessing/` | `SKILL-video-sfm.md`, `plan-video-sfm.md`, `tech-video-sfm.md` | Giữ hướng dẫn SfM cũ; tên `plan.md`/`tech.md` bên trong là tham chiếu lịch sử sai tên |
| `.agents/skills/video-sfm/` | Chưa tồn tại | AGENTS trỏ về bộ thực tế; không tạo bản sao skill SfM |
| `docs/` | `video_sfm.md`, `implementation_report.md`, `tool_versions.md`, `implementation_metrics.json` | Giữ báo cáo lịch sử; mở rộng cùng `docs/` |
| `paper/` | 14 PDF, 209 trang | Inventory hash và review tại [research](research/paper_inventory.md) |
| Gaussian training/renderer, loader hậu COLMAP, meshing/evaluation mới | Chưa có implementation | Chỉ thiết kế trong tài liệu |

Không tạo các thư mục source tương lai hoặc executable mới. Sáu skills mới nằm trong `.agents/skills/`; bộ `.agent/` cũ không di chuyển.

Kiểm tra bảo toàn cuối phiên có **ngoại lệ sidecar SQLite**: truy vấn `mode=ro` đã thay nội dung cache `video_smoke_v0/database.db-shm` và mtime của `video_v0/database.db-shm`. Main databases, WAL và reconstruction/config/code/raw-video hashes không đổi. Không khôi phục bằng ghi đè cache; chi tiết và giới hạn tại [documentation audit](documentation_audit.md). Vì vậy không tuyên bố mọi byte trong toàn cây outputs đều bất biến.

## 2. Dataset và hai nguồn SfM

Gốc scene: `datasets/tanks_and_temples/Ignatius/`.

| Nguồn | Input/artifact | Kết luận |
| --- | --- | --- |
| Video mặc định | `raw/Ignatius.mp4`, 3.067.060.079 bytes; SHA-256 `244802e503a003b212b2da2e7e2265cc15a97b19ec1933487096f94417132017` | Hash đọc lại khớp manifest |
| Video preprocessing | `preprocessing/extracted_frames/video_v0/manifest.json`: 524; `selected_frames/video_v0/manifest.json`: 96 | Có PTS, transform, image hash, scores; chưa kiểm tra lại từng pixel của mọi frame |
| Video SfM | `outputs/tanks_and_temples/Ignatius/sfm_colmap/video_v0/` | `database.db`, state, metrics, commands và logs có thật; `BLOCKED_GPU_BA` |
| Smoke video | Các đường dẫn tương ứng `video_smoke_v0` | 48 candidate, 12 selected; BA cũng blocked |
| Ảnh có sẵn | `raw/image_set/`: 263 JPG | Một nguồn riêng, không được đổi tên thành frame video |
| SfM cũ từ image_set | `outputs/tanks_and_temples/Ignatius/sfm_colmap/` | Có model, TXT, sparse PLY, database; cần qualification trước downstream |
| Ground truth | `raw/ground_truth/`: 1 file | Chỉ inventory metadata; không đọc làm input/selection/training |

Trong bảng trên, các đường dẫn `preprocessing/` và `raw/` tính từ gốc scene; các đường dẫn `outputs/` tính từ repository root.

## 3. Model cũ: xác minh read-only

`sfm_colmap/sparse/0/` có `cameras.bin` (64 bytes), `images.bin` (32.111.397), `points3D.bin` (9.880.856), `rigs.bin`, `frames.bin`, `project.ini`. TXT tương ứng ở `sfm_colmap/txt/`; PLY tại `sfm_colmap/sparse.ply` (1.285.029 bytes). Không mặc định model số 0 cho các lần chạy tương lai.

`project.ini` xác nhận `image_path=.../raw/image_set` và `ba_use_gpu=false`. Đây không phải reconstruction video GPU-strict. Không đủ evidence để gắn lịch sử GPU SIFT/matching của video cho model này.

Đọc TXT bằng `src.geometry.sfm.model_io.read_model`:

| Diagnostic đo trong audit | Giá trị | Giới hạn |
| --- | ---: | --- |
| Cameras / registered images / points | 1 / 263 / 85.656 | TXT parse được; chưa đối chiếu nội dung BIN↔TXT |
| Camera | SIMPLE_RADIAL, 1920×1080 | `f=1159.4932405028942`, `cx=960`, `cy=540`, `k=-0.028378193557882524` |
| Tên ảnh và kích thước | 263/263 khớp | PIL đọc header; không phải kiểm định photometric |
| Observation có point hợp lệ | 689.049 | Tính lại projection có radial distortion |
| Mean / median / p95 reprojection residual | 1,139587 / 0,978000 / 2,639945 px | Euclidean distance mỗi tracked observation, không phải error từ header |
| Mean / median track length | 8,044375 / 5 | Đếm observation, không đồng nghĩa số camera phân biệt |
| Observation depth ≤ 0 | 0 | Trên tập observation đã kiểm tra |
| Track entry sai index / không reciprocal | 0 / 0 | Kiểm tra từng track entry |
| Tracks có nhiều observation cùng image_id | **3.942** | Validator hiện tại yêu cầu một observation/camera; chưa kết luận root cause hoặc model hỏng |
| PLY | Parse hợp lệ, 85.656 vertices | Sparse points, không có bằng chứng triangle mesh |

`model_metrics(model, 263)` **thất bại** với `ValueError: Point must have >=2 distinct camera observations`. Kiểm tra tiếp cho thấy không có singleton track; điều kiện duplicate image trong track kích hoạt cùng thông báo. Không sửa validator hoặc model trong phiên này. Các residual ở bảng là diagnostic riêng, không nâng kết quả quality gate thành SUCCESS. Phase 01 phải xác định policy loại bỏ/canonicalize trên bản dẫn xuất, thống kê các track bị ảnh hưởng và kiểm tra lại.

## 4. Video V0 và GPU evidence

`video_v0/metrics.json`: input 96, keypoints/descriptors 495.006, verified pairs 317, verified inliers 108.426. Các reconstruction metrics vẫn `null`; `commands.json` có feature extractor và sequential matcher, không mapper. 108.426 là pairwise matches, không phải 3D points.

Logs `video_v0/logs/features_742b461d.log` và `matching_3c27416a.log` ghi GPU SIFT extractor/matcher; command JSON có `use_gpu=1`, `gpu_index=0`. BA probe `audits/20261110T154520Z_06778c/ba_probe_40a17f38/report.json` và metrics ghi crash `3221226505` (`0xC0000409`); CASPAR chưa được xác minh. Giữ `BLOCKED_GPU_BA`, không gọi mapper.

## 5. Environment đọc lại

Python 3.14.6; torch 2.11.0+cu128; NumPy 2.4.6; Pillow 12.1.1; PyYAML 6.0.3; PyMuPDF 1.27.2.2. `nvcc --version` báo Toolkit 12.1. Không suy luận torch cu128 dùng compiler 12.8; build extension cần toolchain tương thích. `cl` và `cmake` không có trong PATH, chưa chứng minh không cài ở máy. COLMAP/FFmpeg không trong PATH; executable hiện có được ghi bởi resolver và logs cũ. Open3D/Nerfstudio không có package metadata trong interpreter hiện tại.

Không chạy tensor CUDA, benchmark, preflight hoặc SfM để audit tài liệu. Phiên này không xác minh CUDA runtime mới; GPU evidence là logs hiện có. Xem [development environment](tech/development_environment.md).

## 6. Gap, blockers và quyết định phạm vi

1. **Nguồn chính video chưa có sparse model**: Phase 01 có thể được triển khai sau yêu cầu mới nhưng chưa đủ input để chạy hoàn chỉnh. Không khắc phục BA trong task tài liệu.
2. **Legacy image_set cần protocol riêng và qualification**: không tự dùng làm video baseline; muốn dùng phải có yêu cầu chọn nguồn và experiment riêng. Không yêu cầu permission trong phiên tài liệu vì chưa sử dụng nó để train.
3. **Dataset contract/split chưa có**: phải giữ camera conventions, undistortion và scale; policy fixed-pose/transductive phải công bố.
4. **Windows/CUDA extension chưa qua build**: Python 3.14, Toolkit 12.1 và torch cu128 không phải stack training đã được kiểm chứng.
5. **Mesh backend**: Open3D CPU TSDF trong paper không đáp ứng một stage được chỉ định CUDA. Backend CUDA phải được kiểm chứng riêng; không coi package Windows bất kỳ có CUDA.
6. **Tính mới**: sparse SfM anchors đã có trong 3DGSR; DN-Splatter, GVGS và CDGS tạo overlap đáng kể. [Proposed method](research/proposed_method.md) là thiết kế giả thuyết.

Đọc [master plan](plans/master_plan.md) và [documentation audit](documentation_audit.md) để tiếp tục. Không thực thi Phase 01 trước yêu cầu tiếp theo.
