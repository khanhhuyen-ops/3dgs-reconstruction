# Implementation Plan — Video to SfM (Ignatius, Windows, GPU-strict)

## 0. Scope, nguyên tắc và định nghĩa thành công

**Mục tiêu gần:** từ `Ignatius.mp4` tạo bộ frame đủ chồng lấp, chạy GPU SIFT + GPU matching, xây dựng sparse SfM với GPU bundle adjustment nếu môi trường hỗ trợ. Xuất `camera poses`, `intrinsics`, `points3D.ply`, logs/metrics có thể tái lập.

**Mục tiêu nghiên cứu tiếp theo:** viết SfM V1 theo đúng thuật toán incremental SfM: matching → geometric verification → two-view initialization → triangulation → PnP registration → bundle adjustment.

**Không làm trong task này:** VGGT, Gaussian Splatting, TSDF, dense MVS, meshing, dùng ground truth để định hướng tối ưu.

### Quy tắc STOP/GO

- `GO`: GPU CUDA và các binary GPU cần thiết được xác thực.
- `STOP_GPU_UNAVAILABLE`: không có CUDA, PyTorch CUDA không hoạt động, hoặc feature/matching không có GPU.
- `BLOCKED_GPU_BA`: COLMAP SIFT + matching có GPU nhưng mapper không có **BA trên GPU**. Tiếp tục được đến checkpoint của feature/matching, **không** âm thầm chạy mapper CPU.
- `STOP_INPUT_INVALID`: không có video/không đọc được video/frame sai định dạng.
- `STOP_MAPPING_INVALID`: mapping không sinh model hợp lệ, không được xuất file mô phỏng.

## Phase 0 — Repo audit và GPU/CUDA preflight (ưu tiên số 1)

**Tasks**

- [ ] Kiểm tra `datasets/tanks_and_temples/Ignatius/raw/Ignatius.mp4` tồn tại và `ffprobe` đọc được.
- [ ] Kiểm tra có `ffmpeg`, `ffprobe`, `nvidia-smi`, Python, `torch` CUDA, `colmap` hoặc `COLMAP.bat`.
- [ ] `nvidia-smi`: xác nhận GPU index 0, tên GPU, tổng VRAM, driver.
- [ ] PyTorch: `torch.cuda.is_available()`, `torch.cuda.get_device_name(0)`, chạy tensor nhỏ trên `cuda:0`.
- [ ] COLMAP: lấy version và `feature_extractor -h`, `sequential_matcher -h`, `mapper -h`. Xác định **flags thật của bản đang cài**.
- [ ] Xác định GPU BA backend: `CASPAR` nếu được build và hoạt động; hoặc CUDA-enabled Ceres thực sự. Không suy luận chỉ từ CLI có option `--Mapper.ba_use_gpu`.
- [ ] Tạo `src/common/gpu.py` và `scripts/run_sfm.ps1` với preflight bắt buộc.
- [ ] Lưu `environment.json` và `gpu_report.json` với version/capabilities.

**Done khi:** có báo cáo PASS/BLOCKED rõ ràng, không có silent CPU fallback, không thay đổi dữ liệu raw.

## Phase 1 — Video probing & extraction

**Tasks**

- [ ] Tạo `src/preprocessing/video_probe.py`: lấy duration, codec, width, height, nominal/average FPS, time base, rotation, total frame nếu có.
- [ ] Tạo `src/preprocessing/extract_frames.py`: subprocess FFmpeg, chọn tần suất thời gian có thể cấu hình (`candidate_fps`), xuất JPG chất lượng cao, giới hạn chiều rộng giữ aspect ratio, đặt tên `frame_000001.jpg`...
- [ ] Ghi `extracted_frames/manifest.json`: hash video, FFmpeg version, tham số, danh sách frame và thời gian tương ứng trong video. Không gán timestamp bằng một phép nội suy sai khi video variable-frame-rate.
- [ ] Có `--resume`: kiểm tra manifest và cache key; xử lý file thiếu/hỏng; ghi ra thư mục tạm rồi atomic rename để không nhầm output nửa chừng.
- [ ] Có chế độ tránh overwrite và xóa output cũ nếu chưa được phép.

**Gợi ý ban đầu:** `candidate_fps: 2.0`, `max_width: 1280`. Đây là preset thử nghiệm cho VRAM 6 GB, không phải tham số tối ưu học thuật.

**Done khi:** ảnh mở được, timestamp hợp lệ, báo số frame và các khoảng thời gian, dữ liệu raw không đổi.

## Phase 2 — Keyframe selection có thể tái lập

**Tasks**

- [ ] Tạo `src/preprocessing/select_keyframes.py` đọc `extracted_frames/manifest.json`.
- [ ] Phân chia video thành khoảng thời gian để tránh dồn ảnh vào một đoạn.
- [ ] Tính chỉ số blur (Laplacian/gradient), mức phơi sáng và trùng lặp ảnh; các phép toán tensor lớn dùng PyTorch **cuda:0**.
- [ ] Chọn frame chất lượng tốt trong mỗi khoảng, giữ temporal overlap, thêm các frame bắc cầu nếu hai frame liên tiếp quá xa.
- [ ] Có kiểm soát `target_selected_frames: 96`, `min_selected_frames: 40`, `max_selected_frames: 120`, `seed: 42`.
- [ ] Copy/link ảnh đã chọn vào `selected_frames/`; tạo `manifest.json`, `manifest.csv`, kiểm chứng đường dẫn, thứ tự và duplicate hash.
- [ ] Trường hợp score không đáng tin (blur/duplicate heuristic): report và fallback **thuật toán chọn đều theo thời gian**, không fallback tính toán bắt buộc GPU sang CPU.

**Done khi:** số frame phù hợp, có góc nhìn biến thiên, không tập trung ở các đoạn dừng/đứng yên; có log lọc bỏ frame.

## Phase 3 — SfM V0: COLMAP baseline GPU-strict

**Tasks**

- [ ] Tạo `src/geometry/sfm/colmap_runner.py` hỗ trợ Windows path, `COLMAP.bat`, `--dry-run`, `--resume`, logs/return codes.
- [ ] Feature extractor: GPU 0, `SIFT`, camera model `SIMPLE_RADIAL`, `single_camera: true` khi thực sự là một camera không đổi zoom/focus.
- [ ] Matching: `sequential_matcher` GPU 0, overlap có thể cấu hình; nếu số frame nhỏ và cần closure có thể thử `exhaustive_matcher` theo cấu hình riêng (không tự thay đổi âm thầm).
- [ ] Verify database có images, keypoints/descriptors, verified two-view geometries.
- [ ] Detect GPU BA backend: ưu tiên COLMAP >= 4.1 với Caspar được compile; nếu không có thì xem xét Ceres CUDA build thật.
- [ ] Nếu không có GPU BA: ghi `BLOCKED_GPU_BA`, giữ database và logs; **không chạy mapper mặc định**.
- [ ] Nếu có: chạy `mapper` với GPU BA cho cả local và global; kiểm chứng log/backend. Kiểm tra ít nhất một model hợp lệ, chọn model có số registered images lớn nhất, không mặc định `sparse/0`.
- [ ] Xuất `sparse_txt/` (camera/image/point), `sparse.ply`, `metrics.json`, `commands.json`.
- [ ] Optional: chạy thử 48 frame trước; tăng lên 96/120 nếu chất lượng tốt và GPU memory đủ.

**Done khi:** SIFT/matching xác thực chạy GPU; nếu mapping được bật thì BA GPU cũng đã xác thực; có point cloud/pose thật. Nếu block, báo chính xác checkpoint đã hoàn tất.

## Phase 4 — SfM V1: Custom Incremental SfM (nghiên cứu thuật toán)

Triển khai incremental theo thứ tự, test nhỏ trước, mỗi phép toán ma trận nặng phải đặt trên CUDA; dùng FP32/FP64 (không dùng FP16 cho hình học nhạy số).

- [ ] `features.py`: đọc SIFT keypoints/descriptors từ dữ liệu COLMAP GPU V0; sau này mới cân nhắc feature extractor GPU khác.
- [ ] `matching.py`: GPU descriptor matching + ratio test + mutual check; batching/chunk để không OOM.
- [ ] `two_view.py`: normalized eight-point E/F estimation, batched GPU RANSAC, cheirality, recover relative R/t; nếu K chưa biết, chỉ hỗ trợ nhánh đủ intrinsics trước, không tự đoán K không kiểm chứng.
- [ ] `triangulation.py`: GPU linear triangulation/SVD, positive-depth/angle/reprojection filtering.
- [ ] `registration.py`: 2D–3D track manager, GPU PnP-RANSAC + pose refinement hoặc kernel phù hợp; không thay bằng `cv2.solvePnPRansac` CPU trong chế độ strict.
- [ ] `bundle_adjustment.py`: GPU optimization cho pose/3D (e.g., PyTorch CUDA hoặc solver GPU đã xác minh); tránh dense full Jacobian vượt VRAM.
- [ ] `reconstruction.py`: điều phối incremental registration, track management, thêm điểm, local/global BA.
- [ ] Kiểm thử trên dữ liệu tổng hợp với pose biết trước rồi đối chiếu COLMAP V0 trên cùng số frame.

**Lưu ý:** Custom GPU PnP/BA có độ phức tạp cao; không coi V1 hoàn thành chỉ vì có skeleton hoặc trả số liệu giả. Nếu chưa có GPU solver tin cậy, đánh dấu TODO/BLOCKED rõ ràng.

## Phase 5 — Evaluation, quality gates và reproducibility

- [ ] `src/geometry/sfm/model_io.py`: parse camera model đúng conventions, camera center `C = -Rᵀ t`; xuất JSON/PLY.
- [ ] `metrics.py`: selected frames, registered frames, ratio, valid 3D points, track length, mean/median reprojection error (px), elapsed time từng stage, GPU backend, peak GPU memory nếu có.
- [ ] Tạo `tests/test_preprocessing.py`, `tests/test_gpu_policy.py`, `tests/test_sfm_geometry.py` và ít nhất một smoke test trên video cực ngắn có sẵn hoặc synthetic.
- [ ] Test fail-fast khi `CUDA_VISIBLE_DEVICES` ẩn GPU; test khi Caspar không có thì mapper bị chặn.
- [ ] Hash config và input, lưu command + tool versions + seed, so sánh V0/V1 cùng frame IDs.
- [ ] Viết hướng dẫn chạy 3 command: preflight, preprocessing, V0, sau đó custom V1.

### Tiêu chí đánh giá (không phải cam kết kết quả)

| Kiểm tra | Tiêu chí |
| --- | --- |
| Input | Có video và metadata hợp lệ |
| Frame | Thời gian tăng dần, không duplicate, không vượt giới hạn config |
| GPU | Không có silent fallback; log backend chính xác |
| SfM V0 | >= 2 camera được đăng ký, >= 1 điểm 3D, model thật đọc được |
| Geometry | Pose finite, positive depths, lỗi chiếu ngược có phân phối hợp lý |
| Reproducibility | Manifest/config, phiên bản công cụ, số liệu và trạng thái đủ để tái chạy |

## Lệnh sử dụng sau khi Codex đã viết code

> Các câu lệnh **là giao diện cần Codex triển khai**, chưa phải lệnh có thể chạy ngay bây giờ.

```powershell
python -m src.pipeline --config configs/sfm_ignatius.yaml --stage preflight
python -m src.pipeline --config configs/sfm_ignatius.yaml --stage preprocess
python -m src.pipeline --config configs/sfm_ignatius.yaml --stage colmap --resume
python -m src.pipeline --config configs/sfm_ignatius.yaml --stage custom --resume
```

## Quy trình khi bị block GPU BA

1. Nêu rõ COLMAP version và kết quả `mapper -h`/GPU BA capability.
2. Giữ artifacts `selected_frames`, SIFT và matched database; đừng làm lại các giai đoạn đã cache.
3. Ghi hướng dẫn cụ thể về build/install COLMAP với Caspar (`CASPAR_ENABLED=ON`) hoặc CUDA-enabled Ceres rồi kiểm chứng lại.
4. **Không** chuyển sang CPU mapper, **không** ghi trạng thái COMPLETE.

## Tài liệu nền

- Incremental SfM: Schönberger & Frahm, *Structure-from-Motion Revisited* (CVPR 2016).
- COLMAP CLI: https://colmap.github.io/cli.html
- COLMAP GPU Bundle Adjustment: https://colmap.github.io/faq.html#speedup-bundle-adjustment
- FFmpeg filters: https://ffmpeg.org/ffmpeg-filters.html
