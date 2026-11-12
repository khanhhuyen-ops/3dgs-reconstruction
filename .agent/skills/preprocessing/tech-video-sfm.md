# Technical Specification — Video → SfM (GPU-strict)

## 1. Khung môi trường

- Platform: Windows + PowerShell + VS Code.
- GPU: NVIDIA GeForce RTX 3050 laptop, `cuda:0`, khoảng 6144 MiB VRAM theo thông tin thiết bị đã cung cấp.
- Backend: COLMAP CLI (GPU SIFT, GPU matching, GPU BA nếu được biên dịch), Python 3, PyTorch CUDA, FFmpeg/ffprobe.
- Conventions: dùng đường dẫn tương đối tính từ repo root, lưu paths bằng `pathlib.Path`, không hard-code ổ `C:`.
- Chạy tuần tự từng stage; không load đồng thời nhiều mô hình lớn lên VRAM 6 GB.
- Dataset mặc định: `datasets/tanks_and_temples/Ignatius/raw/Ignatius.mp4`.

### Sự thật về GPU cần tuân thủ

**GPU strict không có nghĩa 100% lệnh hệ điều hành chạy GPU.** FFmpeg demux/decode, filesystem, SQLite, Python subprocess control và một số công đoạn hình học vẫn dùng CPU ở mức nhất định. Yêu cầu "không chạy model trên CPU" được đảm bảo bằng **không chạy neural inference/feature GPU workload trên CPU**, và muốn SfM COLMAP hoàn toàn GPU ở khâu tối ưu phải có CUDA BA backend thật.

COLMAP từ 4.1 hỗ trợ **Caspar GPU BA** nếu được build với `-DCASPAR_ENABLED=ON`; backend `CASPAR` cần được chỉ định cho local/global BA. Nếu không, các bản CUDA COLMAP thông thường có thể vẫn dùng CPU cho mapper/BA mặc định. Ceres CUDA cũng đòi hỏi một build hỗ trợ solver phù hợp. Do đó trong config mặc định `require_gpu_ba: true`, nếu không thỏa thì `BLOCKED_GPU_BA`.

**Không bao giờ giả lập GPU với `device='cuda'` trong config khi phép toán vẫn chạy trên CPU.** Phải kiểm tra thiết bị tensors và log/capability của subprocess.

## 2. Preflight — không chạy khi chưa xác minh

Các lệnh có thể dùng bằng PowerShell (Codex nên bao gói thành Python preflight):

```powershell
nvidia-smi
ffmpeg -version
ffprobe -version
colmap -h   # Hoặc: & "C:\path\to\COLMAP.bat" -h
colmap feature_extractor -h
colmap sequential_matcher -h
colmap mapper -h
python -c "import torch; assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0)); x=torch.randn(4,4,device='cuda:0'); print((x@x).device)"
```

**Kiểm tra version-specific flags** trước khi sinh câu lệnh. COLMAP mới (ví dụ 4.0+) dùng `--FeatureExtraction.use_gpu`, `--FeatureMatching.use_gpu`; một số phiên bản cũ dùng SIFT-specific options. Chỉ được dùng flags xuất hiện trong `-h` của bản đang chạy. Fail nếu GPU flags không có.

Trên Windows, `COLMAP.bat` có thể cần chạy qua `cmd.exe /c` với argument quoting đúng; không ghép shell string tùy tiện từ input. Có thể cấu hình `COLMAP_BIN` trong môi trường hoặc `tools.colmap` ở YAML.

Kiểm tra GPU BA theo thứ tự:

1. Phiên bản/`mapper -h` có các options backend CASPAR (`Mapper.ba_local_backend`, `Mapper.ba_global_backend`) hay không.
2. Có thật sự support CASPAR khi thực thi hay chỉ parse option (build disabled). Kiểm tra qua log/probe tối thiểu và output, **không tự kết luận từ help**.
3. Nếu không có Caspar, xem CUDA Ceres (`Mapper.ba_use_gpu` và build Ceres CUDA/cuDSS). Option hiện diện **không** đồng nghĩa solver chạy GPU.
4. Nếu không thể xác minh: ghi `BLOCKED_GPU_BA` và chỉ chạy tới bước feature/matching.

## 3. Config mặc định cần Codex sinh

`configs/sfm_ignatius.yaml`:

```yaml
project:
  name: 3dgs-mesh-reconstruction
  seed: 42

input:
  dataset: tanks_and_temples
  scene: Ignatius
  video: datasets/tanks_and_temples/Ignatius/raw/Ignatius.mp4
  image_set_reference: datasets/tanks_and_temples/Ignatius/raw/image_set
  ground_truth: datasets/tanks_and_temples/Ignatius/raw/ground_truth
  source_mode: video    # không trộn image_set với frame từ video

preprocessing:
  candidate_fps: 2.0
  max_width: 1280
  image_format: jpg
  jpeg_quality: 2          # FFmpeg -q:v; 2 ~ JPEG chất lượng cao
  target_selected_frames: 96
  min_selected_frames: 40
  max_selected_frames: 120
  use_cuda_quality_metrics: true
  keep_temporal_order: true
  keep_overlap: true

sfm:
  implementation: colmap
  camera_model: SIMPLE_RADIAL
  single_camera: true
  matching: sequential
  sequential_overlap: 10
  max_image_size: 1280
  max_num_features: 4096
  max_num_matches: 8192
  mapper_backend: CASPAR
  min_registered_images: 2

gpu_policy:
  mode: strict
  device: cuda:0
  gpu_index: 0
  allow_cpu_fallback: false
  require_gpu_features: true
  require_gpu_matching: true
  require_gpu_ba: true
  verify_runtime_backend: true
  torch_inference_dtype: float16
  geometry_dtype: float32

paths:
  extracted_frames: datasets/tanks_and_temples/Ignatius/preprocessing/extracted_frames
  selected_frames: datasets/tanks_and_temples/Ignatius/preprocessing/selected_frames
  output: outputs/tanks_and_temples/Ignatius/sfm_colmap
  cache: cache/sfm

tools:
  ffmpeg: ffmpeg
  ffprobe: ffprobe
  colmap: colmap
```

**Ghi chú:** các giá trị trên là starting preset, không khẳng định tốt nhất. Nếu video có ít frame, không ép chọn 40 ảnh trùng lặp; phải fail/report. `sequential_overlap=10` là số lân cận để match, không phải phần trăm. Video quay nhiều vòng có thể cần bổ sung loop matches riêng.

## 4. Video extraction: thực thi & metadata

### 4.1. Lấy thông tin video

```powershell
ffprobe -v error -select_streams v:0 `
  -show_entries stream=codec_name,width,height,avg_frame_rate,r_frame_rate,nb_frames,time_base:format=duration `
  -of json "datasets/tanks_and_temples/Ignatius/raw/Ignatius.mp4"
```

Kiểm tra VFR: `avg_frame_rate` khác `r_frame_rate` có thể là tín hiệu nhưng **không đủ** để khẳng định mọi timestamp. Metadata của frame phải dựa trên timestamp phát sinh ở quá trình decode/filter hoặc PTS thực từ FFprobe/FFmpeg, không suy diễn đơn giản `index/original_fps`.

### 4.2. Trích frame ứng viên

```powershell
ffmpeg -hide_banner -y `
  -i "datasets/tanks_and_temples/Ignatius/raw/Ignatius.mp4" `
  -vf "fps=2,scale=1280:-2:flags=lanczos" `
  -q:v 2 `
  "datasets/tanks_and_temples/Ignatius/preprocessing/extracted_frames/frame_%06d.jpg"
```

Trên CLI thực tế, Codex phải xây dựng filter kích thước tối đa **không upscale** video nhỏ (ví dụ `scale=w='min(iw,1280)':h=-2` cần chuẩn hóa thành số chẵn và thử đúng syntax FFmpeg). Với video crop/rotation metadata, bảo đảm giữ orientation chuẩn.

- CPU decoding **được phép** vì là decode/I/O, không phải neural model; nếu build FFmpeg có NVDEC thì có thể **thử** `-hwaccel cuda` nhưng **phải kiểm chứng khả năng hoạt động**. Không giả định filter `fps/scale` đang chạy GPU.
- Cần lưu PTS/timestamp chính xác; có thể dùng `showinfo`/`ffprobe`, lưu liên kết giữa emitted filename và frame timestamp. Nếu chọn `fps` filter tạo timestamp mới, lưu timestamp **sau filter** và source time tương ứng nếu xác minh được.
- Không dùng `-skip_frame nokey` để chọn frame cho SfM vì đây là **codec I-frame**, không đồng nghĩa keyframe tốt về hình học.
- Luôn ghi lệnh và `ffmpeg_version`; không overwrite dữ liệu raw.

### 4.3. Frame metadata schema

```json
{
  "source_video": "datasets/tanks_and_temples/Ignatius/raw/Ignatius.mp4",
  "source_sha256": "<computed>",
  "ffmpeg_version": "<detected>",
  "sampling_fps": 2.0,
  "frames": [
    {
      "filename": "frame_000001.jpg",
      "timestamp_sec": 0.0,
      "width": 1280,
      "height": 720,
      "sha256": "<computed>"
    }
  ]
}
```

Ví dụ giá trị JSON chỉ để định nghĩa schema; runtime phải tự tính timestamp, kích thước và checksum thật.

## 5. Keyframe selection dành cho SfM

**Nguyên tắc:** chọn đủ góc nhìn/độ chồng lấp, hạn chế motion blur, exposure cực đoan và frame gần như trùng nhau; đừng chỉ chọn ảnh có độ nét cao nhất ở một đoạn ngắn.

Algorithm tham khảo (Codex triển khai có test):

1. Tạo candidate frames bằng sampling thời gian.
2. Load ảnh từng batch, chuyển ảnh xuống `torch.device('cuda:0')` (decode ảnh ở CPU hợp lệ).
3. Tính blur score bằng convolution Laplacian và độ biến thiên gradient trên GPU; tính exposure và near-duplicate similarity trên GPU (histogram/low-res features), không load tất cả ảnh full-resolution cùng lúc.
4. Chia video thành temporal bins; chọn đại diện chất lượng tốt mỗi bin; hạn chế frame trùng, giữ lân cận đủ chồng lấp để matching có thể thành công.
5. Bổ sung frame trung gian nếu khoảng timestamp/camera motion quá lớn. Nếu không thể xác minh overlap từ ảnh, ghi nhận heuristic và dùng geometric matching để kiểm định ở bước V0.
6. Sort theo timestamp, copy selected ảnh, ghi lý do chọn/loại, scores và mapping filename→timestamp.

Khuyến nghị khi OOM: batch 4–8 ảnh thumbnail (ví dụ 320x180) trên GPU; `torch.no_grad()`; giải phóng tensor; không giữ embeddings của mọi ảnh trên GPU; dừng nếu CUDA không khả dụng.

## 6. COLMAP GPU feature extraction và matching

Ví dụ dành cho COLMAP mới, **chỉ thực thi nếu `-h` của bản cài xác nhận đúng flags**:

```powershell
$images = (Resolve-Path "datasets/tanks_and_temples/Ignatius/preprocessing/selected_frames").Path
$out = Join-Path (Get-Location).Path "outputs/tanks_and_temples/Ignatius/sfm_colmap"
New-Item -ItemType Directory -Force "$out\sparse" | Out-Null

colmap feature_extractor `
  --database_path "$out\database.db" `
  --image_path "$images" `
  --ImageReader.single_camera 1 `
  --ImageReader.camera_model SIMPLE_RADIAL `
  --FeatureExtraction.type SIFT `
  --FeatureExtraction.use_gpu 1 `
  --FeatureExtraction.gpu_index 0 `
  --FeatureExtraction.max_image_size 1280 `
  --SiftExtraction.max_num_features 4096

colmap sequential_matcher `
  --database_path "$out\database.db" `
  --FeatureMatching.use_gpu 1 `
  --FeatureMatching.gpu_index 0 `
  --FeatureMatching.max_num_matches 8192 `
  --SequentialMatching.overlap 10
```

GPU matching **không** bảo đảm pose/mapping chạy GPU. SQLite DB lưu features và geometric verification. Chạy `sequential_matcher` chỉ khi thứ tự file phù hợp thứ tự thời gian (`frame_%06d.jpg`). Khi loop/turnback, cân nhắc `exhaustive_matcher` cho <=120 ảnh trong thí nghiệm riêng, ghi rõ chi phí.

## 7. COLMAP mapping: GPU BA bắt buộc

### 7.1. Phương án A — Caspar (ưu tiên nếu đã build)

Trên COLMAP hỗ trợ và được build `CASPAR_ENABLED=ON`, **tham khảo**:

```powershell
colmap mapper `
  --database_path "$out\database.db" `
  --image_path "$images" `
  --output_path "$out\sparse" `
  --Mapper.ba_local_backend CASPAR `
  --Mapper.ba_global_backend CASPAR `
  --Mapper.ba_gpu_index 0
```

- Dùng camera model `SIMPLE_RADIAL` hoặc `PINHOLE` (giới hạn Caspar được tài liệu mô tả).
- Kiểm tra option và runtime; backend yêu cầu build Caspar, **không có sẵn chỉ vì COLMAP hỗ trợ CUDA SIFT**.
- Caspar là experimental; log mọi cảnh báo/unsupported features; không tuyên bố chạy GPU nếu backend bị từ chối hay không khởi chạy.

### 7.2. Phương án B — Ceres CUDA build thật

Nếu không có Caspar và hệ thống có **Ceres CUDA/cuDSS được build với COLMAP**, có thể thử `--Mapper.ba_use_gpu 1` theo phiên bản. Cần xác minh GPU solver **thực sự kích hoạt**; COLMAP có các ngưỡng `BundleAdjustmentCeres.min_num_images_gpu_solver` nên flag một mình không bảo đảm 100% BA trên GPU. Không mặc định rằng binary Windows tải về tích hợp sẵn Ceres CUDA.

### 7.3. Nếu không đủ GPU BA

`BLOCKED_GPU_BA`. Lưu DB + logs + `gpu_report.json`; không tự chạy mapper CPU trong strict mode. Các bộ dữ liệu 6GB có thể cần thử với 48–96 ảnh trước.

### 7.4. Export model thật

Sau khi mapper success:

1. Enumerate các model con hợp lệ dưới `sparse/`; dùng model analyzer/pycolmap để lấy số ảnh đăng ký; chọn model tốt nhất, **không mặc định `sparse/0`**.
2. `colmap model_converter --input_path <actual_model> --output_path <txt_dir> --output_type TXT`.
3. `colmap model_converter --input_path <actual_model> --output_path <sparse.ply> --output_type PLY`.
4. `colmap model_analyzer --path <actual_model>`.
5. Xác minh file sizes, có camera/point thực, TXT/PLY đọc lại được.

## 8. SfM V1: toán học và implementation CUDA

### 8.1. Intrinsics & camera convention

- `K = [[fx,0,cx], [0,fy,cy], [0,0,1]]` cho camera pinhole sau xử lý distortion phù hợp.
- COLMAP pose từ world→camera: `x_cam = R x_world + t`; tâm camera `C_world = -Rᵀ t`.
- Lưu rõ pixel convention và distortion; không giả định điểm ảnh undistorted nếu camera `SIMPLE_RADIAL` có distortion khác 0.
- Monocular SfM chỉ xác định cấu trúc lên tới **global scale**; không báo khoảng cách mét trừ khi có scale alignment riêng.

### 8.2. Feature correspondence

- Đọc keypoints + SIFT descriptors từ DB do COLMAP GPU tạo; hoặc gọi GPU SIFT extractor hỗ trợ rõ ràng.
- Matching GPU: tính nearest neighbors theo **chunk/batch CUDA**, Lowe ratio (ví dụ khởi điểm 0.8, tunable), reciprocal matching.
- Không dùng `cv2.SIFT_create()`/`BFMatcher()` CPU để giả làm custom GPU.

### 8.3. Pair geometry + initialization

- Normalize points bằng `K^-1` khi intrinsics đáng tin cậy.
- Ước lượng E bằng 8-point và RANSAC trên GPU (`torch.linalg.svd`); enforce singular values `(s,s,0)`, kiểm tra Sampson error/inlier ratio.
- Decompose E thành 4 pose candidates, chọn theo cheirality + reprojection error; chọn image pair có đủ inlier và parallax, tránh pure rotation.
- Nếu chưa có intrinsic đáng tin: dùng F + self-calibration được phân định thành task riêng; không gộp F/E và kết luận pose tùy tiện.

### 8.4. Triangulation & tracking

- DLT triangulation từ P1, P2; CUDA batched solve/SVD; reject điểm sau camera, near-zero baseline, reprojection error cao, triangulation angle nhỏ.
- Maintain `track_id`, list observation `(image_id, keypoint_idx)`, 3D point association. Mỗi observation gắn tối đa một 3D point, consistency across images.

### 8.5. Incremental camera registration

- Tìm 2D–3D correspondences từ các tracks hiện có.
- GPU PnP hypothesis generation + RANSAC (hoặc solver GPU đã kiểm chứng); refine R/t bằng robust reprojection residual trên GPU.
- Không gọi OpenCV `solvePnPRansac` CPU trong strict mode; nếu GPU PnP chưa triển khai thì báo blocker V1 đúng trạng thái.
- Cập nhật new tracks, triangulate multi-view, lọc outlier, loop iteration.

### 8.6. Bundle adjustment

Minimize robust reprojection objective:

`min_{R_i,t_i,X_j} Σ_(i,j) ρ( ||π(K_i,R_i,t_i,X_j) - x_ij||² )`

- Optimization GPU thực, ví dụ solver GPU sparse hoặc PyTorch CUDA + local/windowed BA dùng chunks; **không build full dense Jacobian cho toàn scene trên RTX 3050 6GB**.
- Cố định gauge freedom (một camera pose + global scale convention), manifold/rotation parametrization hợp lệ, robust loss.
- Không dùng FP16 để tính E/PnP/triangulation/BA; dùng FP32, tăng FP64 cho đoạn test/solver cần ổn định nếu GPU hỗ trợ và còn VRAM.

## 9. Diagnostics và metrics

`metrics.json` nên chứa:

```json
{
  "status": "SUCCESS | BLOCKED_GPU_BA | FAILED",
  "input_frames": 96,
  "registered_images": null,
  "registration_ratio": null,
  "points3d_count": null,
  "mean_reprojection_error_px": null,
  "runtime_seconds": {},
  "gpu": {
    "name": "<detected>",
    "index": 0,
    "feature_backend": "CUDA",
    "matching_backend": "CUDA",
    "ba_backend": "CASPAR | CERES_CUDA | UNAVAILABLE",
    "verified": false
  }
}
```

**Các số 96, CUDA, v.v. là ví dụ schema**, không phải kết quả thực nghiệm. Dùng `null` khi metric chưa được tính hoặc mapping chưa chạy. Không fake runtime hoặc memory usage. Ghi `registered_images / selected_frames`, `mean track length`, median error khi có thể.

## 10. Các lỗi thường gặp trên máy 6GB và hướng khắc phục

| Triệu chứng | Điều chỉnh được phép |
| --- | --- |
| CUDA OOM khi xử lý frame | giảm thumbnail batch, target frames, max_width |
| SIFT GPU OOM | giảm `FeatureExtraction.max_image_size`, `SiftExtraction.max_num_features` |
| Matching CUDA OOM | giảm `FeatureMatching.max_num_matches`, giảm overlap |
| Ảnh đăng ký ít | kiểm tra overlap, blur, matching pair graph, camera model/intrinsics |
| Reprojection error cao | kiểm tra distortion, tracks, RANSAC/outliers, BA backend |
| Mapper chạy CPU | **STOP `BLOCKED_GPU_BA`**, không fallback |
| Không thấy GPU utilization | kiểm tra log, backend, running stage; không dựa chỉ vào snapshot 0% |
| Output sparse rỗng | báo lỗi thực, không dựng PLY giả |

## 11. Links chính thức

- Codex Skills: https://learn.chatgpt.com/docs/build-skills
- COLMAP CLI & version flags: https://colmap.github.io/cli.html
- COLMAP FAQ (SIFT, GPU BA/Caspar): https://colmap.github.io/faq.html
- COLMAP source release notes: https://colmap.github.io/changelog.html
- FFmpeg filters: https://ffmpeg.org/ffmpeg-filters.html
- Reference paper: Schönberger & Frahm (2016), *Structure-from-Motion Revisited*.
