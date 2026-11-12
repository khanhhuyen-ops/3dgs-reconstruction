# Báo cáo triển khai — 10/11/2026 (Asia/Saigon)

**Trạng thái: `BLOCKED_GPU_BA`; chưa hoàn thành reconstruction từ video.** Code V0 đã được triển khai và chạy thật đến CUDA SIFT + CUDA matching/geometric verification. Mapper không được gọi. Custom V1 chưa triển khai vì chưa đạt điều kiện V0 được kiểm chứng.

## Kết quả theo phase

| Phase | Kết quả thực tế |
| --- | --- |
| 0 — Environment audit | Hoàn thành audit/report. PyTorch CUDA test PASS; standalone CASPAR BA probe crash, ghi `BLOCKED_GPU_BA` |
| 1 — Extraction | PASS: full video → 524 JPG 1280×720, source PTS và SHA-256 manifests |
| 2 — Selection | PASS: 96 frame khác checksum, theo thời gian, scores và duplicate similarity chạy `cuda:0` |
| 3 — COLMAP V0 | CUDA SIFT + matching PASS. 317 verified pairs. Mapping/BA/export scene chưa thực thi |
| 4 — Custom V1 | Chưa triển khai; V0 GPU-strict reconstruction chưa được kiểm chứng |
| 5 — Tests/evaluation | Unit/smoke/resume và geometry diagnostic tests PASS. Metrics reconstruction chưa đo, giữ `null` |

Đầu vào: Ignatius H.264, 3840×2160, 7.845 frame, 30000/1001 FPS, duration 261,7615 giây. SHA-256 video: `244802e503a003b212b2da2e7e2265cc15a97b19ec1933487096f94417132017`; giữ nguyên sau extraction và kiểm tra resume.

Không có `.git` trong workspace, vì vậy không có Git diff/revision để báo; pipeline lưu source content hash. Code trước nhiệm vụ chưa có; output COLMAP cũ dùng `raw/image_set` và ghi `ba_use_gpu=false`, đã được giữ nguyên. Experiment mới nằm dưới `video_v0`/`video_smoke_v0`. Không dùng ground truth.

## Số đo thực tế

| Hạng mục full video | Số đo |
| --- | ---: |
| Candidate frames | 524 |
| Selected / unique checksum | 96 / 96 |
| Khoảng timestamp lớn nhất giữa selected frames | 5,005 giây |
| Keypoints / descriptor rows | 495.006 / 495.006 |
| Verified two-view pairs | 317 |
| Verified inlier matches cộng trên các cặp | 108.426 |
| Frame extraction runtime, gồm probe/hash/I/O | 114,755 giây |
| Selection runtime, gồm validate/copy/I/O | 19,690 giây |
| CUDA SIFT external stage runtime | 4,170 giây |
| CUDA matching + geometric verification runtime | 7,799 giây |
| CUDA selection peak allocated / reserved | 14.971.392 / 25.165.824 bytes |
| COLMAP SIFT peak VRAM lấy mẫu toàn device | 968 MiB |
| COLMAP matching peak VRAM lấy mẫu toàn device | 408 MiB |

VRAM COLMAP lấy mẫu mỗi 0,5 giây bằng `nvidia-smi`; có thể bỏ lỡ peak ngắn và bao gồm process khác. Không gọi đây là exact per-process peak. 108.426 là số inlier matches của các cặp ảnh, không phải số điểm 3D hay tracks của reconstruction.

Subset đã chạy trước full video: 24 giây đầu → 48 candidate frame ở 640px → 12 keyframe → CUDA SIFT/matching, 27 verified pairs và 6.831 inlier matches. Subset cũng dừng `BLOCKED_GPU_BA`.

GPU được xác minh: RTX 3050 6GB Laptop, PyTorch `2.11.0+cu128`, runtime CUDA 12.8; phép nhân tensor trên `cuda:0` có kết quả đúng. COLMAP `4.2.1`, commit `bd1fcf6`, báo build `with CUDA`. Logs thực có `Creating SIFT GPU feature extractor`, `Bind FeatureMatcherWorker to GPU device 0`, `Creating SIFT GPU feature matcher`, đi cùng CLI flags `use_gpu=1`, `gpu_index=0` và database có dữ liệu thật. Decode/resize, I/O, SQLite, orchestration và phần hình học COLMAP được phép dùng CPU.

## Commands đã chạy và validation

Các lệnh Python bên dưới chạy tại repository root, với `COLMAP_BIN` trỏ tới executable hiện có được ghi trong `environment.json`:

```powershell
python -m src.pipeline --config configs/sfm_ignatius.yaml --stage preflight
python -m src.pipeline --config configs/sfm_ignatius_smoke.yaml --stage all --resume
python -m src.pipeline --config configs/sfm_ignatius.yaml --stage all --resume
python -m src.pipeline --config configs/sfm_ignatius.yaml --stage preprocess --resume
python -m src.pipeline --config configs/sfm_ignatius.yaml --stage colmap --resume
python -m src.pipeline --help
python -m src.pipeline --config configs/sfm_ignatius.yaml --stage colmap --dry-run
python -m pytest -q
python -m ruff check src tests
python -m compileall -q src
.\scripts\run_sfm.ps1 -Config configs/sfm_ignatius.yaml -Stage colmap -DryRun
```

- `pytest`: **17 passed in 7.54s**. VFR FFmpeg extraction và CUDA thumbnail selection chạy thật; tests dùng mocked subprocess chỉ kiểm tra control flow, không được dùng để chứng minh GPU runtime.
- `ruff`: **All checks passed**. `compileall`, CLI help, dry-run và PowerShell wrapper dry-run: exit `0`.
- Preprocessing resume: exit `0`, `SUCCESS`. COLMAP resume: exit `2`, `BLOCKED_GPU_BA` như thiết kế.
- Checksum video trước/sau resume khớp. Checksum database trước/sau resume: `f521b42e2a5d711875477862b37cd07162c4441af3b5cce9caf66940ba97db8f`, không đổi; số lệnh trong `commands.json` vẫn là 2 (extraction, matching), không có mapper.
- Diagnostic synthetic model đã được COLMAP `model_converter` đọc/chuyển TXT thành công và `model_analyzer` nhận 3 images, 48 points, 144 observations. Giá trị reported error trong fixture là placeholder của định dạng test, không phải chất lượng đo của Ignatius; probe tính lại residual từ observations thay vì tin reported error đó.

## Blocker và giới hạn xác minh

Standalone `bundle_adjuster --BundleAdjustment.backend CASPAR` trên fixture không tầm thường bị crash với exit code `3221226505` (`0xC0000409`), chỉ log được dòng bắt đầu global BA. Không có bằng chứng solver GPU hoạt động. Chưa xác định root cause của crash; không kết luận chắc chắn Caspar không được compile chỉ từ lỗi này. Các flags local/global backend có trong help không đủ để cho mapper chạy.

Ceres CUDA chưa được wrapper xác minh theo yêu cầu không CPU fallback; không thử CPU BA. Hướng xử lý: kiểm tra/build COLMAP với Caspar enabled, CUDA/dependencies tương thích và chạy lại standalone probe trước mapper. Xem [COLMAP GPU BA chính thức](https://colmap.github.io/faq.html#speedup-bundle-adjustment). Không thay binary hoặc hạ policy tự động. Thay binary làm cache provenance đổi, cần output experiment mới để tránh nhập nhằng dữ liệu.

Chưa có camera intrinsics/extrinsics hoặc sparse PLY mới cho thí nghiệm video; registered image ratio, point count, track length và reprojection error đều `null`. Các hàm parse/export/quality gate đã được kiểm tra với synthetic fixtures, nhánh mapper/export trên scene thật chưa được xác minh. Overlap selection là heuristic, 317 verified pairs là evidence matching nhưng chưa chứng minh toàn bộ 96 views có thể register. `single_camera=true` giả định zoom/focus cố định, chưa có calibration độc lập.

## Files mới và artifacts

Source/config/test/docs mới; không chỉnh sửa AGENTS, skill cũ hoặc output reconstruction cũ:

```text
.gitignore
README.md
requirements.txt
pytest.ini
configs/sfm_ignatius.yaml
configs/sfm_ignatius_smoke.yaml
scripts/run_sfm.ps1
src/__init__.py
src/pipeline.py
src/common/__init__.py
src/common/config.py
src/common/environment.py
src/common/gpu.py
src/common/io.py
src/common/telemetry.py
src/preprocessing/__init__.py
src/preprocessing/video_probe.py
src/preprocessing/extract_frames.py
src/preprocessing/select_keyframes.py
src/geometry/__init__.py
src/geometry/sfm/__init__.py
src/geometry/sfm/gpu_probe.py
src/geometry/sfm/colmap_runner.py
src/geometry/sfm/model_io.py
src/geometry/sfm/metrics.py
tests/test_gpu_policy.py
tests/test_preprocessing.py
tests/test_sfm_geometry.py
docs/video_sfm.md
docs/tool_versions.md
docs/implementation_report.md
docs/implementation_metrics.json
```

Artifacts chính (được `.gitignore` loại khỏi commit):

- `datasets/tanks_and_temples/Ignatius/preprocessing/extracted_frames/video_v0/manifest.json` và 524 JPG.
- `datasets/tanks_and_temples/Ignatius/preprocessing/selected_frames/video_v0/manifest.json`, CSV và 96 JPG.
- `outputs/tanks_and_temples/Ignatius/sfm_colmap/video_v0/`: `database.db`, `environment.json`, `gpu_report.json`, `metrics.json`, `colmap_state.json`, `commands.json`, `resume_verification.json`, `logs/`, `audits/` và console logs.
- Smoke artifacts ở các đường dẫn tương ứng `video_smoke_v0`. Partial extraction của lần giới hạn thời lượng bị lỗi được giữ nguyên để debug; lỗi đã sửa và có test hồi quy.
- FFmpeg portable ở `.tools/`, version và archive checksum ở `docs/tool_versions.md`; không cài package Python, tải neural model hoặc copy repository bên thứ ba.

Số đo tóm tắt và provenance có thể đọc bằng máy: [implementation_metrics.json](implementation_metrics.json). Hướng dẫn vận hành/cache: [video_sfm.md](video_sfm.md).
