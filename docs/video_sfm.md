# Vận hành pipeline video-to-SfM

## Phạm vi và hướng dẫn nguồn

Triển khai Phase 0–3 V0 và phần kiểm thử/evaluation độc lập của Phase 5. Custom incremental SfM V1 chỉ triển khai sau khi V0 từ video được kiểm chứng bằng reconstruction GPU-strict. Chưa triển khai VGGT, Gaussian Splatting, TSDF hoặc mesh.

Đã đọc `AGENTS.md` và bộ skill thực tế tại `.agent/skills/preprocessing/{SKILL-video-sfm.md,plan-video-sfm.md,tech-video-sfm.md}`. Bộ `.agents/skills/video-sfm/` được nhắc trong yêu cầu chưa có trên workspace; giữ nguyên các tài liệu cũ.

## Môi trường và lệnh

Python cần `torch` có CUDA, `PyYAML`, `Pillow`, `numpy`; `pytest` dùng cho kiểm thử. Các package đã có trong môi trường triển khai, không cài lại PyTorch. `requirements.txt` liệt kê phụ thuộc tối thiểu; wheel CUDA của PyTorch phải được chọn riêng theo Windows/Python/driver thực tế. Không tự cài torch CPU để lách kiểm tra.

FFmpeg và ffprobe phải cùng build. COLMAP CLI cần CUDA SIFT/matching; đặt `COLMAP_BIN` trỏ đến `colmap.exe` khi chưa có trong PATH. Nếu nhận `COLMAP.bat`, resolver dùng `bin/colmap.exe` cạnh wrapper để giữ argument list an toàn trên Windows. Các batch wrapper bố trí khác phải chỉ rõ executable; không ghép chuỗi shell từ đường dẫn người dùng.

```powershell
# Chỉ mô tả cấu hình, không tạo file hoặc chạy probe.
python -m src.pipeline --config configs/sfm_ignatius.yaml --stage all --dry-run

# Audit tạo environment.json, gpu_report.json và lịch sử audits/.
python -m src.pipeline --config configs/sfm_ignatius.yaml --stage preflight --resume

# Trích frame và chọn keyframe; tiếp tục được dù GPU BA bị block.
python -m src.pipeline --config configs/sfm_ignatius.yaml --stage preprocess --resume

# Chỉ dùng selected_frames đã kiểm tra hash/provenance.
python -m src.pipeline --config configs/sfm_ignatius.yaml --stage colmap --resume

# Kiểm tra subset 24 giây, 640px, mục tiêu 12 keyframe trước full video.
python -m src.pipeline --config configs/sfm_ignatius_smoke.yaml --stage all --resume

# Có PowerShell wrapper, trả nguyên exit code của Python.
.\scripts\run_sfm.ps1 -Config configs/sfm_ignatius.yaml -Stage colmap -Resume

python -m pytest -q
python -m ruff check src tests
```

`ruff` là công cụ kiểm tra tùy chọn đã có tại môi trường triển khai, không phải dependency runtime. Exit `0`: stage được yêu cầu thành công hoặc `DRY_RUN`; exit `2`: `BLOCKED`, `BLOCKED_GPU_BA`, `STOP_*` hoặc `FAILED`. `preflight` có thể trả `BLOCKED` khi preprocessing vẫn khả dụng. `--stage custom` trả `BLOCKED_V0_PREREQUISITE`, không gọi COLMAP dưới tên custom.

## Modules và artifacts

| Module | Trách nhiệm |
| --- | --- |
| `src/common/config.py` | YAML, chính sách strict, đường dẫn nằm trong root, bảo vệ raw/input |
| `src/common/environment.py`, `gpu.py` | phiên bản, input metadata, tensor CUDA thật, probe GPU BA |
| `src/common/io.py`, `telemetry.py` | hash, atomic JSON, subprocess logs/exit code, VRAM lấy mẫu |
| `src/preprocessing/video_probe.py` | codec, duration, FPS, time base, rotation, frame count |
| `src/preprocessing/extract_frames.py` | FFmpeg sampling bằng source PTS, manifest, staging directory |
| `src/preprocessing/select_keyframes.py` | blur/exposure/similarity trên CUDA, temporal bins và bridge |
| `src/geometry/sfm/gpu_probe.py` | CLI capabilities và standalone BA probe tổng hợp |
| `src/geometry/sfm/colmap_runner.py` | CUDA SIFT, matcher và verification, gate mapper, export và checkpoints |
| `src/geometry/sfm/model_io.py`, `metrics.py` | parse model TXT, pose conventions, residuals/distortion và PLY validation |
| `src/pipeline.py` | orchestration theo stage |

Cấu hình mặc định dùng subdirectory `video_v0` bên dưới extracted/selected/output để không ghi đè output `sfm_colmap` đã có. Cấu hình smoke dùng `video_smoke_v0`. Chỉ video được dùng làm input; không trộn `raw/image_set` và không đọc ground truth.

Manifest mỗi frame chứa `frame_id`, path, PTS integer/time base, timestamp, video SHA-256, image checksum, kích thước, resize/crop/orientation metadata, score và lý do. `select` của FFmpeg giữ PTS gốc, không nội suy `index/FPS` và không dùng codec I-frame để định nghĩa keyframe. Manifest lưu phép scale từ ảnh đã autorotate đến ảnh đầu ra; khi dùng calibration từ raw cần áp dụng orientation trước scale. Intrinsics trong model COLMAP được fit trực tiếp trên ảnh đầu ra. Rotation ngoài bội số 90° bị từ chối. Decode/resize chạy CPU theo chính sách cho phép.

Selection decode ảnh ở CPU rồi chuyển thumbnail lên `cuda:0`, dùng FP32 convolution/reductions và similarity. Số ảnh trên GPU bị giới hạn bởi batch. So sánh duplicate với các frame đã chọn cũng trên CUDA; descriptors nhỏ được chuyển về CPU để tránh giữ toàn bộ ảnh trên VRAM. Nếu quality heuristic không đủ tin cậy, chuyển thuật toán sang chọn đều theo thời gian, vẫn tính score/similarity trên CUDA. Không nhân bản ảnh để đạt min frame count. Khoảng thời gian và appearance chỉ là heuristic overlap; chỉ matching/geometry mới kiểm tra được overlap thực.

SIFT extraction/matching sinh `database.db`; kiểm tra đúng image names, dữ liệu keypoint/descriptor, verified pair count và signatures. Matcher COLMAP thực hiện geometric verification, có thể dùng CPU cho phần hình học được cho phép. Không khẳng định mọi phép tính SfM chạy GPU.

## Gate GPU BA và giới hạn

Trước mapper, chạy standalone Caspar BA trên fixture tổng hợp 3 camera/48 điểm có nhiễu XYZ và observations từ geometry đã biết. Fixture nằm trong `audits/.../ba_probe_*/synthetic_input/`, được đánh dấu `DIAGNOSTIC_ONLY.json`; không phải output Ignatius hoặc benchmark. Probe chỉ pass khi command thành công, có solver log phù hợp, model parse được và reprojection error thực giảm. Hash executable ràng buộc probe với binary mapper hiện tại.

Wrapper hiện chỉ cho phép **CASPAR GPU-exclusive** với `SIMPLE_RADIAL` hoặc `PINHOLE`; cả local/global BA đều phải chọn backend này. Ceres CUDA chưa được hỗ trợ để bật mapper: các generic flags/solver thresholds có thể dẫn đến CPU fallback, nên báo chưa xác minh và giữ gate đóng. Không suy luận GPU BA từ `mapper -h` hoặc GPU utilization.

Nếu probe lỗi/crash/không đủ evidence: giữ features/matches, ghi `BLOCKED_GPU_BA`, không tạo sparse model giả, không gọi mapper. Cần build COLMAP tương thích với `-DCASPAR_ENABLED=ON`, CUDA và camera model hỗ trợ, rồi chạy lại probe. Không đổi GPU policy sang practical hoặc bật CPU BA. Thay binary/config/code làm cache key đổi; dùng experiment output mới để giữ provenance, giữ nguyên DB cũ để phục hồi/migrate có kiểm tra về sau.

Khi mapper hợp lệ: kiểm tra các model con thực sự tồn tại, chọn model hợp lệ có nhiều registered images nhất, chuyển TXT/PLY, xuất camera poses JSON, đọc lại PLY và tính residuals từ projections với distortion. Kiểm tra finite poses, baseline khác 0, positive depth, reciprocal tracks, >=2 registered images, >0 points và mean reprojection error <=4px (có thể đặt `sfm.max_mean_reprojection_error_px` trong experiment riêng). Không đánh giá novel-view hoặc chất lượng mesh ở stage này. Nhánh mapping/export chưa có validation thực tế trên Ignatius khi GPU BA bị block.

Extrinsics COLMAP là world-to-camera: `x_cam = R X_world + t`; tâm camera `C = -R^T t`. JSON xuất cả world-to-camera và camera-to-world. Hệ trục camera x sang phải, y xuống dưới, z ra trước. Monocular có global scale ambiguity, không suy ra đơn vị mét hoặc tự align bằng ground truth. `single_camera: true` là giả định cùng camera, không đổi zoom/focus; chưa có kiểm chứng calibration độc lập cho video.

## Cache, resume và bảo vệ dữ liệu

Cache key bao gồm video content hash, preprocessing config, transforms/frame list, tool version/binary hash và source revision hash theo stage; seed được lưu. Chỉ reuse khi key khớp và checksum/path/timestamp/dimensions hợp lệ. Database resume kiểm tra signatures của feature blobs và verified geometry. Output mapping đã ghi checkpoint phải có đúng artifact hashes và vượt qua validation lại.

`--resume` không xóa file hỏng hoặc ghi đè experiment có config/key khác. Khi hỏng/thiếu file, báo lỗi và yêu cầu đường dẫn experiment mới; dữ liệu cũ vẫn giữ nguyên. Extraction/selection ghi vào `.partial-*`, kiểm tra rồi rename; khi lỗi vẫn giữ thư mục tạm cùng `failure.json` để debug. Không tự xóa các partial cũ. JSON reports của experiment do pipeline quản lý có thể cập nhật, audits/failures và subprocess logs được giữ theo từng lần chạy. `--dry-run` không gọi subprocess hay tạo thư mục.

Stage SUCCESS là trạng thái cục bộ: preprocessing hoặc features SUCCESS không có nghĩa reconstruction SUCCESS. Metrics chưa có reconstruction để `null`. Peak tensor allocation/reservation dùng số đo PyTorch; VRAM của COLMAP là **device-wide sampled memory** mỗi 0,5 giây, có thể bỏ lỡ peak ngắn và gồm process khác, không được gọi là exact per-process peak.

## Kiểm thử

Tests dùng video tổng hợp VFR thật qua FFmpeg: đối chiếu emitted PTS với source frame PTS, không upscale, bounded duration, giữ source hash, resume/key mismatch/corruption. CUDA thumbnail selection chạy thật khi CUDA sẵn có, kiểm tra lựa chọn lặp lại cùng frame IDs. Kiểm tra hide GPU bằng process riêng với `CUDA_VISIBLE_DEVICES=-1`, fail-fast và không CPU fallback. Các bài test control flow mock subprocess để chứng minh mapper không được gọi khi BA blocked; chúng không phải bằng chứng runtime GPU. Geometry tests dùng synthetic poses/tracks để kiểm tra camera centers, residuals, negative depth, track errors và PLY hỏng.

Tài liệu chính thức: [COLMAP GPU BA / Caspar](https://colmap.github.io/faq.html#speedup-bundle-adjustment), [COLMAP CLI](https://colmap.github.io/cli.html), [FFmpeg filters](https://ffmpeg.org/ffmpeg-filters.html). Mọi CLI flag thực thi đều dựa trên `-h` của binary đang chạy.

