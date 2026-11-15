---
name: video-sfm
description: Build, test, and debug a reproducible Windows video-to-incremental-SfM pipeline for the Ignatius Tanks and Temples scene. Extract and select frames, run CUDA-only feature extraction and matching, require verified GPU bundle adjustment for strict COLMAP reconstruction, and implement a research-oriented custom SfM module. Use for video preprocessing, camera pose recovery, sparse point clouds, SfM baselines, and GPU enforcement in this repository.
---

# Video → Incremental SfM (GPU-strict)

## 1. Nhiệm vụ

Bạn là Codex, kỹ sư Computer Vision chịu trách nhiệm **viết và kiểm thử mã nguồn thật** cho dự án `3dgs-mesh-reconstruction`. Triển khai pipeline:

`Ignatius.mp4 → extracted_frames → selected_frames → SIFT/matching → Incremental SfM → camera poses + sparse point cloud + metrics`.

Giai đoạn này **không triển khai VGGT, 2DGS, TSDF, mesh reconstruction hoặc huấn luyện neural network**. Có hai implementation riêng:

- `colmap`: bản chuẩn (baseline V0), gọi COLMAP qua CLI có kiểm tra CUDA.
- `custom`: bản phục vụ nghiên cứu (V1), tự viết các phép toán SfM quan trọng; không thay toàn bộ bằng lệnh gọi COLMAP. V1 triển khai sau khi V0 và preprocessing được kiểm chứng.

Đọc `plan.md` để biết thứ tự việc làm và tiêu chí hoàn thành; đọc `tech.md` trước khi chọn thuật toán, tham số hoặc câu lệnh.

## 2. Đường dẫn dự án (không tự ý thay raw)

- Gốc: `3dgs-mesh-reconstruction/` (Windows, PowerShell, VS Code).
- Video: `datasets/tanks_and_temples/Ignatius/raw/Ignatius.mp4`.
- Ảnh tham khảo có sẵn: `datasets/tanks_and_temples/Ignatius/raw/image_set/`.
- Ground truth: `datasets/tanks_and_temples/Ignatius/raw/ground_truth/` — **chỉ được dùng để đánh giá khi được yêu cầu**, không dùng để tối ưu SfM.
- Frame trích xuất: `datasets/tanks_and_temples/Ignatius/preprocessing/extracted_frames/`.
- Frame được chọn: `datasets/tanks_and_temples/Ignatius/preprocessing/selected_frames/`.
- Kết quả: `outputs/tanks_and_temples/Ignatius/sfm_colmap/` và `outputs/tanks_and_temples/Ignatius/sfm_custom/`.
- Cache có dấu vết phiên bản: `cache/sfm/`.

**Dùng video làm nguồn mặc định**, không trộn ảnh từ `raw/image_set/` vào pipeline video. Không sửa/xóa `raw/` hoặc ground truth. Nếu file/đường dẫn khác với dự kiến: dò file, báo lại đường dẫn thực, không tự bịa dữ liệu.

## 3. Chính sách phần cứng bắt buộc

- GPU mục tiêu: NVIDIA GeForce RTX 3050 laptop, khoảng **6 GB VRAM**, `cuda:0`.
- Mặc định `gpu_policy.mode: strict`, `allow_cpu_fallback: false`.
- Kiểm tra `nvidia-smi`, CUDA runtime, `torch.cuda.is_available()` và khả năng thực thi GPU của COLMAP **trước** các bước có tính toán CUDA.
- Mọi mô hình deep learning, tensor computation quan trọng, SIFT GPU và feature matching GPU **phải thực thi trên GPU**; thiếu CUDA thì dừng với thông báo hành động cần làm.
- Trong **strict end-to-end COLMAP**, cả bundle adjustment (BA) phải chạy backend GPU đã kiểm chứng (Caspar hoặc Ceres CUDA build phù hợp). **Không tự động chạy mapper/BA CPU**.
- Chỉ được dùng CPU cho hệ điều hành, Python orchestration, SQLite, đọc/ghi file, giải mã video/metadata hoặc các tác vụ không có backend GPU phù hợp, và phải ghi rõ giới hạn. Quy tắc này không đồng nghĩa "CPU usage = 0%".
- **Không đổi `strict` sang `practical`, không tắt `use_gpu`, không giảm chính sách khi gặp lỗi.** Nếu build COLMAP không hỗ trợ GPU BA: có thể hoàn thành preprocessing + GPU feature extraction/matching; đánh dấu mapper `BLOCKED_GPU_BA`, giải thích yêu cầu build, không báo reconstruction đã thành công.
- Nếu phát hiện CUDA out-of-memory: giải phóng tài nguyên, giảm `max_image_size`, số frame, số feature/match, hoặc batch size rồi chạy lại GPU. Không fallback CPU.
- Không tuyên bố tác vụ chạy GPU chỉ dựa vào `nvidia-smi` có tên GPU: cần tham số CLI, logs/backend hoặc thông tin tensor device phù hợp.

## 4. Quy trình thực hiện cho Codex

1. Khảo sát repository trước khi tạo file: tránh ghi đè mã, config hoặc output đã tồn tại.
2. Đọc `plan.md` và `tech.md`; tạo checklist công việc theo phase, mỗi phase có output kiểm chứng.
3. Preflight: kiểm tra FFmpeg/ffprobe, COLMAP/COLMAP.bat, Python, PyTorch CUDA, GPU memory; lưu báo cáo môi trường và phiên bản.
4. Viết module `src/preprocessing/` để probe video, trích xuất frame **giữ thứ tự/timestamp**, chọn frame theo thời gian, chất lượng và độ chồng lấp; lưu manifest.
5. Viết `src/geometry/sfm/colmap_runner.py`: gọi GPU SIFT + GPU matcher; xác định mapper GPU BA nào sẵn có bằng capabilities thực tế; không truyền các flags không được version hiện tại hỗ trợ.
6. Nếu BA GPU chưa khả dụng, dừng ở mốc hợp lệ, giải thích block và chỉ dẫn bật Caspar/Ceres CUDA. **Không lén dùng COLMAP CPU mapper.**
7. Khi đã có COLMAP GPU-BA hợp lệ, xuất đúng model, TXT, PLY và báo cáo metric. Không mặc định thư mục model số `0` luôn tồn tại.
8. Tạo khung và triển khai từng phần SfM V1 theo `plan.md`; test toán học / GPU với dữ liệu nhỏ trước khi tích hợp thực tế.
9. Tạo CLI dễ chạy bằng PowerShell; có `--dry-run`, `--stage`, `--resume`, `--config` và log lỗi rõ ràng. Mọi subprocess phải kiểm tra `returncode`.
10. Viết unit tests + smoke tests; chạy được những test không phụ thuộc dữ liệu/GPU trong môi trường phù hợp và **báo thực tế** phần nào chưa thể chạy.

## 5. Kiến trúc mục tiêu (chỉ tạo khi cần)

```text
configs/sfm_ignatius.yaml
src/
  __init__.py
  pipeline.py
  common/gpu.py
  preprocessing/
    __init__.py
    video_probe.py
    extract_frames.py
    select_keyframes.py
  geometry/sfm/
    __init__.py
    colmap_runner.py
    model_io.py
    metrics.py
    custom/
      features.py
      matching.py
      two_view.py
      triangulation.py
      registration.py
      bundle_adjustment.py
      reconstruction.py
scripts/run_sfm.ps1
tests/test_preprocessing.py
tests/test_sfm_geometry.py
tests/test_gpu_policy.py
```

`src/pipeline.py` điều phối các module, không dồn toàn bộ code vào một file. Hạn chế thêm dependency nặng. Dùng `pathlib`, cấu hình YAML, `subprocess` an toàn; hỗ trợ Windows path có dấu cách.

## 6. Đầu ra bắt buộc

- `selected_frames/` có danh sách ảnh và `manifest.json` / `manifest.csv` (timestamp, source, score, lý do chọn).
- `outputs/.../sfm_colmap/` chứa database, sparse reconstruction, TXT/PLY nếu mapper thành công, logs, metrics, `environment.json`, `gpu_report.json`.
- `outputs/.../sfm_custom/` chứa pose/point-cloud và metrics tương đương khi V1 chạy được.
- `README` hoặc usage trong module: lệnh PowerShell dùng chính xác, các phụ thuộc, VRAM requirements, cách resume.
- Không ghi success cho output rỗng, các pose không hợp lệ hoặc mapper bị block.

## 7. Acceptance / kiểm chứng

- Số frame được chọn nằm trong giới hạn config (mặc định mục tiêu 80–120, không vượt `max_selected_frames`). Frame không trùng, thứ tự/timestamp đơn điệu, có độ chồng lấp hợp lý.
- Chạy lại cùng config, video hash và seed cho cùng manifest (deterministic), trừ khi FFmpeg version/decoder không tương thích; nếu thế ghi lại khác biệt.
- Khi CUDA không sẵn có: **fail fast**, không có CPU fallback ở bước bắt buộc GPU.
- Khi có feature/matching GPU mà không có GPU BA: báo trạng thái `BLOCKED_GPU_BA`, giữ lại dữ liệu trung gian.
- Khi reconstruction thành công: số camera đã đăng ký > 1, số điểm 3D > 0, pose hữu hạn, phép chiếu ngược có giá trị thống kê, file PLY đọc được.
- In ra summary: số ảnh, ảnh đăng ký, tỷ lệ đăng ký, số điểm 3D, reprojection error, runtime theo stage, GPU backend và peak VRAM nếu đo được.

## 8. Cách báo cáo sau mỗi phase

Trả lời bằng tiếng Việt, ngắn gọn và chính xác:
- Đã tạo/sửa file nào.
- Đã chạy command nào, PASS/FAIL/BLOCKED.
- GPU backend nào đã thực sự được dùng.
- Lỗi và bước xử lý tiếp theo. Không giả định đã chạy trên máy người dùng khi không có quyền truy cập môi trường đó.

> Nếu không thể tuân thủ chính sách GPU-only cho một bước, **dừng và giải thích giới hạn**, không lén đổi yêu cầu.
