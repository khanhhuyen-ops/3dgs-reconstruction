# Video → SfM, GPU-strict

**Mốc tài liệu14/11/2026:** [repository audit](docs/repository_audit.md) giữ trạng thái trước implementation. Trạng thái hiện tại tại [docs/README.md](docs/README.md) và [master plan](docs/plans/master_plan.md). Legacy image_set model khác thí nghiệm video; video V0 vẫn `BLOCKED_GPU_BA`. Các lệnh SfM bên dưới thuộc pipeline cũ; lệnh B0 nằm trong usage Phase02.

**Phase01 (15/11/2026):** đã thêm COLMAP data preparation, frozen splits, train-only initialization và lazy CPU loader. [CLI/config/artifact contract](docs/tech/phase_01_usage.md), [kết quả và acceptance gates lịch sử](docs/phase_01_implementation_report.md). Legacy là experiment riêng được user cho phép; video vẫn thiếu sparse model. Actual native B0 loader/training được kiểm tra tiếp ở Phase02.

**Phase02 B0 (16/11/2026):** low-memory đã hoàn tất30.000 iterations trên CUDA,249.213 Gaussians;33 test views đạt PSNR23,3754 /SSIM0,8399 /LPIPS0,1150. Smoke/resume/native PLY reload đã kiểm chứng; full-resolution dừng `BLOCKED_RESOURCE` tại7401, checkpoint7000; [Kaggle Run All](docs/tech/kaggle_phase02_usage.md) đã upload Private, Linux CUDA build và actual checkpoint restore7000 đã kiểm chứng ([evidence](docs/phase_02_kaggle_publish.json)). [Báo cáo và artifacts](docs/phase_02_implementation_report.md), [setup/training/resume/render/evaluation](docs/tech/b0_training_usage.md), [dependency lock](configs/training/dependency_lock.json). Low-memory là resource variant, không phải paper benchmark; Phase03 chưa bắt đầu.

Pipeline Windows cho `Ignatius.mp4`: probe video → trích frame theo PTS thật → chọn keyframe trên PyTorch CUDA → CUDA SIFT/matching bằng COLMAP → kiểm tra GPU BA → reconstruction/export khi backend đã vượt qua probe.

```powershell
python -m src.pipeline --config configs/sfm_ignatius.yaml --stage preflight --resume
python -m src.pipeline --config configs/sfm_ignatius.yaml --stage preprocess --resume
python -m src.pipeline --config configs/sfm_ignatius.yaml --stage colmap --resume
```

Cần cấu hình `COLMAP_BIN` hoặc `tools.colmap` nếu COLMAP không có trong `PATH`. FFmpeg/ffprobe có thể ở `PATH`, biến `FFMPEG_BIN`/`FFPROBE_BIN`, hoặc `.tools/`. Lần chạy đầu trong thư mục experiment trống có thể bỏ `--resume`. Exit code `2` biểu thị lỗi/blocker; xem JSON và logs để phân biệt.

Hướng dẫn, chính sách cache, giới hạn thực thi và cách xử lý GPU BA: [docs/video_sfm.md](docs/video_sfm.md). Báo cáo lần triển khai: [docs/implementation_report.md](docs/implementation_report.md).

