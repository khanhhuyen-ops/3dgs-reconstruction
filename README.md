# Video → SfM, GPU-strict

**Research documentation-first (14/11/2026):** bắt đầu tại [docs/README.md](docs/README.md), [repository audit](docs/repository_audit.md) và [master plan](docs/plans/master_plan.md). Bộ tài liệu hậu COLMAP đã thiết kế architecture/specs/7 phases/skills; chưa triển khai hoặc train Gaussian model. Legacy image_set model khác thí nghiệm video; video V0 vẫn `BLOCKED_GPU_BA`. Các lệnh bên dưới là CLI SfM hiện có, không phải lệnh training mới.

**Phase01 (15/11/2026):** đã thêm COLMAP data preparation, frozen splits, train-only initialization và lazy CPU loader. [CLI/config/artifact contract](docs/tech/phase_01_usage.md), [kết quả và acceptance gates](docs/phase_01_implementation_report.md). Legacy là experiment riêng được user cho phép; video vẫn thiếu sparse model. Chưa chạy training hoặc full native Gaussian scene loaders.

Pipeline Windows cho `Ignatius.mp4`: probe video → trích frame theo PTS thật → chọn keyframe trên PyTorch CUDA → CUDA SIFT/matching bằng COLMAP → kiểm tra GPU BA → reconstruction/export khi backend đã vượt qua probe.

```powershell
python -m src.pipeline --config configs/sfm_ignatius.yaml --stage preflight --resume
python -m src.pipeline --config configs/sfm_ignatius.yaml --stage preprocess --resume
python -m src.pipeline --config configs/sfm_ignatius.yaml --stage colmap --resume
```

Cần cấu hình `COLMAP_BIN` hoặc `tools.colmap` nếu COLMAP không có trong `PATH`. FFmpeg/ffprobe có thể ở `PATH`, biến `FFMPEG_BIN`/`FFPROBE_BIN`, hoặc `.tools/`. Lần chạy đầu trong thư mục experiment trống có thể bỏ `--resume`. Exit code `2` biểu thị lỗi/blocker; xem JSON và logs để phân biệt.

Hướng dẫn, chính sách cache, giới hạn thực thi và cách xử lý GPU BA: [docs/video_sfm.md](docs/video_sfm.md). Báo cáo lần triển khai: [docs/implementation_report.md](docs/implementation_report.md).

