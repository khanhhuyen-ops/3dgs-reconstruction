# Video → SfM, GPU-strict

Pipeline Windows cho `Ignatius.mp4`: probe video → trích frame theo PTS thật → chọn keyframe trên PyTorch CUDA → CUDA SIFT/matching bằng COLMAP → kiểm tra GPU BA → reconstruction/export khi backend đã vượt qua probe.

```powershell
python -m src.pipeline --config configs/sfm_ignatius.yaml --stage preflight --resume
python -m src.pipeline --config configs/sfm_ignatius.yaml --stage preprocess --resume
python -m src.pipeline --config configs/sfm_ignatius.yaml --stage colmap --resume
```

Cần cấu hình `COLMAP_BIN` hoặc `tools.colmap` nếu COLMAP không có trong `PATH`. FFmpeg/ffprobe có thể ở `PATH`, biến `FFMPEG_BIN`/`FFPROBE_BIN`, hoặc `.tools/`. Lần chạy đầu trong thư mục experiment trống có thể bỏ `--resume`. Exit code `2` biểu thị lỗi/blocker; xem JSON và logs để phân biệt.

Hướng dẫn, chính sách cache, giới hạn thực thi và cách xử lý GPU BA: [docs/video_sfm.md](docs/video_sfm.md). Báo cáo lần triển khai: [docs/implementation_report.md](docs/implementation_report.md).

