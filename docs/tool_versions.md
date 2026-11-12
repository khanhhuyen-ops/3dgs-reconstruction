# Nguồn và phiên bản công cụ đã dùng

- Python/PyTorch/CUDA và COLMAP được kiểm tra từ môi trường hiện có, không copy repository bên thứ ba vào `src/`. Phiên bản và hash executable thực tế nằm trong `environment.json` của mỗi experiment.
- FFmpeg portable bổ sung tại `.tools/ffmpeg/`, nằm ngoài Git; build `ffmpeg-9.0.2-essentials_build` từ Gyan, một nhà phân phối Windows được trang download FFmpeg giới thiệu.
- URL archive: `https://www.gyan.dev/ffmpeg/builds/ffmpeg-release-essentials.zip`.
- SHA-256 archive: `60f467265b1e312373dbcd92200c2618a74850f98d3d078e94296bb3fa2047ba`, đã đối chiếu bằng `Get-FileHash` với file `.zip.sha256` từ cùng nhà phân phối.
- URL release có thể đổi nội dung theo thời gian; tái lập phải đối chiếu checksum hoặc chọn bản 9.0.2 cố định, không coi URL latest là version pin.
- Không tải model/checkpoint, không cài package Python hoặc build COLMAP trong lần triển khai này.
