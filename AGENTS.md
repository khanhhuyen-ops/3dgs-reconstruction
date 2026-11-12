# AGENTS.md — Quy định chung cho dự án 3DGS Mesh Reconstruction

> File hướng dẫn toàn repository cho Codex. Đặt tại **thư mục gốc `3dgs-mesh-reconstruction/`**. Quy định ở đây áp dụng cho mọi module; các skill chuyên biệt bổ sung hướng dẫn chi tiết theo từng tác vụ.

## 1. Mục tiêu nghiên cứu

Xây dựng pipeline **tái tạo cảnh 3D từ video monocular**, hỗ trợ hiển thị cảnh bằng Gaussian Splatting và trích xuất bề mặt mesh:

`Video / RGB frames -> Preprocessing -> SfM / VGGT -> 3DGS / 2DGS -> Depth & Normal -> TSDF / Surface Extraction -> Mesh -> Evaluation`

Thứ tự ưu tiên:
1. Độ chính xác hình học, kết quả có thể kiểm chứng.
2. Khả năng tái lập thí nghiệm, nguồn gốc dữ liệu rõ ràng, so sánh baseline công bằng.
3. Tối ưu VRAM, thời gian chạy và tài nguyên máy cá nhân.
4. Source code dễ đọc, module độc lập, cấu hình rõ ràng.

**Mốc hiện tại: triển khai SfM trước.** Ưu tiên `Ignatius.mp4 -> extracted frames -> selected keyframes -> COLMAP SfM V0 -> camera poses + sparse point cloud`. Chỉ chuyển sang custom SfM V1, VGGT, 2DGS, TSDF khi mốc trước đã được kiểm chứng hoặc khi người dùng yêu cầu đổi phạm vi.

## 2. Môi trường phát triển

- Hệ điều hành: **Windows**, IDE: **VS Code**, shell mặc định: **PowerShell**.
- GPU: **NVIDIA GeForce RTX 3050 Laptop**, khoảng **6 GiB VRAM**, thiết bị CUDA `0`.
- Công cụ dự kiến: Python, PyTorch CUDA khi cần, FFmpeg/ffprobe, COLMAP có CUDA; Open3D dùng xem kết quả nếu phù hợp.
- Không tự giả định công cụ đã được cài hoặc có đúng backend CUDA. Kiểm tra trước bằng lệnh, phiên bản và bài test ngắn.
- Đường dẫn dùng `pathlib.Path`, tính từ repository root; không hardcode ổ đĩa hoặc đường dẫn máy cá nhân.
- Giảm tài nguyên bằng cách xử lý tuần tự từng stage, batching/chunking, cache ra ổ đĩa và giải phóng VRAM sau mỗi bước. Không nạp đồng thời nhiều mô hình lớn trên GPU 6GB.

### Chính sách GPU bắt buộc

- Mặc định `gpu_policy.mode: strict`, `allow_cpu_fallback: false`, GPU target `cuda:0`.
- **Model deep learning, neural inference và các công đoạn tính toán được chỉ định dùng GPU phải thực thi bằng CUDA**; bao gồm GPU SIFT extraction và GPU feature matching của COLMAP khi cấu hình yêu cầu.
- **Không tự fallback CPU** khi CUDA lỗi, thiếu backend hoặc hết VRAM. Báo rõ lỗi, checkpoint đã hoàn thành và cách khắc phục.
- Nếu CUDA OOM: có thể giảm độ phân giải, batch size, số frame, số features/matches; phải log mọi thay đổi và vẫn chạy trên GPU.
- **Không khẳng định toàn bộ SfM đều chạy GPU.** Decode video, I/O, SQLite, orchestration và một số thủ tục hình học có thể dùng CPU. COLMAP thông thường có thể chạy GPU cho extraction/matching nhưng **Bundle Adjustment vẫn ở CPU**.
- Với chế độ GPU-strict cho SfM, chỉ thực thi mapping/BA khi đã **xác minh backend GPU BA đang hoạt động** (ví dụ Caspar/Ceres CUDA được build và chạy thật). Nếu không, ghi `BLOCKED_GPU_BA`, giữ database/features/matches, **không âm thầm chạy CPU mapper**.
- Kiểm chứng runtime từ tensor device, CLI flags và log backend; chỉ thấy GPU trong `nvidia-smi` không đủ để kết luận GPU đã xử lý phép toán.
- CPU được dùng cho điều phối, đọc/ghi dữ liệu và các tác vụ phụ; **không hiểu “GPU-strict” là CPU usage phải bằng 0%**.

## 3. Cấu trúc repository

Giữ nguyên các thành phần đã có. Chỉ tạo thêm thư mục khi thực sự triển khai tính năng tương ứng; không tạo hàng loạt placeholder rỗng.

```text
3dgs-mesh-reconstruction/
├── AGENTS.md                       # quy tắc chung của Codex
├── .agents/skills/                 # các skill Codex theo tác vụ
│   └── video-sfm/
│       ├── SKILL.md
│       ├── plan.md
│       └── tech.md
├── configs/                        # cấu hình versioned YAML/JSON
├── datasets/
│   └── tanks_and_temples/Ignatius/
│       ├── raw/
│       │   ├── Ignatius.mp4
│       │   ├── image_set/
│       │   └── ground_truth/
│       └── preprocessing/
│           ├── extracted_frames/
│           └── selected_frames/
├── src/
│   ├── preprocessing/
│   ├── geometry/sfm/
│   ├── geometry/vggt/               # giai đoạn sau
│   ├── gaussian_splatting/          # giai đoạn sau
│   ├── mesh_reconstruction/         # giai đoạn sau
│   └── evaluation/
├── cache/                           # intermediate files có thể tạo lại
├── outputs/                         # logs, checkpoints, meshes, metrics
├── scripts/                         # CLI / PowerShell entry points
├── tests/
└── docs/                            # kiến trúc, protocol, nghiên cứu
```

Code của thư viện bên thứ ba phải tách biệt và lưu version/commit. Không copy nguyên nhiều repository vào `src/` nếu có thể dùng dependency hoặc wrapper.

## 4. Quy định về dataset

- **Không chỉnh sửa, ghi đè hoặc xóa dữ liệu trong `datasets/**/raw/`.** Raw là nguồn dữ liệu bất biến.
- Thí nghiệm SfM hiện tại mặc định lấy từ `raw/Ignatius.mp4`; **không tự trộn** với `raw/image_set/`. Nếu muốn dùng ảnh có sẵn, phải tạo experiment/config riêng.
- `ground_truth/` mặc định **chỉ dùng cho evaluation**. Không được dùng để huấn luyện, tối ưu pose, chọn frame hoặc điều chỉnh mô hình trừ khi protocol có giám sát được xác định và công bố rõ.
- Với mỗi ảnh được trích xuất/chọn, ghi manifest: `frame_id`, đường dẫn, timestamp thực, video hash, thông số resize/crop, blur/selection score, lý do chọn.
- Khi resize/crop ảnh phải cập nhật intrinsics hoặc lưu phép biến đổi ảnh để cập nhật chính xác sau này.
- Ghi rõ hệ tọa độ, convention camera (world-to-camera/camera-to-world), đơn vị chiều sâu, scale ambiguity và alignment sử dụng trong monocular reconstruction.
- Dữ liệu lớn, cache, checkpoints, outputs không commit vào Git mặc định. Dùng `.gitignore` phù hợp; vẫn lưu config/summary metrics để tái lập.

## 5. Quy trình nghiên cứu và baseline

- Các phương pháp so sánh phải dùng **cùng input frames, resolution, test split, evaluation mask và metric definition**, trừ biến đang được nghiên cứu.
- Chuẩn bị config riêng cho B0, B1, B2, Ours khi đến giai đoạn evaluation. **Không tự đặt định nghĩa cho các baseline**; phải dựa vào protocol được phê duyệt.
- Không dùng ground truth làm input ngầm. Đánh giá novel-view rendering trên ảnh held-out, không báo metric training set như metric test.
- Với SfM, ghi: số ảnh đầu vào, registered images và tỷ lệ, sparse 3D points, track lengths, mean/median reprojection error (px), runtime từng stage, peak VRAM đo được, phiên bản tool và CUDA backend.
- Khi đến giai đoạn rendering: PSNR, SSIM, LPIPS. Khi đến giai đoạn mesh: Chamfer Distance, Precision/Recall/F-score, completeness theo dataset protocol thực tế.
- Phân biệt **kết quả đo thực nghiệm**, số liệu từ bài báo, giả thuyết và giá trị dự kiến. **Không tạo số liệu giả hoặc tuyên bố benchmark thành công khi chưa chạy**.
- Cache theo dataset/scene, danh sách frame, image transforms, config, model checkpoint và code revision. Phải xác minh cache key trước khi tái sử dụng.

## 6. Chuẩn source code và bảo vệ dữ liệu

- Python có type hints, docstrings vừa đủ, logging, exceptions rõ ràng; hàm/module chịu trách nhiệm đơn nhất.
- Phân tách preprocessing, geometry, training, meshing, evaluation, visualization. `src/pipeline.py` chỉ điều phối.
- Ưu tiên `pathlib`, YAML configs, deterministic seeds, `subprocess` có kiểm tra exit code và log stdout/stderr.
- Trước khi thêm dependency nặng, kiểm tra tương thích Windows, Python và CUDA thực tế; tránh download model/checkpoint không cần thiết.
- Không hardcode absolute path, không đưa dữ liệu riêng tư hoặc secrets vào source, commit hay log.
- Không tự thực hiện lệnh phá hủy dữ liệu (`git reset --hard`, `git clean -fd`, xóa `raw/`, xóa hàng loạt outputs) hoặc overwrite experiment cũ nếu chưa được cho phép.
- CLI nên có `--config`, `--stage`, `--dry-run`, `--resume` nếu phù hợp; ghi chính xác lệnh, tham số, version và thời gian chạy.
- Chỉ gắn `SUCCESS` khi output tồn tại, không rỗng, parse được và vượt qua quality checks. Dùng `FAILED`, `SKIPPED`, `BLOCKED_GPU_BA` khi phù hợp.

## 7. Workflow bắt buộc cho Codex

1. **Inspect first:** kiểm tra repo, cấu hình, code và môi trường đã có; không ghi đè file chỉ vì tên giống.
2. **Read task guidance:** khi làm video/SfM, đọc `.agents/skills/video-sfm/SKILL.md`, `plan.md`, `tech.md`. Thư mục `.agent/skills/` cũ (nếu có) có thể chứa ghi chú; không xóa hoặc di chuyển tự động.
3. Xác nhận phạm vi user yêu cầu, lập kế hoạch ngắn theo checkpoint; không tự triển khai stage tương lai.
4. Viết code có thể chạy được theo từng stage nhỏ, lưu cấu hình, log và artifact để debug.
5. Chạy unit tests và smoke tests khi môi trường cho phép. **Không ghi “passed” cho test chưa chạy**.
6. Với GPU-heavy stages, chạy subset nhỏ trước; đo VRAM thực tế rồi mới tăng số frame/resolution.
7. Sau mỗi lần làm việc, báo: file thay đổi, command thực thi, test và kết quả, backend GPU xác minh được, blockers, đề xuất bước kế tiếp.

## 8. Điều kiện hoàn thành mốc SfM hiện tại

- FFmpeg frame extraction và keyframe selection có thể tái lập; có timestamp manifests; `raw/` nguyên vẹn.
- CUDA SIFT extraction và GPU matching chạy thật và được xác minh qua cấu hình/log.
- GPU-strict: chỉ gọi SfM mapper khi GPU Bundle Adjustment backend thực sự khả dụng; nếu không báo blocker minh bạch, **không lén chạy CPU mapper**.
- Khi reconstruction thành công: xuất intrinsics/extrinsics, camera poses, sparse point cloud (TXT/PLY), metrics và logs.
- Không thay toàn bộ custom SfM bằng COLMAP rồi gọi đó là thuật toán tự xây dựng; không coi code skeleton hoặc dummy outputs là hoàn thành.

## 9. Cách giao tiếp

- Giải thích bằng **tiếng Việt**, giữ tên thuật toán, CLI options, biến, class và metrics bằng tiếng Anh chuẩn.
- Nêu rõ giả định kỹ thuật, giới hạn GPU, khả năng tái lập và lỗi còn tồn tại.
- Nếu thiếu CUDA backend: có thể viết code cấu hình/tests không cần GPU, nhưng không thực thi GPU-required model trên CPU để lách quy định.
- Yêu cầu cụ thể của người dùng quyết định phạm vi công việc; tuyệt đối không tự hạ chính sách GPU-strict hoặc làm thay đổi protocol đánh giá.
