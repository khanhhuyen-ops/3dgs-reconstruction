# Development environment

**PGSR runtime đã kiểm chứng17/11/2026:** `.tools/pgsr_env` riêng, Python3.10.20/Torch2.4.1+cu121/Toolkit12.1/MSVC14.29/sm86, PGSR plane rasterizer + native KNN forward/backward trên RTX3050. PyTorch3D0.7.7 official pure-Torch transforms chạy CUDA; unused C++ extension không build. B0/SfM environments giữ nguyên. [Lock](../../configs/training/pgsr_dependency_lock.json), [usage và evidence](pgsr_training_usage.md). Bảng14/11 bên dưới là snapshot lịch sử, không mô tả interpreter training mới. Kaggle environment còn phải qua chính CUDA gates tại runtime.

Snapshot read-only14/11/2026 tại interpreter hiện tại; không chạy CUDA tensor/benchmark.

| Thành phần | Quan sát | Ý nghĩa |
| --- | --- | --- |
| OS/shell | Windows, PowerShell | Paths/subprocess phải Windows-safe |
| Python | 3.14.6 | Chưa xác minh official extensions hỗ trợ |
| torch | 2.11.0+cu128 | Package CUDA runtime12.8, không chứng minh compiler12.8 |
| nvcc | Toolkit12.1, V12.1.66 | Khác torch runtime; extension build cần compatibility audit |
| NumPy/Pillow/YAML | 2.4.6 /12.1.1 /6.0.3 | Existing, không cài lại |
| PyMuPDF | 1.27.2.2 | Đã dùng đọc PDF, không là training dependency |
| Open3D/Nerfstudio | Không tìm thấy package metadata | Không giả định loader/mesher từ các stack này đã có |
| cl/cmake | Không trong PATH | Có thể có installation ở nơi khác; chưa kiểm tra VS developer shell |
| COLMAP/FFmpeg | Không trong PATH | Resolver/tool logs chỉ ra executable riêng; không làm mất evidence vì PATH |
| GPU target | RTX3050 laptop khoảng6GiB, cuda:0 theo project | Logs10/11 chứng minh CUDA extraction/matching; chưa retest runtime hôm nay |

## Environment chiến lược cho implementation tương lai

Tạo **environment riêng** khi phase build được yêu cầu; không thay interpreter đang chạy SfM. Candidate stack khởi đầu là Python3.10 hoặc3.11, phiên bản torch/CUDA toolkit theo official backend đã pin và driver thực. Đây là lựa chọn nghiên cứu build, **chưa là compatibility đã xác nhận**. Không đưa lệnh cài đặt không có version lock hoặc cài CPU torch để giải quyết lỗi CUDA.

Preflight tương lai phải ghi Python/torch/CUDA runtime/toolkit/driver, Visual Studio C++ toolset, compiler flags và GPU compute capability; kiểm tra executable/source hashes. Một custom extension cần forward/backward CUDA smoke và tensor-device assertions. Presence trong `nvidia-smi`, `nvcc` hoặc CLI help không đủ chứng minh đã xử lý operation bằng GPU.

COLMAP BA blocker hiện tại không được sửa trong task này. Khi user yêu cầu xử lý: chỉ dùng existing workflow [video_sfm](../video_sfm.md); nontrivial CASPAR runtime probe phải pass trước mapper. Đọc [audit](../repository_audit.md) để tránh lấy synthetic diagnostic models làm scene input.

## Commands và giới hạn phiên tài liệu

Đã chạy `python --version`, `nvcc --version`, metadata lookup, `git status`, parse TXT/PLY và image-header checks. Không chạy `src.pipeline --stage preflight` vì audit đó thực thi CUDA/BA probes và ghi output. Không chạy `pytest` toàn repo vì có tests dùng CUDA thực. Tests nguồn cũ không đổi; kiểm tra hiện tại chỉ documentation links, schema examples và skill metadata.

Các câu lệnh build/training/meshing tương lai chưa tồn tại như một project CLI thống nhất. Agent phải triển khai CLI ở phase được yêu cầu, không chạy lệnh tưởng tượng như `--stage train` vào `src.pipeline` cũ.
