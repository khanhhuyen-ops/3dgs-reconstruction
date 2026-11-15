# Technical stack và backend decisions

Status: thiết kế, chưa cài dependency. [Environment](development_environment.md) là snapshot đọc lại; [reproducibility](reproducibility.md) quy định version locks.

| Layer | Lựa chọn | Lý do / gate |
| --- | --- | --- |
| Orchestration | Python, pathlib, YAML, structured subprocess | Theo code hiện có; Windows paths, stage logs và return codes |
| Dataset adapter | Existing COLMAP TXT parser ở read boundary; adapter mới riêng | Không sửa SfM để phục vụ training; unsupported model phải reject |
| Numerical diagnostics | NumPy CPU/FP64 | Read-only projection/pose checks, không là neural fallback |
| Optimization | PyTorch CUDA FP32, method-native Adam | Gradients và renderer CUDA thực; mixed precision không mặc định |
| B0/B1/B2 | Official implementations qua adapters, pinned revisions/submodules | Giữ method identity; chưa build local |
| Ours | B2 + fixed-anchor module | Isolate one proposed change; không thêm heavyweight prior model |
| B3 | Official GOF branch, optional | Ray-based opacity evaluator và tetrahedral dependencies khác |
| Main meshing | Verified CUDA sparse/block TSDF và extraction backend | Stream views, bounded VRAM; availability phải test bằng operation nhỏ |
| Alternatives | Poisson/oriented points; opacity/tetrahedra | Method-specific; không thay main route để vượt backend gate |
| Geometry metrics | Versioned official dataset evaluator khi có đủ protocol | Match masks/units/alignment; không gọi generic CD là official TNT score |
| RGB metrics | Fixed PSNR/SSIM/LPIPS implementation/version | LPIPS weights phải có source/hash; chưa download |
| Visualization/export | PLY/OBJ + optional Blender/Open3D viewer | Visualization không chứng minh geometry accuracy |

Không mặc định `pip install open3d` trên Windows cho CUDA TSDF. [Open3D tensor integration documentation](https://www.open3d.org/docs/release/tutorial/t_reconstruction_system/integration.html) mô tả CPU/CUDA device; wheel/backend thực phải kiểm tra sau. Legacy CPU `ScalableTSDFVolume` trong các reproduction là reference, không đáp ứng stage cấu hình CUDA. Không có supported CUDA backend → `BLOCKED_BACKEND`; giữ rendered depths.

GOF CGAL Delaunay là tác vụ geometry CPU được công bố trước, không phải neural fallback; tổng pipeline có CPU components. Nếu project cấu hình yêu cầu một extractor cụ thể chạy CUDA thì không thay bằng CPU implementation cùng tên. Poisson reference không được chạy như fallback âm thầm.

B0 chọn faithful original variant, tắt optional depth regularization/anti-aliasing/exposure bổ sung ở official main mới. B2 chọn official-code SHA tại [implementation notes](../research/pgsr_implementation_notes.md). B1/B3 phải pin code khi bắt đầu phase, trước khi tạo experiment; chưa pin được thì không ghi reproducible run. Source/library đặt ngoài `src/` theo versioned dependency boundary, không clone/cài trong phiên tài liệu.

License và Windows build là compatibility gates cho dependency thật, không lý do tạo approval flow cho việc đọc/soạn tài liệu. Heavy dependencies chỉ được triển khai khi user yêu cầu phase lập trình tương ứng.
