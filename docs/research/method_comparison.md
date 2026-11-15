# Method comparison và baseline contract

Nguồn: [P01–P14 review](literature_review.md); mỗi số liệu giữ đúng paper/protocol. B0–B3 là định nghĩa user yêu cầu; các lựa chọn extractor phụ dưới đây là **thiết kế dự án**, chưa triển khai.

**B0 execution16/11/2026:** official2023 SHA414b553 với native CUDA renderer/KNN, RGB-only objective. Final local low-memory config: seed42,480×270,197 train views,76.223 initial points,SH3,30.000 iterations/29.999 Adam calls, native density schedule với declared250k cap; final249.213 Gaussians. Frozen197/33/33 split và transductive legacy source;33 test views PSNR23,3754 /SSIM0,8399 /AlexNet-v0.1 LPIPS0,1150. Full-resolution uncapped profile là run riêng. [Actual report/artifacts/limitations](../phase_02_implementation_report.md), [locked configs](../../configs/training/b0_low_memory.yaml). Đây chưa là comparison group của Phase06 hoặc paper benchmark; B1/B2/B3/Ours và mọi extractor vẫn planned.

| Nhánh | Representation / supervision | Native mesh route | Compatibility | Quyết định |
| --- | --- | --- | --- | --- |
| B0 Original3DGS | Full3D covariance, RGB loss | Không có | Native3DGS checkpoint/renderer | Rendering baseline bắt buộc |
| B0+TSDF diagnostic | B0 không đổi training; depth extraction thêm sau | Declared expected-z → common TSDF | Phải chứng minh depth semantics/gradient nếu cần; không giả là original | Geometry comparison extension riêng |
| B1 2DGS | Planar disks, distortion/normal | Depth → TSDF (P01 §6.1) | Scale dimension2, specialized rasterizer | Baseline bắt buộc |
| B2 PGSR | Flattened3DGS, plane-depth, single/multi-view | Plane-depth → TSDF (P14 §V) | Official variant pinned; own state/app model | Nền geometry chính |
| B3 GOF | 3DGS ray integration, opacity field | Opacity0.5 + tetrahedra/binary search (P07 §3.4) | Own ray renderer/evaluator; B0 conversion không là fullGOF | Optional, backend/resource gate |
| Ours | B2 + fixed SfM confidence anchors | Cùng B2/common TSDF | Cùng B2 SHA, state schema thêm loss/config | Hypothesis, chưa novelty claim |
| DN-Splatter | Sensor/mono depth, normals, flatten/smooth | Oriented backprojection → Poisson (P05 §4.5) | Prior provenance, gsplat stack | Reference, không mandatory baseline |
| 3DGSR | GS + neuralSDF + sparseSfM depth | SDF zero-level Marching Cubes | GS/SDF joint state | Novelty reference |
| SuGaR | Surface-aligned GS; reviewed qua related literature | Density surface samples → Poisson, optional mesh binding | Chưa đọc primary PDF local | Related method, không tuyên bố full review |
| SurfaceSplat | Voxurf↔3DGS cycles, synthetic train views | SDF Marching Cubes | Multi-stage separate models | Background sparse-view |
| MILo | Gaussian pivots + trainable signed scalars | In-loop differentiableMT | Extra fields/topology/cache; 10GB+ paper | Không mandatory trên6GB |
| MeshSplat | Cross-scene feed-forward pixel-aligned2DGS | TSDF | Needs pretrained network/different data budget | Không công bằng nếu gộp per-scene budget |

## Hai cách so sánh mesh

1. **Native extraction**: từng method dùng đúng route native, thông số được lưu. B0 mesh là `N/A`; nếu chạy B0+TSDF phải đặt nhãn riêng. Cách này so toàn hệ thống, bao gồm extractor.
2. **Common TSDF diagnostic**: cùng train frames/cameras/units, voxel/truncation/filter/ROI và backend. B0/B1/B2/Ours xuất depth với semantics ghi rõ. B3 native giữ riêng; chỉ thêm GOF-depth→TSDF như variant nếu có adapter hợp lệ. Cách này giảm confound extractor nhưng không xóa khác biệt depth definition.

Không đổi checkpoint file extension để tạo compatibility. Mỗi artifact khai báo `method`, `schema_version`, `renderer_revision`, activation/SH conventions, normalization và depth definition. Chỉ chuyển đổi khi có adapter được kiểm thử numerical parity.

## Trade-offs quyết định

PGSR có regularizers phù hợp surface task và tận dụng cameras/sparse points hiện có. 2DGS kiểm tra lợi ích representation khác; 3DGS giữ mốc rendering. TSDF có thể mất detail khi voxel lớn, nhưng độc lập với optimizer và dễ kiểm tra trên synthetic planes. GOF giảm phụ thuộc grid cố định nhưng tăng Delaunay/toolchain complexity. MILo cho gradients từ mesh nhưng paper base10GB vượt budget6GB. Task tài liệu14/11 chưa chạy các phương án; implementation B0 sau đó có evidence riêng ở trên, không xác nhận feasibility cho các nhánh khác.

Xem [evaluation specification](../tech/evaluation_specification.md) cho splits, budget, alignment, metrics và failure reporting.
