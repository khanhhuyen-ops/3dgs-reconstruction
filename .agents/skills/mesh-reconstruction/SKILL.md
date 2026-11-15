---
name: mesh-reconstruction
description: Implement or validate surface extraction, TSDF fusion, opacity-field meshing, mesh cleaning, and PLY or OBJ export from compatible Gaussian outputs. Use for Phase04 or declared extraction variants, with explicit depth and field compatibility.
---

# Surface and mesh reconstruction

## Kích hoạt và context

Đọc AGENTS, [Phase04](../../../docs/plans/phase_04_mesh_extraction.md), [mesh architecture](../../../docs/architecture/mesh_architecture.md), [meshing spec](../../../docs/tech/meshing_specification.md), [evaluation spec](../../../docs/tech/evaluation_specification.md). Required papers: P01 §6.1, P14 §V, P05 §4.5, P07 §3.4, P11 §4.

## Workflow

1. Inspect method/schema/renderer/checkpoint và user execution scope. Gaussian/sparse PLY không phải triangle mesh.
2. Chọn native/common route rõ: B0 native mesh N/A; B0+TSDF diagnostic; B1/B2/Ours depth→TSDF; GOF opacity→MT; MILo cần trained signed scalars/topology.
3. Validate axial/ray depth, normals frame/orientation, K/T, alpha masks/units/normalization. Không fuse inverse-depth hoặc plane distance như axialdepth.
4. Gate CUDA integration/extractor được chỉ định GPU bằng runtime operation; CPU-only Open3D không làm fallback. Giữ rendered depths nếu missing backend.
5. Unload trainer/Adam; stream views/blocks, bound ROI, truncation/voxel cùng units; chunk overlaps/seams có test.
6. Preserve raw mesh; conservative cleaning, report removed area/components. Không largest-component-only mặc định cho whole scene, không tune holes/ROI theo test GT.
7. Export PLY/OBJ có faces và inverse normalization metadata; Blender display transform riêng nếu cần. Chỉ thao tác live Blender khi task yêu cầu, không dùng mesh export task làm lý do sửa scene hiện tại.

## Tools và commands

Read-only parsers/metadata; phase-specific CUDA fusion/mesh validation tests khi đã implement. Native official extractors chỉ dùng với compatible checkpoint/protocol và output mới. Không cài/clone/train trong task tài liệu; no invented CLI stages.

## Validation, lỗi và hoàn thành

Synthetic plane/cube, depth conversion, sign/scale, unknown regions, chunk seams; finite vertices/faces/indices, bounds, normals, reopen/export/import. Failures thường gặp: flippedTSDF, inconsistent units, excessivefiltering, CPU fallback, GOF/MILo field confusion.

Done khi mesh artifacts có thật, quality checks và requested import smoke đã chạy, extraction config/coverage/provenance đủ. Phân biệt mesh usability với quantitative geometry accuracy chưa đo.
