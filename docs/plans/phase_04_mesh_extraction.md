# Phase 04 — Surface and mesh reconstruction

Trạng thái hiện tại: **PLANNED — chưa triển khai/chạy**. User request tiếp theo quyết định việc thực thi; xem [master plan](master_plan.md). Mọi thông số preset là thiết kế, không benchmark đã đo.

## 1. Phase overview

Chuyển verified Gaussian checkpoints sang triangle meshes. Không sửa SfM; không train thêm model để bù lỗi extraction.

## 2. Research objective

Mesh phản ánh geometry của checkpoint, controls extraction confound và usable export trong Blender.

## 3. Scientific references

P01 §6.1; P14 §V; P05 §4.5/App.C; P07 §3.4; P11 §4; [meshing spec](../tech/meshing_specification.md).

## 4. Current dependencies

At least one B1/B2 checkpoint qualified với renderer depth contract; backend CUDA TSDF/extraction availability. B0+TSDF diagnostic cần explicit depth extension, không nativeB0.

## 5. Input requirements

Tagged checkpoint/render bundle, train-view K/T/depth/normal/alpha, normalization inverse, frozen ROI/voxel/truncation/filter config. No test RGB/GT geometry input.

## 6. Expected outputs

Raw/cleaned triangle mesh PLY/OBJ, extraction/cleaning metadata, bounds/topology/coverage stats, rendered depth bundles, export/import smoke report. Không overwrite sparse PLY hoặc existing experiments.

## 7. Detailed architecture

Compatible renderer → axial depth/normal/valid mask → chunked CUDA TSDF → zero isosurface → conservative cleaning → inverse-normalization/export. Alternative Poisson/GOF adapters riêng; MILo chưa nằm phase implementation này.

## 8. Mathematical requirements

Projective signed-distance fusion, weights/truncation, unknown voxel mask, unit consistency, camera-to-world back projection. GOF opacity 0.5 field/MT binary search và MILo signed scalars được phân biệt trong spec.

## 9. Implementation tasks

Implement depth export adapter, filter stats, compute ROI/memory budget, verified CUDA fusion/extraction, chunk overlap/seam handling, raw mesh validation, declared cleaning, normal correction, PLY/OBJ roundtrip; Blender compatibility smoke khi được thực thi. Native/common extraction tracks separate.

## 10. Ordered execution steps

1) Validate buffers/conventions. 2) Synthetic plane/cube CUDA fusion. 3) Verify chunked vs unchunked fixture. 4) Small real subset render/fuse. 5) Check faces/bounds/normals/coverage. 6) Export/import. 7) Freeze config trước final experiments. Missing backend giữ depths và report blocker.

## 11. Technical dependencies

CUDA sparse/block TSDF+isosurface candidate backend; Open 3D build phải có runtime CUDA operation. Poisson reference cần oriented points/backends riêng; no implicit CPU fallback. GOF CGAL branch thuộc optional Phase 06.

## 12. Hardware and resource constraints

Unload training/Adam trước meshing. One view/chunk, precompute grid dimensions; development voxel 0.02/research 0.005 normalized units và truncation 5voxels là presets đề xuất. Sparse allocation vẫn có overhead; no guarantee 6GB.

## 13. Validation strategy

Synthetic depth known surface trong 1–2voxels tùy fixture; axial/ray conversion, invalid/unknown regions, extrinsic flip tests; seams, duplicate vertices, degenerates; meshreopen và Blender coordinate/import check.

## 14. Evaluation metrics

Faces/vertices/components/boundary/nonmanifold, covered area và depth residual distributions, extraction time/VRAM/filebytes. GTCD/F1 chưa thuộc phase quality claim; raw/clean stats riêng.

## 15. Risks and mitigation

CPU-only wheel, reversed TSDF sign, wrongunits, largest component xóa objects, over-filtering tăngaccuracy giả, chunkseams. Mitigate backend gates/synthetic fixtures/conservative cleaning/coverage logging.

## 16. Acceptance criteria

Real triangle faces>0, finite geometry, valid indices/bounds, metadata/source hashes, surfaces match synthetic fixture và input depth sanity. Missing CUDA backend không thể báo SUCCESS bằng CPU mesh.

## 17. Definition of Done

Artifacts parse/reopen, meaningful validationđã chạy, native/common route labels rõ; unknown units không ghi meters; Blender smoke chưa chạy thì ghi pending, không xác nhận compatibilitythực tế.

## 18. Next phase dependencies

Phase 05 geometry monitoring dùng same extract or; Phase 06 produce comparison meshes; Phase 07 evaluator chỉ nhận frozen mesh config, no test-drivencleaning.

