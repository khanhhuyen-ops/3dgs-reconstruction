# Meshing specification

Thiết kế, chưa có implementation. Sources: P01 §6.1; P14 §V; P05 §4.5/App.C; P07 §3.4; P11 §4. [Architecture](../architecture/mesh_architecture.md), [depth math](../research/mathematical_foundations.md) và [evaluation protocol](evaluation_specification.md).

## 1. Input contract và filtering

Nhận tagged checkpoint qua compatible renderer hoặc pre-render bundle có hashes. Main TSDF buffers: axial depthZ `[H,W]float32`, RGB `[H,W,3]`, normals camera/world label, alpha `[H,W]`, booleanvalid, PINHOLE K, Tcw. Ray-distance `t` đổi sang axial `Z=t*r_z` khi rayunit; không đưa normalized-inverse-depth hoặc blended distance trực tiếp vào TSDF.

Render **train cameras** cho extraction; không lấy test RGB/depth làm input. Valid nếu finite, Z>0, covered và trong depth range frozen. Proposed common-filter alpha≥0.5, znear0.001 normalized units, zfar1.2×99.5th percentile positive train-point observeddepth; record masked fractions. Thresholds là design/sweep onvalidation, không paper defaults. Paper/native extraction track có thể khác, phải label riêng.

Optional neighbor depth/normal consistency filter chỉ từ train neighbors và với same config cho B0diagnostic/B1/B2/Ours; log accepted fraction để không tăng accuracy bằng bỏ gần hết surface. V0 common route dùng alpha/depth filter trước, thêm consistency là ablation riêng. Sky/GT evaluation masks không trở thành hidden training/meshing masks.

## 2. Main route: TSDF và isosurface

Backend yêu cầu CUDA runtime verification cả integration và extractor operation được cấu hình GPU. [Open3D tensorTSDF reference](https://www.open3d.org/docs/release/tutorial/t_reconstruction_system/integration.html) là candidate, chưa có package/build local; CPU-only wheel → `BLOCKED_BACKEND`. Legacy CPU Open3D trong paper không dùng làm fallback.

Common config **đề xuất** trong normalized sceneunits: development voxel0.02/truncation0.10; research-quality voxel0.005/truncation0.025. Native2DGS paper voxel0.004/truncation0.02 không mặc định cùng unit/normalization. Cùng commonconfig cho mọi compared method, freeze trước test.

ROI từ robust1st–99th percentile accepted train-point bounds, pad5%; nếu bounds cắt vùng cần reconstruct phải tạo declared larger-ROI variant từ train inputs, không theo GT. Log full bounds và excluded train-point coverage. Official evaluator ROI khác với fusion ROI. Unbounded fullscene không dùng grid vô hạn; boundedROI/scalable blocks hoặc native GOF.

Stream một view tại một thời điểm. Sparse block allocation/chunked grids; memory budget tương ứng training profile, nhưng trainer/Adam phải unload trước meshing. Overlap≥truncation band+one extraction stencil; integration weights cap/log; unknown voxels không là known empty. TSDF zero-isosurface xuất faces; merge seams bằng world-space tolerance cùng voxel scale. Densegrid memory O(nx*ny*nz), dimensions/capacity phải tính trước allocate; sparsegrid có overhead hash/blocks, không hứa O(Npoints).

## 3. Alternatives và lựa chọn

| Route | Required inputs | Ưu điểm / giới hạn | Project use |
| --- | --- | --- | --- |
| TSDF→Marching Cubes | Calibrated validdepths, scale/K/T | Stream/block được; voxel smoothing/cubic memory nếu dense; holes khi depth missing | Main B1/B2/Ours; B0diagnostic |
| Screened Poisson | Dense oriented samples, consistent normals | Smooth surfaces/robust noisy points; octreeRAM, can hallucinate closure | DN reference; separate declared backend, không CPU fallback |
| GOF opacity→MT | Native compatible Gaussian field/evaluator+training views | Adaptive surface, supports unbounded scenes; Delaunay/time/memory và sparse primitives | OptionalB3 |
| MILo signed scalars→MT | Trained pivots+signed values+topology+renderer | Mesh losses duringtraining; requires differentmodel10GB+ reference | Independent laterresearch |

Marching Tetrahedra là extractor cần scalar field, không là cách tự tạo geometry từ sparsePLY. GOF uses opacity0.5 crossing refinement/binary search; MILo signedfield interpolation khác. Poisson không trực tiếp nhận un-oriented sparseCOLMAP để đảm bảo accurate scene. Indoor có smooth lowtexture surfaces cần reliability; outdoor cần masks/bounds/scale và background completeness disclosure, không universalbest.

## 4. Cleaning và Blender export

Lưu rawmesh immutable trong run; cleanedmesh có riêng config/parenthash. Remove nonfinite vertices/degeneratefaces (invalid geometry phải báo trước sửa), duplicatefaces/vertices trong tolerance1e−6×sceneradius đề xuất. Normals consistent theo connectivity; report nonmanifold/boundaryedges và connectedcomponents. Do not automatically erase all but largestcomponent; thresholdpruning phải giữ đối chứng/raw và thống kê diện tích mất. Holefilling/smoothing/decimation optionalvariants, không tune bằng testGT.

PLY binary/ASCII hoặc OBJUTF8 có vertex+face arrays, finitecoordinates, validindices, no zeroareafaces, orientationmetadata. Export với inverse scene normalization về arbitrarySfM units; nếu Blender displaytransform dùng thêmma trận phải lưu world-to-display và inverse. Không claim meters khi chưametric alignment. Texturebaking/UV không thuộc mainmesh requirement, có thể để phase riêng.

## 5. Acceptance và tests tương lai

Synthetic plane/cube với calibratedposes: verify axial/ray-depth conversion, signs, intrinsics resize, TSDFsurface location trong tolerance1–2voxels tùy fixture. Unknown-depth không sinhsurface; emptyinput fail rõ; chunkseams match unchunkedfixture, faces/indices valid. GPUoperation device/logs explicit, sampledVRAM không exactpeak.

Real-subset mesh phải reopen parse; no NaN/Inf, countfaces>0, bounds hợp lý so cameras/points, front-facing meshdepth agrees input depths trên validregions; report residualdistribution, coverage và remainingholes. Import PLY/OBJ vào Blender để xác minh scale/orientation/normals khi user yêu cầu Phase04 thực thi. Chưa có meshquality metric/result trong phiên tài liệu.
