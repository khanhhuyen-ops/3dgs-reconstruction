# Gaussian representation và checkpoint compatibility

Nguồn: P02 §4–5; P01 §4; P14 §IV.A; P07 §3; P11 §4, theo [review](../research/literature_review.md).

| Nhánh | Trainable tensors dự kiến | Semantics |
| --- | --- | --- |
| B0 | means[N,3], log_scales[N,3], quaternions[N,4], opacity_logits[N,1], SH[N,(l+1)²,3] | Full3D covariance, unit quaternion, opacity sigmoid |
| B1 | means[N,3], log_scales[N,2], quaternions[N,4], opacity_logits, SH | Rank2 disk, specialized intersection |
| B2/Ours | B0-shaped Gaussian tensors; train-view exposure parameters; native ADC state | Flat3D via min-scale regularization; normal shortest axis; plane buffers |
| B3 | Native GOF3DGS+renderer/appearance state | Ray-based evaluation/opacity field; source-specific filtering |
| MILo reference | Selected pivots, signed values[Npivot,9], tetrahedral connectivity | Extra learnables and topology; không compatible B2 |

Shapes là logical contract; native repo có thể transpose SH, separate DC/rest, dùng log/activated scales khác. Adapter khai báo memory layout và activation, không đổi fieldnames rồi assume equivalent. Native quaternion order phải xác minh; COLMAP WXYZ metadata không tự bảo đảm quaternion store bên renderer cùng order.

Initialization từ accepted train-point positions/color; opacity/sizes theo chosen baseline revision. B0 scales từ neighborhood distances (P02 §5.1), clamp positive minimum chỉ numerical safety có log. Recompute point colors train-only theo data contract. Development subsampling deterministic, stratified spatial/visibility nếu có, ghi selected pointIDs; không random dense fake points khi có qualified SfM source.

Parameters FP32 trên CUDA; parser/coordinate diagnostics FP64 CPU. Không dùng FP16 cho quaternion normalization, plane denominators hoặc tiny scales mặc định. Log norms/ranges/saturation và nonfinite counts. Normal direction sign consistent camera-facing per renderer; không lưu arbitrary per-view flipped normals như global orientations.

Checkpoint metadata phải chứa `method`, `variant`, `schema_version`, parentdataset/split/normalization hashes, native revision/submodules, tensor layout/activations/SH degree, renderer version và full optimizer state. Gaussian PLY thường chỉ là inference/export parameter set, không đủ resume. Raw Gaussian PLY cũng không là triangle mesh.

Cross-method conversion chỉ được bổ sung sau khi có defined mapping, loss of information report và forward-render parity test. B0→GOF extraction có thể là supported research variant theo P07 §3.2, nhưng không đồng nghĩa full GOF training. B0→2DGS cần collapse/renderer change nên phải train/compare variant riêng.
