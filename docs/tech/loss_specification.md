# Loss specification

**Implementation17/11:** B2 objectives đã có tại `src/pgsr/geometry.py`, native de24f1a official-code guarded variant; [actual reductions/overrides/tests](pgsr_training_usage.md). Real CUDA smoke đã chạy, full research/evaluation pending. Với B2 hiện tại, single-view giữ full H×W mean sau masked-zero; geo/NCC giữ native mean(weight×residual), không tự áp quy tắc chia weight sum khái quát bên dưới. Ours/B1 vẫn chưa triển khai.

B0 RGB loss đã tích hợp từ pinned official2023 `utils.loss_utils` trong Phase02; CUDA finite-gradient checks đã chạy, xem [B0 usage](b0_training_usage.md). Các geometry/B1/B2/Ours losses chưa triển khai. Scientific origin: [P01/P02/P05/P07/P09/P14](../research/literature_review.md); camera/plane math tại [foundations](../research/mathematical_foundations.md); Ours tại [proposal](../research/proposed_method.md). Mọi training reduction trả FP32 CUDA scalar. RGB/geometry masks hợp lệ lấy từ contract, không GT test.

## Inputs và reductions

Reference `I,C:[3,H,W]`, alpha/plane distance/depth `[1,H,W]`, rendered normals `[3,H,W]`, mask `[1,H,W]bool`; camera K/T fixed. Neighbor buffers cùng camera-specific shapes, cần grid sampling chuyển đúng resolution. Anchors `xy:[M,2]`, `z,c:[M]`, valid `[M]`. B1 distortion map do native renderer xuất; không materialize all-pairs `[N,N,H,W]`.

Masked mean: sum trên valid elements chia valid count (hoặc weight sum cho weighted loss). Empty mask → connected zero và active-count0; caller phát hiện kéo dài thiếu coverage. Không “mean” trên array rỗng, không dùng `nan_to_num` che invalid renderer. Masks/selection/detached reliability không được gradient; predicted geometric values vẫn có gradient như method yêu cầu.

## L-RGB — Existing

`Lrgb=(1−λ)*mean(|C−I|)+λ*(1−SSIM(C,I))`, λ0.2 trong chosen project convention; source P02 §5.1 Eq.(7). DSSIM có khác factor2 giữa implementations phải pin. SSIM window/kernel/data-range/channels giữ đúng chosen revision. Background/masks/color space đồng nhất theo evaluation spec.

Gradient tới geometry/opacity/SH qua renderer; input RGB/cameras frozen. Activate từ iteration1. Test identical images→L1=0, SSIM=1; perturbed render có finite gradients. Không so values từ masked RGB với unmasked baseline mà không công bố.

## L-scale — Existing PGSR

`Ls=mean_visible(min_k s_ik)`, weights và reduction theo chosen B2 variant; P14 Eq.(1) p.5. Input activated positive scales `[N,3]`, visible `[N]`; output scalar. Gradient vào smallest activated scale; ties nondifferentiable, test tránh ties. Weight100 ở reference; không áp full3D flatten term cho rank2 B1.

Active theo native training behavior; hard floor/log-scale stability không ép exact zero vì Gaussian3D covariance cần finite. Test rotation không đổi smallest scale, analytic finite gradient trên unique-min fixture; no visible Gaussians→zero/log count.

## L-single — Existing PGSR

`Lsv=mean_valid(w_edge(p)*||N_depth(p)−N_render(p)||1)`, P14 Eq.(5–6) pp.6; code/paper edge weights khác trong [notes](../research/pgsr_implementation_notes.md). `N_depth` từ finite differences backprojected plane-depth; `N_render` theo native renderer normalization. Orientation cùng camera frame; không normalize plane normal **trước** depth division làm mất matched weighting.

Gradient flow qua depth và rendered normal trừ detach được native code công bố. Edge map từ input RGB detached, no gradient tới camera. Mask image borders/invalid depth/near-zero cross product. Constant RGB gradient range phải có epsilon-safe behavior bằng documented hardening. Activate B2 research sau7000; development100 là smoke deviation.

Tests: plane cho depth/normal đồng nhất→zero/tolerance; flipped normal làm loss tăng; constant image khôngNaN; edge weighting không smooth ngang discontinuity giả. Finite-difference gradients trên fixture plane slanted, gồm visibility/mask cố định.

## L-multi-geometry — Existing PGSR

Input reference/neighbor depth, K/T, pixel samples `[M,2]`; backproject→neighbor→sample neighbor depth→reference→dehomogenize. Residual `φ=||u_back−u_ref||2:[M]`. Weighted mean theo chosen native code hoặc paper Eq.(9); không giả hai implementations giống nhau. Camera-relative transform từ `T_nw*T_wr`, source P14 Eq.(7–9) p.6.

Valid khi zpositive, inside-image, renderer validity và declared occlusion gate; code variant threshold1px/exp-weight tham chiếu notes. Hard gate và confidence detached; residual vẫn differentiable qua sampled depth. Không detach toàn neighbor geometry vô tình. Mỗi step một train neighbor; no neighbor→zero và log coverage, không mượn test frame.

Numerics: dehomogeneous divisor/plane offset bounded; invalid points loại trước division/log; geometric occlusion không chỉ bounding-box test. Test two identical poses/planes consistent→small residual; known translation/occlusion cho expected masks. Compare homography/direct3D projection để tránh plane-sign bug.

## L-multi-photo — Existing PGSR

Grayscale patches reference/warped neighbor `[M,P²]`; paper P14 Eq.(10) p.7 NCC11×11; chosen official-code variant khác patch size trong notes. Per-patch NCC normalized by variances, stabilize epsilon, reject low-variance patches/outside-image. Loss `mean_valid(1−NCC)` hoặc native LNCC cụ thể phải giữ pinned; không thay squared NCC bằng NCC mà gọi same baseline.

Gradient qua warp tới geometry, RGB input constant. `grid_sample` normalization/align_corners và COLMAP pixel-center conversion phải thống nhất, test identity sampling và half-pixel offset. Activation theo B2 multi-view gate. Chunk patches, không giữ toànfull pair graph cho mọi cameras.

## L-exposure — Existing PGSR

P14 Eq.(12–14) p.8: `C_app=exp(a_i)*C+b_i`; L1 dùng app image khi raw structural loss<0.5; SSIM giữ canonical render. Exposure train variables có gradient theo gate, input image/camera fixed. Monitor exp overflow, identity initialization `a=b=0`; checkpoint app model và optimizer.

Gate branch không smooth tại threshold; unit test hai phía, không gradient-check đúng threshold. Research held-out không fit `a,b` từ RGBtest; canonical appearance primary. Exposure variant và dataset use phải declared; paper chỉ dùng TNT, không assume mọi dataset.

## B1 và B3 native losses

P01 Eq.(13–16) p.5: depth distortion và normal alignment; alpha1000 bounded/100 unbounded, beta0.05 **paper reference**. Native renderer computes distortion efficiently, tested against small explicit sum. Gradient qua sample depths và weights theo method. P07 Eq.(11) p.4 detach weights khi distortion; normal view-dependent định nghĩa khác shortest-axis B2. Reusing B2 normal tensor cho GOF là invalid compatibility.

B0 không có các terms này. Ours dựa B2 không tự thêm distortion vì đã nghiên cứu methods khác. B3 phải dùng native loss/field contract; không chạy GS covariance/training CPU nếu thiếu CUDA.

## L-anchor — PROPOSED

Theo [proposal](../research/proposed_method.md), sparse samples `Z_hat=bilinear(Z,xy):[M]`; Huber residual `r=log(Z_hat)−log(z)`; `L=sum(m*c*rho_delta(r))/max(sum(m*c),eps)`. `δ=0.1` là proposed sweep start, không paper default. Confidence fixed từ train observations, detach; no gradient vào anchors, poses hoặc sparse point cloud. Gradient predicted depth→Gaussian geometry/opacity, không SH trực tiếp trừ shared renderer phụ thuộc.

Research activate7000/ramp2000 và λmax sweep như proposal. Positive finite depth filtering **trước** log; epsilon denominator chứ không biến invaliddepth thành valid. If allweights0→zero/log; nếu geometry coverage collapse thì stage quality gate fail, không báo loss0 là thành công. Không thêm predicted-residual weight giúp model né khó anchors ở v0.

Tests bắt buộc: prediction=anchor→zero; common similarity scale thay cả depths giữ residual; good vs badconfidence đúng normalization; zero/negative/NaN depth rejected; duplicate track filtering; all-empty finitezero; anchor/camera gradients absent; geometry finite-difference gradient. Shuffle-confidence negative control ở experiment, không là unit test chứng minh quality.

## Composed objectives và logging

`LB0=Lrgb`; `LB1` theo P01; `LB2=Lrgb+λsLs+λsvLsv+λgeoLgeo+λnccLncc`; `LOurs=LB2+λanchor(t)Lanchor`. Code-specific norm/reduction/appearance phải pin bằng variant metadata. Log loss raw, coefficient, weighted value, valid count, gradient norms, alpha coverage và schedule state. Loss reduction trong một run không chứng minh better mesh hoặc SOTA.
