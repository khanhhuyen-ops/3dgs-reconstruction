# Mathematical foundations và conventions

Nguồn chính theo [paper IDs](paper_inventory.md). Công thức chuyển hệ tọa độ là derivation hình học cho contract dự án; proposed losses được đánh dấu riêng. Không copy biểu thức khi dấu/notation khác implementation.

## 1. Camera, projection, similarity

Theo [COLMAP format](https://colmap.github.io/format.html#images-txt), extrinsics world-to-camera:

```math
x_c=R_{cw}X_w+t_{cw},\quad C_w=-R_{cw}^{T}t_{cw},\quad
T_{wc}=\begin{bmatrix}R_{cw}^T&C_w\\0&1\end{bmatrix}.
```

Quaternion COLMAP theo thứ tự `(qw,qx,qy,qz)`, Hamilton; camera x-right/y-down/z-forward. IDs không phải contiguous indices. Pinhole projection `u=fx*x/z+cx`, `v=fy*y/z+cy` chỉ dùng sau đúng distortion processing.

Với SIMPLE_RADIAL: normalized `(x/z,y/z)` nhân `1+k*r²` rồi nhân focal và cộng principal point. Intrinsics từ model cũ1920×1080 không dùng trực tiếp cho video1280×720. Resize/crop là phép image transform `A`, `K'=AK`; khi dùng half-pixel resampler phải tính translation tương ứng, không chỉ nhân scale. COLMAP first-pixel center `(0.5,0.5)` phải được adapter chuyển rõ sang pixel-index/`grid_sample`, có synthetic test. Không thêm half-pixel hai lần.

Normalization được **đề xuất**, suy trực tiếp từ similarity: `X'=s*(X-c)`, `C'=s*(C-c)`, `R'=R`, `t'=s*(t+R*c)`. Lưu `s>0`, `c`, inverse transform; depth axial `Z'=sZ`. Một scene dùng cùng transform cho points/cameras/mesh. Không dùng GT để chọn `s,c` cho training. Monocular units vẫn arbitrary.

## 2. Gaussian và compositing

P02 §§4–5, Eq.(3–7), pp.3–5:

```math
G_i(X)=\exp[-\tfrac12(X-\mu_i)^T\Sigma_i^{-1}(X-\mu_i)],\quad
\Sigma_i=R_i\,\mathrm{diag}(s_i^2)R_i^T,
\quad\Sigma'_i=J R_{cw}\Sigma_iR_{cw}^TJ^T.
```

`μ∈R³` mean, `s>0` scales, `R` từ unit quaternion, `J∈R²×³` projection Jacobian. Scale dùng exponential/log parameterization; covariance positive definite nếu ba scales dương. Render projected footprint cho alpha `a_i(p)`, visibility sorting tạo `w_i=a_i ∏_{j<i}(1−a_j)` và `C=Σ_i w_i c_i +(1−Σ_iw_i)c_bg`. Gradient flow qua renderer tới geometry/opacity/SH; ordering/cull/split rời rạc không phải smooth everywhere.

P01 §4.1 Eq.(4–6): `X(u,v)=μ+s_u*t_u*u+s_v*t_v*v`; chỉ hai scales, normal `t_u×t_v`. Covariance rank2; không invert3×3 như B0. Cần ray-splat intersection renderer riêng.

## 3. Depth và normals không cùng semantics

Expected axial depth được chọn cho B0 diagnostic: `Z_exp=Σw_i z_i / max(Σw_i,ε)` trên ray có alpha hợp lệ; đây là normalized expectation, không surface guarantee. P05 Eq.(3) và phần normalization p.3 mô tả route tương tự. Phải ghi khác với unnormalized sum, median depth của2DGS và ray-distance.

P14 §IV.A Eq.(1–4) pp.5–6: flatten minimum scale; alpha-composite camera-frame plane normals `N=Σw_i n_i` và plane offset `d=Σw_i d_i`. Với plane convention **`nᵀx=d`**, axial depth:

```math
Z(p)=\frac{d(p)}{N(p)^T K^{-1}\tilde p}.
```

Giữ numerator/denominator cùng weighting: normalize `N` đơn phương làm mất tỉ lệ và đổi depth. Điều kiện `|Nᵀr|>ε`, opacity sufficient, finite, `Z>0`; mask parallel/degenerate rays. Công thức loại ảnh hưởng accumulated opacity cho cùng plane, không đảm bảo một surface khi trộn nhiều planes.

Normal-from-depth: backproject `P(u,v)=Z K^{-1}[u,v,1]ᵀ`; cross finite differences và normalize, align orientation với camera-facing normal. P01 Eq.(15), P14 Eq.(5). Biên ảnh, missing depth, near-zero cross product phải mask. Đổi normal camera↔world bằng rotation, không bằng translation.

## 4. Homography, consistency và dấu plane

Derivation dự án: với `x_n=R_nr*x_r+t_nr`, plane `nᵀx_r=d` thì `H=K_n(R_nr+t_nr*nᵀ/d)K_r^{-1}`. Nếu plane dùng `nᵀx_r+δ=0`, `δ=−d`, biểu thức thành `R_nr−t_nr*nᵀ/δ`. PGSR Eq.(8) p.6 trình bày dấu trừ với convention của paper/code. Không copy dấu nếu chưa ràng buộc cả normal/offset/relative transform. Bắt buộc chứng minh homography warp bằng direct3D projection trên plane tổng hợp.

P14 Eq.(9): roundtrip pixel residual, **dehomogenize sau mỗi warp**, reject negative/out-of-image/occluded samples. Eq.(10): patch NCC, constant patches phải mask/epsilon denominator; NCC không là RGB loss trên held-out images. P09 Eq.(5–8) bổ sung Gaussian co-visibility, không xác định được visibility từ chỉ frustum membership.

## 5. Geometry objectives

P02 Eq.(7): `(1−λ)L1+λLD-SSIM`, λ0.2. Theo implementation đã chọn cần ghi DSSIM là `1−SSIM` hay `(1−SSIM)/2`; weight cùng tên không đủ bảo đảm loss tương đương.

P01 Eq.(13–16): distortion Σᵢⱼwᵢwⱼ|zᵢ−zⱼ|, normal Σᵢwᵢ(1−nᵢᵀN_depth). Hai terms có semantics khác P14 single-view L1 normal difference. P07 Eq.(11) detach blend weights trong distortion; không mặc định áp dụng cùng gradient choice cho mọi method.

P05 Eq.(5): affine fit `argmin_(a,b) Σ_valid |a D_mono+b−Z_SfM|²`, không áp affine depth lên inverse-depth nếu chưa xác định output prior. Rank/coverage kém thì reject fit; không lấy GT để sửa scale.

P03 §4.3 p.6: L2 giữa rendered depth và sparse SfM z tại observed keypoints đã là existing supervision. P09 Eq.(9): local calibration dùng median/MAD, không tự thay bằng least squares rồi gọi nguyên GVGS.

## 6. Surface extraction

Weighted truncated distance fusion, theo [Curless–Levoy](https://graphics.stanford.edu/papers/volrange/), được đặc tả cho projective-z backend:

```math
f_i(X)=\mathrm{clip}((Z_i(\pi(T_iX))-(T_iX)_z)/\tau,-1,1),\quad
F'=(WF+w_i f_i)/(W+w_i).
```

Chỉ integrate valid observed voxels theo backend, không đánh dấu toàn vùng sau surface thành free space. `τ` và voxel_size cùng scene units, backend distance definition phải lưu. Isosurface zero, unknown voxels không tự thành surface. [Meshing spec](../tech/meshing_specification.md) tách backend/native variants.

Poisson (P05 §4.5; P04 pp.12–13) cần oriented samples, giải indicator field phù hợp normal vector field, rồi extract isosurface. Không coi oriented sparse COLMAP points đủ dense. Có thể fill hole không được quan sát; completeness metric cần mask hợp lệ.

GOF P07 Eq.(9–10), §3.4: opacity tăng theo ray rồi lấy minimum qua training views, extract level0.5. Field không là signed distance nên binary search refine crossing cần thiết. MILo P11 §4.2–4.3 dùng signed scalars, interpolation `v=(f_a*p_b−f_b*p_a)/(f_a−f_b)` với hai signs trái nhau; gradient conditional trên topology cố định. Hai fields không tráo cho nhau.

## 7. Proposed mathematics

Chỉ các công thức mang nhãn `PROPOSED` trong [proposed method](proposed_method.md) và [loss specification](../tech/loss_specification.md) thuộc Ours. Không gán chúng cho PGSR hay trình bày như định lý confidence đã calibrated.
