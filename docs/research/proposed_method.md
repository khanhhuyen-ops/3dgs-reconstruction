# Proposed method — giả thuyết chưa kiểm chứng

Working title theo user: **COLMAP-guided Confidence-aware Geometry Regularization for Gaussian Splatting**. Trạng thái: `PROPOSED`, chưa novelty claim, chưa implementation/training/result. [Gap audit](research_gaps.md) ghi overlap trực tiếp P03/P05/P09 và CDGS; chỉ ghép depth, confidence, edge-aware terms không đủ thành đóng góp mới.

## 1. Câu hỏi và phạm vi

Liệu reliability của **training-only COLMAP observations** có giúp sparse-anchor regularization trên PGSR ổn định hơn unweighted/error-only anchors, dưới cùng6GB resource cap? Primary hypothesis H1: weighted anchors cải thiện geometry ngoài chính các anchor samples mà không mất rendering quá mức. H0: không có lợi ích reproducible hoặc làm completeness xấu đi. Ngưỡng meaningful effect phải preregister trước khi xem final test; không đặt “đạt F1=…” từ số paper.

Existing B2 giữ flatten, edge-aware normal, multi-view geometry/NCC và exposure behavior đã chọn. Ours v0 chỉ thêm **một sparse-anchor loss**; không thêm monocular inference, dense confidence propagation, pose optimization hoặc loss edge-aware thứ hai. Các nhánh này nếu nghiên cứu sau là variant riêng.

## 2. Dữ liệu cố định

Mỗi accepted point `j`: position `X_j`, training observations `(image_id, point2D_idx, xy)`, residuals tính lại với camera/distortion, unique training-camera count `L_j`, parallax từ train rays. Không dùng point error từ header nếu tính trên cả held-out views. Visibility chỉ xác nhận observation support; không dùng frustum projection để bịa dense visible labels.

Sau undistortion, chuyển đúng observed keypoints vào image mới. Anchor ở `(i,j)` có axial `z_ij=(R_i X_j+t_i)_z>0`. Cùng normalization cho Gaussian/anchor/pose. Chỉ train images tạo loss, multi-view neighbors và confidence. Các tracks có duplicate same-image observations phải qua qualification của Phase01, không đếm duplicate để tăng confidence.

## 3. Confidence heuristic — PROPOSED

```math
c_j=\mathrm{clip}(\ell_j\,r_j\,a_j,0,1),\quad
\ell_j=\min(1,\log(1+L_j)/\log(1+L_{cap})),\quad
r_j=\exp[-e_j^2/(2\sigma_e^2)],\quad
a_j=\sin^2(\min(\theta_j,\pi/2)).
```

`L_j`: unique train cameras; `e_j`: median train reprojection residual ở **prepared resolution**; `θ_j`: median pairwise angle của train observation rays; `L_cap`, `σ_e` là tunable config từ training/validation, chưa tối ưu. Cache theo camera/point/image-transform/split hashes. Confidence là ranking heuristic, **không xác suất đúng** hoặc posterior uncertainty. Nhiều video frames gần nhau tạo correlated evidence; angle factor chỉ là giả thuyết giảm bias đó.

Để kiểm tra logic ban đầu có thể đề xuất `L_cap=10`, `σ_e=2px`, nhưng các giá trị này chưa được kiểm chứng, phải sweep/config riêng ở Phase05. Không đổi resolution mà giữ σ theo pixel nguồn; không normalize độc lập mỗi baseline.

## 4. Sparse loss — PROPOSED

```math
r_{ij}=\log Z_i(u_{ij})-\log z_{ij},\qquad
L_{anchor}=\frac{\sum_{ij}m_{ij}\,\mathrm{stopgrad}(c_j)\,\rho_\delta(r_{ij})}
{\max(\sum_{ij}m_{ij}\,\mathrm{stopgrad}(c_j),\epsilon)}.
```

`Z_i` là PGSR plane-depth sampled bilinear tại **observed** pixel; `m` giữ inside-image, positive finite depths và valid renderer coverage. Huber δ mặc định thiết kế0.1 trong log-depth, cần sweep. Không thay invalid/negative depth bằng epsilon rồi coi đó valid. All-empty → connected scalar zero và log active-count0; không divide0.

Gradient đi qua predicted depth tới Gaussian parameters. Anchors/confidence/cameras frozen. Tránh tự weight bằng predicted residual giảm dần tới0 gây model “tắt” supervision. Nếu thêm occlusion test predicted-depth phải warmup và ablate vì có thể khóa reconstruction sai; v0 không thêm circular confidence gate ngoài renderer validity.

Objective: `L_ours=L_B2+λ_anchor(t)*L_anchor`. Research schedule đề xuất start7000, ramp2000 iterations; λmax sweep `{0,0.01,0.05,0.1}`, chọn trên validation cùng compute budget. Đây là **thiết kế**, không tham số PGSR paper. Development schedule riêng chỉ smoke, không so quality.

## 5. Ablations bắt buộc

| ID | Giữ/đổi duy nhất | Mục đích |
| --- | --- | --- |
| A0 | B2, λanchor=0 | Integration parity |
| A1 | Anchors, uniform weights | Giá trị của sparse depth đã có |
| A2 | A1 + reprojection-error weights | Đối chứng existing reliability cue |
| A3 | A2 + track-length factor | Kiểm tra unique-view support |
| A4 | A3 + angle factor | Candidate fullOurs |
| A5 | Shuffle weights, cùng distribution | Negative control về reliability ranking |
| A6 | Warmup/ramp ablation | Tác động curriculum so fixed schedule |

Không thêm edge regularizer đã có trong PGSR rồi tính là novelty. Nếu test dense priors: thêm DN/CDGS-like control với cùng prior checkpoint và inference cost, không gộp A4 vì confound.

Đo RGB held-out, native/common mesh, coverage, residual distributions, anchor/non-anchor regions, runtime/peak allocation. Seeds42/43/44 đề xuất; một scene là pilot. GPU atomics có nondeterminism phải báo dispersion. Final test chỉ chạy sau freeze; không chọn loss weight bằng GT geometry test.

## 6. Điều kiện đóng research question

Chỉ viết “experimentally validated” khi có artifacts parse được, protocol/config/splits pinned, runs đủ, evaluator và variance rõ. Nếu gain không ổn định hoặc trùng prior: ghi reproduction/negative result. Phase05 phải cập nhật novelty matrix trước implementation algorithm mới; kết quả hiện tại của Ours là **chưa có**, mọi metric là `null`.
