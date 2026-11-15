---
name: geometry-regularization
description: Design, implement, or review depth, normal, planar, multi-view, and sparse-confidence losses for Gaussian reconstruction. Use for geometric supervision or proposed-method changes, keeping existing methods and research hypotheses distinct.
---

# Geometry regularization

## Kích hoạt và context

Đọc AGENTS, [loss spec](../../../docs/tech/loss_specification.md), [math](../../../docs/research/mathematical_foundations.md), [Phase03](../../../docs/plans/phase_03_pgsr_training.md) hoặc [Phase05](../../../docs/plans/phase_05_proposed_method.md), [proposal](../../../docs/research/proposed_method.md). Required papers: P14 §IV, P01 §5, P03 §4.3 p.6, P05 §4, P09 §3; novelty check có CDGS trong [gaps](../../../docs/research/research_gaps.md).

## Workflow

1. Chọn exact objective/variant và ghi existing/proposed status. Không gọi sparse SfM anchors hoặc confidence weighting là mới khi chưa kiểm chứng overlap.
2. Xác định tensor shapes, camera/pixel/normal/depth units, valid mask, gradient/detach boundaries và schedule. Plane distance/normal phải cùng accumulated weighting.
3. Derive homography từ signed plane convention; test direct3D projection parity trước dùng formula từ paper. Check axial/ray depth và inverse-depth khác nhau.
4. Với Ours v0, chỉ thêm fixed training-only confidence anchors; không thêm edge term trùng PGSR, monocular model hoặc pose refinement.
5. Guard positive depth trước log/division, near-parallel rays, zero cross products, empty masks và constant-patch/image normalization. Connected zero không đồng nghĩa geometry coverage tốt.
6. Unit/finite-difference gradient tests rồi CUDA subset nếu scope cho phép. λanchor=0 phải có parity với cùng B2 revision/config; ablate uniform/error-only/full/shuffle.
7. Log raw/weighted loss, masks/coverage/gradient norms và compare trên frozen validation protocol, không tuning test GT.

## Tools và commands

Đọc PDF/source primary; NumPy fixtures cho math và PyTorch CUDA cho required compute. Dùng targeted tests được phase triển khai, không tạo fake training metrics. Trước đổi thuật toán phải cập nhật cited equation/variant và config, không copy long paper text.

## Validation, lỗi và hoàn thành

Common failures: plane sign, dehomogenization thiếu, normal frame mismatch, all-view residual/confidence leakage, model tắt weights để né hard samples, all-empty loss bị coi convergence. Done khi formula/source/status rõ, meaningful gradient/mask tests pass thực tế và zero-weight parity có evidence. Scientific claim vẫn pending tới full evaluation.
