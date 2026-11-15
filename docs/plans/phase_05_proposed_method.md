# Phase 05 — Proposed research method

Trạng thái hiện tại: **PLANNED — chưa triển khai/chạy**. User request tiếp theo quyết định việc thực thi; xem [master plan](master_plan.md). Mọi thông số preset là thiết kế, không benchmark đã đo.

## 1. Phase overview

Implement candidate sparse-confidence anchors trên B2 **sau yêu cầu mới**, chưa claim novelty. Phase này chuẩn bị pilot/ablation, không tự chạy đầy đủ comparative campaign.

## 2. Research objective

Kiểm định H1/H0 trong [proposal](../research/proposed_method.md): confidence có giá trị vượt unweighted/error-only anchors và ngoài anchor pixels không?

## 3. Scientific references

P03 §4.3 p.6, P05 §4.1, P09 §3, P14 §IV; [novelty audit](../research/research_gaps.md), CDGS related source. Existing terms và proposed term phải phân biệt.

## 4. Current dependencies

B2 và Phase 04 geometry pipeline ổn định; Phase 01 train observations qualified; novelty matrix được rà lại, protocol/split/resource frozen. Chưa có valid video model thì không có real pilot.

## 5. Input requirements

Train-only point observations/residuals/unique view count/ray angles, prepared cameras/transforms, B2 state/config. Không dùng reported all view error hoặc test points/color để tính confidence.

## 6. Expected outputs

Versioned confidence cache, anchor loss module/config, per cue/coverage diagnostics, parity tests, pilot log và preregistered ablation matrix. Full Ours metrics vẫn null khi chưa measured.

## 7. Detailed architecture

Offline fixed heuristic from COLMAP → observation depth sampler → weighted Huber log depth loss → B2 objective. All B2 existing flatten/single/multi/exposure giữ nguyên; no extra neural prior or pose optimizer.

## 8. Mathematical requirements

Proposed confidence product of log track support/error decay/parallax; observed axial depth; positive-only log residual; detached confidence weight sum normalization; schedule 7000/ramp 2000. [Loss spec](../tech/loss_specification.md) canonical.

## 9. Implementation tasks

Build cache keys with split/transforms, residual recompute; train unique observations; confidence cue ablations; bilinear anchor sampling; empty/invalid depth guards; λ=0 parity; logging active counts, anchor/non-anchor residuals, compute overhead. Không thêm edge aware loss trùng B2.

## 10. Ordered execution steps

1) Rà soát overlap matrix. 2) Freeze ablations A0–A6. 3) Tạo cache chỉ từ train observations. 4) Chạy unit/gradient tests. 5) Kiểm tra parity với B2 khi λ=0. 6) Chạy CUDA pilot nhỏ để đo stability/VRAM. 7) Freeze hyperparameter sweep và selection rule cho Phase 06.

## 11. Technical dependencies

Existing B2 renderer+torch CUDA; CPU diagnostic precompute được phép, anchor loss/backward CUDA. Không download Depth Anything/Omnidata ở Ours v0.

## 12. Hardware and resource constraints

Anchors chunked, one image/step; cache metadata CPU, sample tensors CUDA. Cùng Gaussian/resolution caps với B2. Memory/runtime overhead phải đo, không quảng cáo negligible trước run.

## 13. Validation strategy

Analytic confidence monotonic/capped; duplicate/train-only tests; log loss similarity scale invariance; zero/negative/NaN/empty safeguards; gradient presence to geometry/no camera/anchor gradient; deterministic shuffle control manifest.

## 14. Evaluation metrics

Pilot loss/active coverage/gradient norms, anchor vs non-anchor residuals, time/VRAM overhead. Final geometry/NVS và dispersion do Phase 06/07; không chỉ dùng anchor fit để chứng minh hypothesis.

## 15. Risks and mitigation

Redundant novelty, biased tracks correlated frames, circular visibility masks, sparse coverage/texture bias, hyperparameters test leakage. Mitigate direct prior controls, angle factor ablation, train-only recomputation và validation-only selection.

## 16. Acceptance criteria

Khi λ=0, behavior khớp B2 trong tolerance; gradients hữu hạn, không dùng test input, cache có provenance; A1 uniform/A2 error-only/A4 full/A5 shuffle sẵn sàng. Scientific acceptance cần kết quả so sánh thực nghiệm, không đánh đồng code test với việc chấp nhận H1.

## 17. Definition of Done

Candidate implementation chạy CUDA pilot và artifacts có thật nếu input/backend cho phép; hypothesis vẫn PROPOSED tới comparative evaluation. Nếu phương pháp trùng prior work hoặc cho kết quả âm, giữ finding đó và không claim novelty.

## 18. Next phase dependencies

Phase 06 chạy frozen matrix/seeds; Phase 07 kiểm định và báo cáo. Thêm dense prior/edge propagation là variant/task khác có prior provenance, không tự mở rộng Ours v0.

