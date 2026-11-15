---
name: experiment-evaluation
description: Run or review fair baseline comparisons, ablations, rendering and mesh metrics, alignment, resource reports, and research claims. Use for Phase06 and Phase07 experiments or evaluation protocol changes in this repository.
---

# Experiments and evaluation

## Kích hoạt và context

Đọc AGENTS, [Phase06](../../../docs/plans/phase_06_baselines.md), [Phase07](../../../docs/plans/phase_07_evaluation.md), [evaluation](../../../docs/tech/evaluation_specification.md), [reproducibility](../../../docs/tech/reproducibility.md), [comparison](../../../docs/research/method_comparison.md). Paper references: P02 §7, P01 §6, P14 §V, P05 App.B, P07 §4, P11 §6.

## Workflow

1. Inspect measured artifacts và all run outcomes, selected protocol/group keys và actual user scope. Không khởi training khi request chỉ review/report.
2. Freeze same data/resolution/pose/split/masks/budgets/evaluator/seed/extraction group. Declare transductive SfM; never call training-view PSNR held-out NVS.
3. GT chỉ ở evaluator. Verify source/hash, metric units, official scene threshold/crop/alignment, evaluator revision. Missing prerequisite → metric blocked/null, không suy threshold từ paper khác.
4. Validate known metric/scale/alignment fixtures; pin SSIM/LPIPS definitions/weights. CUDA-required learned metrics không CPU fallback.
5. Use same registration policy; separate native/common extraction, raw/cleaned metrics. Mesh sampling theo area, not vertex density.
6. Aggregate perview/scene/seed với failed/skipped counts và dispersion; retained OOM không bị drop ngầm. Final test không quay lại chọn hyperparameters.
7. Produce tables/qualitative views cùng camera/crop/color mapping và source links; label paper/proposed/measured rõ, giữ negative results. Không claim SOTA/novelty từ một pilot scene.

## Tools và commands

Read JSON/CSV/configs/logs; primary official evaluators và targeted tests khi implementation đã có. Tool/package presence không đủ official-protocol compliance. Static scientific plots xuất standalone artifacts khi phase reporting yêu cầu; không tạo charts bằng invented numbers.

## Validation, lỗi và hoàn thành

Common failures: CD squared-vs-unsquared, percent-vs-fraction F1, unknown SfM scale, per-method ICP advantage, test exposure fitting, stale cache/profile mismatch, paper numbers chép vào result table. Done khi each metric traces to inputs/evaluator/protocol và tests thực chạy; thiếu GT protocol giữ blocker, không gọi final evaluation Done.
