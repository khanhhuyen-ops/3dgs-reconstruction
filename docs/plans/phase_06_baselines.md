# Phase 06 — Research baselines

Trạng thái hiện tại: **PLANNED — chưa triển khai/chạy**. User request tiếp theo quyết định việc thực thi; xem [master plan](master_plan.md). Mọi thông số preset là thiết kế, không benchmark đã đo.

## 1. Phase overview

Orchestrate fair B0/B1/B2/Ours experiment campaign; B3 optional. B0/B2/Ours implementations thuộc phase trước; B1 mới được tích hợp ở đây.

## 2. Research objective

Tách lợi ích representation, proposed loss và extract or khỏi resource/dataset confounds.

## 3. Scientific references

P02 original 3DGS; P01 2DGS; P14PGSR; P07GOF; [comparison](../research/method_comparison.md), [evaluation spec](../tech/evaluation_specification.md).

## 4. Current dependencies

Prepared data và Phase 02–05 gates; compatible 2DGS build; source SHA/profile/metric/mesh protocol freeze. Optional GOF Windows/resource gate riêng. Final GT metrics chưa được tính/tune ở phase này.

## 5. Input requirements

Comparison manifest gồm methods/variants/seeds/configs, cùng input/transforms/split/poses, common profile, native/common extractor configs và ablation selection rule đã định trước.

## 6. Expected outputs

B1 adapter/checkpoints; optional B3 report/checkpoints; campaign manifest và all run status/logs/checkpoints/renders/meshes; validation selection record; failures/OOM/skip reasons retained.

## 7. Detailed architecture

Shared comparison runner dispatches method native adapters; builds new runs, validates comparability keys, renders heldout only after freeze, invokes qualified extractor. B3 own renderer/opacity/MT interface; no direct checkpoint sharing.

## 8. Mathematical requirements

B1 rank 2disk/intersection+distortion/normal; B0RGB; B2 plane depth; Ours anchor objective; B3 opacity field. Match projection/units, không force incompatible losses thành một architecture.

## 9. Implementation tasks

PinB1/B3 sources/submodules; implement/test B1 adapter; define campaign config expansion; reject mismatched comparison keys; reuse completed run only when hashes match; seed matrix 42/43/44, A0–A6; save all outcomes. Optional GOF only after build/subset gate, no installation merely because optional.

## 10. Ordered execution steps

1) Validate immutable comparison manifest. 2) B1 synthetic CUDA+development run. 3) Optional B3 capability gate. 4) Run sequential methods/seeds under resource budget. 5) Validation-only sweep selection. 6) Freeze checkpoints/settings. 7) Export predictions and meshes for Phase 07. No changed test split after seeing performance.

## 11. Technical dependencies

NativeB1/optional GOF extensions+CGAL/tetrahedra components; optional dependency availability/logs explicit. Shared RGB metrics từPhase 02, meshing Phase 04, proposed cache Phase 05.

## 12. Hardware and resource constraints

Sequential one GPU6GB, unload models between runs. Research profile caps/resolution common, native full reproduction group separate. B3 could beBLOCKED_RESOURCE/BACKEND; không omit failed scene khỏimean.

## 13. Validation strategy

Recompute comparison keys and dataset membership; B1 forward/backward/mesh fixture; optional B3 field/extraction fixture; spotcheck parent hashes/native route. Existing mock tests không chứng minh new native backend.

## 14. Evaluation metrics

Training/validation curves, resource/runtime/counts, completion rates; saved test predictions đểPhase 07evaluate. Paper results/table generation không điền thay local metrics.

## 15. Risks and mitigation

Different training iterations, prior training data, cameras, exposure test fit hoặc extract or ROI làm comparison invalid. Mitigate frozen matrix, protocol versions và separate native/common tracks. Không patch chỉOursbasebug mà bỏB2.

## 16. Acceptance criteria

B0/B1/B2/Ours có cùnggroupkeys và saved real artifacts hoặc explicit failed status; optional B3 reason clear; seeds/variants/jobs fully accounted. No post hoc test tuning, raw data preserved.

## 17. Definition of Done

Campaign records complete gồm all success/failed/skipped jobs; no numerical quality claim until qualified evaluator. Code/integrations tested ở levelthực tế, no fake completed runs.

## 18. Next phase dependencies

Phase 07 consumes frozen predictions/protocol. Rerun required chỉ khi artifacts invalid/protocol versionsđổi; không broaden/repeat expensive runs sauchecks đã đạt mà không lý do.

