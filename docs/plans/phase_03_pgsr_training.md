# Phase 03 — PGSR training

Trạng thái hiện tại: **PLANNED — chưa triển khai/chạy**. User request tiếp theo quyết định việc thực thi; xem [master plan](master_plan.md). Mọi thông số preset là thiết kế, không benchmark đã đo.

## 1. Phase overview

Tạo B2 geometry branch dựa official PGSR, chưa thêm Ours. Chọn official-code variant và giữ paper-faithful option riêng nếu có.

## 2. Research objective

Tạo plane-depth/normal nhất quán và kiểm chứng single/multi-view/exposure, không đánh đồng render đẹp với geometry đúng.

## 3. Scientific references

P14 §IV pp.5–8 và §V; [paper/code notes](../research/pgsr_implementation_notes.md); [loss spec](../tech/loss_specification.md).

## 4. Current dependencies

Phase 01 qualified data và Phase 02 shared GPU/checkpoint/metrics infrastructure. Official PGSR SHA+submodules và native build phải verify riêng; B0 success không chứng minh PGSR extension hoạt động.

## 5. Input requirements

Cùng data/split/source/pose/normalization/profile; train neighbors only; train exposure state; no GT geometry hoặc monocular prior.

## 6. Expected outputs

B2 checkpoints, plane-distance/normal/depth buffers, individual loss logs/validity masks, neighbor policy, exposure diagnostics, same resource telemetry và parity report.

## 7. Detailed architecture

PGSR native Gaussian Store → plane renderer → min-scale loss → normal-from-depth → single-view and neighbor roundtrip/NCC → exposure-aware RGB → Adam+Abs GS/trim → checkpoint. Adapter không thay representation bằng 2DGS.

## 8. Mathematical requirements

Eq.(1–4) flatten/plane-depth ratio; Eq.(5–11) normals/homography/NCC; Eq.(12–15) exposure/objective. Explicit signed-plane convention và dehomogenization; preserve matched normal/distance weights.

## 9. Implementation tasks

Audit chosen code revision vs paper; implement camera adapters/lazy loading; port Unix backup behavior safely; renderer buffers/gradients; epsilon-safe constant-gradient handling có versioned diff; native schedules/optimizer/ADC+exposure state; train-only neighbor graph; renderer export contract.

## 10. Ordered execution steps

1) Pin variant và resolved args. 2) Synthetic plane render/depth sign/normal tests. 3) Per loss gradient checks. 4) Enable single-view then multi-view gates on development schedule. 5) Exposure/resume smoke. 6) Real subset quality/VRAM report. 7) Research schedule only after gates.

## 11. Technical dependencies

PGSR CUDA extensions + existing torch; no extra pre trained normal model ở B2. Build/readme support không đủ chứng minh Windows backend. Source/native license/version recorded.

## 12. Hardware and resource constraints

One reference+one train neighbor, sample/chunk NCC patches, no all images-on GPU. Densification/re-render trim có peakcost; development vs research profile đã khai báo khác code defaults, cần report deviation.

## 13. Validation strategy

Fronto/slanted plane analytical depth; near-parallel invalid rays; direct 3D projection vs homography; identity warps và constant patch NCC; known occlusion; exposure identity/gates; resume restores app optimizer/geometry schedules.

## 14. Evaluation metrics

Per loss raw/weighted values, valid/occluded ratio, normal unit/orientation stats, depth quantiles, roundtrip error, Gaussian min scale/counts, RGB validation và VRAM/time. Chưa gọi dense accuracy nếu chỉ so SfM anchors.

## 15. Risks and mitigation

Paper/code weights/patch size khác; plane offset signs; opacity-normal cancellation; renderer gradient accidentally detached; held-out exposure fitting. Mitigate pinned variant, synthetic proofs, shape/gradient checks và canonical test appearance.

## 16. Acceptance criteria

CUDA buffers finite trên valid mask, depth semantics declared, projection parity, empty-mask behavior đúng, per loss gradients audited, no test neighbors, full state resume. B2 geometry chưa qua gate thì Ours chưa được bật.

## 17. Definition of Done

B2 adapter/tests/config/runtime evidence đủ tái lập đã chạy; all overrides documented. Paper results tách khỏi measured results; no claim paper reproduction nếu capped profile khác.

## 18. Next phase dependencies

Phase 04 dùng B2 rendered depth+normal; Phase 05 mở rộng **cùng B2** với λanchor=0 parity trước. Phase 06 dùng frozen variant và source SHA.

