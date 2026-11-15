# Phase 02 — Original 3DGS baseline

Trạng thái hiện tại: **PLANNED — chưa triển khai/chạy**. User request tiếp theo quyết định việc thực thi; xem [master plan](master_plan.md). Mọi thông số preset là thiết kế, không benchmark đã đo.

## 1. Phase overview

Xây B0 faithful Original 3DGS sau Phase 01; downstream code chưa có. Mesh extension không thuộc B0 native.

## 2. Research objective

Tái lập rendering pipeline và kiểm soát optimizer/ADC/checkpoint/resource trước geometry modifications.

## 3. Scientific references

P02 §§3–7 pp.4–10, Eq.(3–7); [training spec](../tech/training_specification.md); [representation](../tech/gaussian_representation.md).

## 4. Current dependencies

Qualified prepared dataset, frozen split; official code/renderer revision+submodules pinned; compatible isolated Windows/CUDA build; user đã yêu cầu phase implementation.

Handoff thực tế từ Phase01: [prepared-v1 usage](../tech/phase_01_usage.md), [report/paths/gates](../phase_01_implementation_report.md), `configs/data/ignatius_legacy.yaml` cho experiment riêng và `configs/data/ignatius.yaml` cho video còn blocked. Dùng `PreparedDataset`/frozen `splits.json`, train-only points/colors trong `sparse/0`; không dùng colors trong `qualified_model` hoặc native implicit splits. Actual native SceneInfo + `apply_native_split` phải được kiểm thử khi build dependencies; Phase01 chỉ kiểm tra native COLMAP-reader components, không gọi training-ready hoàn chỉnh.

## 5. Input requirements

Train RGB/cameras/points với train-only appearance policy; validation split; no test RGB trong loop; resolved profile và seed.

## 6. Expected outputs

B0 native checkpoints/full state; run metadata/logs; validation/test render interface; shared RGB metric validator; CUDA smoke/resource report. Không tạo fake mesh để đủ outputs.

## 7. Detailed architecture

Dataset adapter → native 3DGS Gaussian Store/init → CUDA tile rasterizer → RGB loss → Adam/ADC → checkpoint/renderer. Planned `src/gaussian_splatting/` adapter, shared minimal metrics ở `src/evaluation/`, không vendor cả repo vào src.

## 8. Mathematical requirements

Positive covariance RSSᵀRᵀ, quaternion normalization, projected covariance, ordered alpha blending, SH và photometric Eq.(7). Không depth/normal/geometric loss ở B0.

## 9. Implementation tasks

Pin original method variant và disable newer optional depth/AA/exposure features; build renderer in separate environment; initialize scales/colors from accepted points; native optimizer/scheduler/ADC; full checkpoint/resume; no-grad validation; implement shared PSNR/SSIM/LPIPS interface với weight provenance.

## 10. Ordered execution steps

1) Verify build/source hashes. 2) Synthetic CUDA forward/backward. 3) Small real train subset. 4) Development 500-step smoke có density event/resume. 5) Check metrics/VRAM. 6) Chỉ full research run khi gates đạt và phase request bao gồm run. Freeze outputs cho comparison về sau.

## 11. Technical dependencies

PyTorch CUDA và native rasterizer/KNN extensions; build toolchain phải tương thích. LPIPS chỉ cài/download khi scope phase bao gồm, có source/hash rõ; thiếu metric dependency giữ null, không mock thành kết quả.

## 12. Hardware and resource constraints

RTX3050 khoảng 6GB; development 640px/80k hard cap, research 1280px/500k cap là đề xuất chưa đo. One image/step, lazy cache; densification transient peak cần kiểm soát. Không CPU training/render fallback.

## 13. Validation strategy

Actual CUDA tensor/device+renderer evidence; finite gradient check mean/scale/rotation/opacity/SH trên synthetic scene; clone/split/prune state consistency; resume parity small deterministic sequence; known RGB metric fixtures.

## 14. Evaluation metrics

Train/validation loss, held-out RGB khi authorized/frozen; Gaussian counts, stage wall time, peak allocated/reserved và sampled devicewide VRAM. Không gán paper Table 1 score cho local run.

## 15. Risks and mitigation

Toolchain mismatch Python 3.14/Toolkit 12.1/torchcu 128; high tile counts, OOM at ADC; model loaded as only PLY không đủ resume. Mitigate isolated build gate, resource profile và full state contract.

## 16. Acceptance criteria

Real CUDA forward/backward passes, non finite counts 0, at least one ADC event checked, checkpoint reopen/resume được, metadata/split/hash correct. RGB loss decrease là sanity check, không cam kết NVS quality.

## 17. Definition of Done

Code/config/tests/docs cho B0 đủ chạy theo CLI sẽ được triển khai; evidence unit/CUDA/scene phân biệt rõ. Inference PLY/checkpoint not equal final mesh. Untested full-training paths ghi rõ.

## 18. Next phase dependencies

Phase 03 reuse dataset/logging/state infrastructure, không reuse incompatible native checkpoint mặc định. Phase 06 reuse B0 artifacts nếu keys/profile/seed khớp, không train lại vô lý.

