# Phase 02 — Original 3DGS baseline

Trạng thái ngày16/11/2026: **B0 LOW-MEMORY COMPLETE / UNCAPPED REFERENCE BLOCKED_RESOURCE / KAGGLE HANDOFF PREPARED**. Official2023 CUDA renderer/KNN, actual native loader/init/backward/ADC, smoke1.200 iterations, full-state resume, low-memory30.000 iterations và33 held-out test metrics đã chạy. Low-memory PSNR23,3754 /SSIM0,8399 /LPIPS0,1150; final249.213 Gaussians. Run dừng ở7401 bởi resource guard; checkpoint7000 đã kiểm tra. [Kaggle usage](../tech/kaggle_phase02_usage.md), [verification](../phase_02_kaggle_verification.json). Kaggle CUDA/train chưa chạy. [Report/acceptance](../phase_02_implementation_report.md), [usage/config](../tech/b0_training_usage.md), [dependency lock](../../configs/training/dependency_lock.json), [master plan](master_plan.md).

## 1. Phase overview

Xây B0 theo pinned official2023 Original3DGS sau Phase01, bằng `src/gaussian_splatting/` và minimal RGB evaluator `src/evaluation/`. Mesh extension không thuộc B0 native. Nguồn thực tế là prepared legacy experiment riêng, không giả video sparse model đã tồn tại.

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

Dataset adapter → native 3DGS Gaussian Store/init → CUDA tile rasterizer → RGB loss → native ADC/Adam → checkpoint/renderer. Đã triển khai `src/gaussian_splatting/` adapter, shared minimal RGB metrics ở `src/evaluation/`; third-party source ở `external/`, không vendor cả repo vào src.

## 8. Mathematical requirements

Positive covariance RSSᵀRᵀ, quaternion normalization, projected covariance, ordered alpha blending, SH và photometric Eq.(7). Không depth/normal/geometric loss ở B0.

## 9. Implementation tasks

Đã pin original-code2023 SHA414b553, tách source ở `external/`, build renderer/KNN project-local trên torch2.4.1+cu121/CUDA12.1/MSVC14.29; native initialization/optimizer/scheduler/ADC, RNG+sampler checkpoints, native PLY roundtrip và no-grad metrics đã có actual tests. Newer depth/AA/exposure features không có trong selected variant. [Usage](../tech/b0_training_usage.md) tách paper math, code behavior và resource/metric engineering decisions; low-memory/canonical profiles không bị gộp.

## 10. Ordered execution steps

Đã thực hiện: verify build/source hashes → isolated CUDA fixture → real native dataset/init/backward → smoke550 interruption/resume1200 →33 test metrics → low-memory30000/recovery25000 → native PLY reload/33 test metrics. Full-resolution uncapped được pilot500 rồi resume501 riêng sau khi các gates đạt. Logs/artifacts lưu immutable; không chọn profile bằng test scores.

## 11. Technical dependencies

PyTorch CUDA và native rasterizer/KNN extensions; build toolchain phải tương thích. LPIPS chỉ cài/download khi scope phase bao gồm, có source/hash rõ; thiếu metric dependency giữ null, không mock thành kết quả.

## 12. Hardware and resource constraints

RTX3050 khoảng6GB. Actual profiles: smoke480×270/80k cap; low-memory480×270/250k cap; reference1920×1080/no cap. Bản actual YAML supersede heuristic presets trước implementation; reference feasibility phải đo riêng. One image/step, CPU lazy cache có bound, image target chuyển CUDA theo step;5GiB Torch reserved guard/CUDA OOM fail closed. Low-memory max recorded Torch allocated378,8926MiB/reserved826MiB, là maximum của recorded process peaks qua segments, không total device peak.

## 13. Validation strategy

Actual CUDA tensor/device+renderer evidence; finite gradient check mean/scale/rotation/opacity/SH trên synthetic scene; clone/split/prune state consistency; resume parity small deterministic sequence; known RGB metric fixtures.

## 14. Evaluation metrics

Train/validation loss, held-out RGB khi authorized/frozen; Gaussian counts, stage wall time, peak allocated/reserved và sampled devicewide VRAM. Không gán paper Table 1 score cho local run.

## 15. Risks and mitigation

Toolchain mismatch Python 3.14/Toolkit 12.1/torchcu 128; high tile counts, OOM at ADC; model loaded as only PLY không đủ resume. Mitigate isolated build gate, resource profile và full state contract.

## 16. Acceptance criteria

Real CUDA forward/backward passes, non finite counts 0, at least one ADC event checked, checkpoint reopen/resume được, metadata/split/hash correct. RGB loss decrease là sanity check, không cam kết NVS quality.

## 17. Definition of Done

Code/config/tests/docs B0 chạy thật; low-memory training/held-out metrics/native PLY/full checkpoint recovery đủ evidence. Full suite62passed/1skipped (Windows symlink privilege),0failed. Evidence unit/CUDA/real scene và full-resolution reference được phân biệt trong report; không gọi Gaussian PLY là mesh hoặc local scores là paper benchmark. Reference run chưa kết thúc phải giữ trạng thái riêng, không báo full reference completion.

## 18. Next phase dependencies

Phase 03 reuse dataset/logging/state infrastructure, không reuse incompatible native checkpoint mặc định. Phase 06 reuse B0 artifacts nếu keys/profile/seed khớp, không train lại vô lý.

