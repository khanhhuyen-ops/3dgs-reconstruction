---
name: gaussian-training
description: Implement, debug, or validate Gaussian optimization, CUDA renderers, density control, checkpoints, and per-method training adapters for 3DGS, 2DGS, and PGSR in this repository. Use when training behavior is the task, with scope set by the current phase.
---

# Gaussian training

## Kích hoạt và context

Đọc AGENTS, [master plan](../../../docs/plans/master_plan.md), current Phase02/03/06, [training spec](../../../docs/tech/training_specification.md), [representation](../../../docs/tech/gaussian_representation.md) và [environment](../../../docs/tech/development_environment.md). Papers P02 §§4–6, P01 §§4–6, P14 §IV; [PGSR paper/code notes](../../../docs/research/pgsr_implementation_notes.md).

## Workflow

1. Inspect existing changes và determine method/variant, user scope, dataset/split/profile và native revision. Không coi yêu cầu tài liệu là yêu cầu train/cài dependency.
2. Qualify input và build environment riêng khi được yêu cầu; không đổi environment SfM đang dùng. Thiếu CUDA/backend → fail rõ, không CPU fallback.
3. Dùng adapter theo method, không tráo checkpoints vì cùng PLY extension. B0 original không thêm loss depth/exposure/AA mới; B2 code variant phải pin.
4. Real CUDA forward/backward trên fixture nhỏ trước scene subset. Verify tensor device, outputs, finite/nonzero relevant gradients; process trong nvidia-smi không đủ evidence.
5. Preserve native optimizer/ADC order và state surgery; guard transient memory trước densify. Lazy-load train images, neighbor train-only, no all-images GPU preload.
6. Save/reopen full optimizer/scheduler/RNG/ADC/exposure checkpoint. Test interruption/resume; inference PLY không đủ exact continuation.
7. Run đúng stage/profile được yêu cầu. OOM giữ checkpoint, log thay đổi và dùng versioned CUDA retry profile; comparison group phải cùng setting. Validation không fit exposure từ test RGB.

## Tools và commands

Phase03 actual adapter: [usage](../../../docs/tech/pgsr_training_usage.md), `src/pgsr` và `configs/training/pgsr_*.yaml`. Dùng `.tools/pgsr_env` riêng, native PGSR plane rasterizer/KNN; B0 checkpoints không tương thích. Smoke kích hoạt geometry20 để chạy thực các terms; research giữ native7000. Kiểm current run/evidence trước tuyên bố research complete. Kaggle helper riêng reuse sealed prepared `scene/`, không dùng B0 state; recovery/evaluation manifests tách biệt.

`rg`, Git diff/status, package/compiler metadata, official source và phase-specific tests. Existing `src.pipeline` chỉ có SfM stages; không gọi `--stage train` chưa triển khai. CUDA smoke/training commands chỉ chạy sau implementation tương ứng và trong user scope; record exact arguments/return codes.

## Validation, lỗi và hoàn thành

Kiểm tra Gaussian layouts/activations, quaternion normalization, backward, clone/split/prune optimizer shapes, empty masks, checkpoint resume và resource telemetry. Common failures: Python/CUDA ABI mismatch, covariance/plane semantics sai, detached gradients, transient OOM, held-out appearance leakage.

Done khi required tests chạy thật, checkpoint parse và CUDA small-run evidence rõ, artifacts/provenance có thật. Báo riêng unit/mock, CUDA smoke, full scene và final evaluation; không nâng một cấp evidence thành cấp khác.
