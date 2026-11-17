# Training specification

Kaggle migration: [setup, derived-checkpoint resume, export/import guide](kaggle_phase02_usage.md). Original checkpoints remain immutable; Linux build identities and resource settings have a separate audit. Full-resolution B0 training30000, final import and one-view Windows CUDA PLY inference were verified on17/11/2026; see [result evidence](../phase_02_kaggle_result_import.json) and [Phase03 entry](../phase_03_entry_readiness.md). PGSR runtime remains unverified.

B0 Phase02 đã có implementation tại `src/gaussian_splatting/` và đang thực thi theo request ngày15/11/2026; [actual usage/config/algorithm boundary](b0_training_usage.md). Các phần B1/B2/Ours và profiles nghiên cứu dưới đây vẫn là thiết kế cho phase tương ứng. [Architecture](../architecture/training_architecture.md), [data contract](data_specification.md), [representation](gaussian_representation.md) và [loss spec](loss_specification.md) là dependencies. Không thêm training keys vào `sfm_ignatius.yaml`; file đó chỉ thuộc SfM.

## B0 implementation contract

Pin official `414b553ef1d3032ea634b7b7a1d121173f36592c`, rasterizer `59f5f77`, KNN `44f7642`; Python3.10.20 / torch2.4.1+cu121 / CUDA12.1 / MSVC14.29 / sm86 đã có actual CUDA forward/backward evidence. Three runnable YAMLs ở `configs/training/b0_{smoke,low_memory,research_quality}.yaml` thay cho việc coi proposed profiles dưới đây là CLI đã chạy. Native `GaussianModel`, `loadCam`, `readColmapSceneInfo`, renderer, L1/SSIM và native ADC được gọi qua adapter, không port rasterizer sang Python/CPU.

B0 dùng `sparse/0/points3D.ply` train-only colors, explicit Phase01 split và bounded lazy CPU cameras; poses/XYZ canonical không normalize thêm. Native train extent chỉ scale position LR/ADC thresholds. Adam/SSIM/density defaults từ pinned source; reference native ADC chạy trước optimizer step và bỏ final Adam step (30k iterations/29.999 steps). Capped profiles defer whole ADC event có count/log; native reference profile không cap. Paper resolution warmup và newer upstream features không tự thêm vào selected code variant. Loss chỉ RGB; viewer/network GUI tắt.

Checkpoint native capture + RNG Python/NumPy/TorchCPU/CUDA + camera stack + source/config/native/extension/code hashes; atomic files và integrity pointer. Gaussian PLY export/reload/render parity là validation riêng, không đủ resume. Device/process VRAM peaks và sampled device-wide usage giữ scope khác nhau; resume process peak không tự là max mọi segments. Held-out evaluation sau checkpoint/config freeze; project AlexNet LPIPS weights có hash lock, không test exposure fitting. Chi tiết/report limits: [B0 usage](b0_training_usage.md).

## 1. Loop contract

1. Xác minh dataset/split/native revision và CUDA renderer; source thiếu qualified model thì dừng trước initialization.
2. Seed Python/NumPy/torch; native CUDA FP32 tensors trên `cuda:0`. Tải reference RGB lazily, neighbors chỉ từ train set.
3. Initialize Gaussian state từ accepted train points; preserve method activation/SH/layout. Native normalization phải compose với project normalization rõ ràng.
4. One-view differentiable render; B2/Ours thêm tối đa một neighbor mỗi iteration, sampled patches theo resource budget.
5. Tính scalar loss theo variant, log raw và weighted từng term, valid counts/coverage. Camera/GT tensors không có gradient.
6. Zero gradients, backward, assert finite loss/gradients và geometry gradients khác0 trên fixture không tối ưu; optimizer step theo đúng native ADC order.
7. Densify/prune chỉ theo schedule và capacity gate; update Adam state/gradient accumulators consistent new rows. Không prune bằng test metrics.
8. Log timing/VRAM và save checkpoint; validation no-grad. Giải phóng validation/neighbor graphs trước iteration sau.

Native optimizer/ADC order phải được pin và test; không tự reorder prune/step khi port Windows rồi gọi faithful baseline. Optional fused Adam/AA/depth-prior features trên official main không bật vào B0 mặc định.

## 2. Optimizer và scheduling

B2 official-code snapshot ở [notes](../research/pgsr_implementation_notes.md): Adam groups cho mean/SH/opacity/scale/rotation; position LR decay0.00016→0.0000016 trong30k; SH feature0.0025, opacity0.05, scale0.005, rotation0.001; native scene scaling factor phải ghi. Exposure LR0.001 theo P14 §V. Đây là baseline reference, không tuning result.

B0/B1 dùng native Adam schedule của revision được pin khi phase bắt đầu, ghi resolved parameter groups đầy đủ trước run; không kế thừa implicit PGSR loss/settings. SH degree tăng theo native schedule; count/head layout giữ trong checkpoint. Những hyperparameters chưa đủ evidence code ở task tài liệu không được tự gắn nhãn paper-default.

Research B2 geometry activation sau7000 theo official-code variant; Ours thêm anchor từ7000 và ramp2000 là proposed curriculum. B2/Ours cùng base settings, cùng camera sampling order khi so ablation. B3 schedule riêng, không load optimizer B2.

## 3. Hai configuration profiles (chỉ tài liệu)

Các giá trị sau là **preset thiết kế ban đầu cho project**, không đo fit VRAM hoặc tái hiện kết quả paper. Schema dự kiến cần loader/config riêng trong phase implementation; hiện `src.pipeline` chưa nhận các keys này.

```yaml
profiles:
  development:
    purpose: cuda_forward_backward_resume_smoke
    device: cuda:0
    allow_cpu_fallback: false
    max_width: 640
    reference_batch_size: 1
    train_view_limit: 12
    initialization_point_cap: 20000
    gaussian_hard_cap: 80000
    iterations: 500
    sh_degree_max: 1
    geometry_start_iteration: 100
    densify_from_iteration: 50
    densify_until_iteration: 300
    densify_interval: 50
    checkpoint_interval: 100
    validation_interval: 100
    multiview_neighbors_per_step: 1
    multiview_sample_count: 2048
    image_preload_on_gpu: false
    mixed_precision: false
    torch_reserved_budget_gib: 4.5
  research_quality:
    purpose: comparable_full_training_under_resource_budget
    device: cuda:0
    allow_cpu_fallback: false
    max_width: 1280
    reference_batch_size: 1
    train_view_limit: null
    initialization_point_cap: null
    gaussian_hard_cap: 500000
    iterations: 30000
    sh_degree_max: 3
    geometry_start_iteration: 7000
    densify_from_iteration: 500
    densify_until_iteration: 15000
    densify_interval: 100
    checkpoint_interval: 1000
    validation_interval: 1000
    multiview_neighbors_per_step: 1
    multiview_sample_count: 16384
    image_preload_on_gpu: false
    mixed_precision: false
    torch_reserved_budget_gib: 5.0
```

Development view subset là deterministic train subset **sau** frozen split, không tạo benchmark split mới ngầm. Research-quality không là exact-paper reproduction: resolution cap/Gaussian cap/patch sample budget đã thay resource regime. Nếu initial accepted points vượt cap thì `BLOCKED_RESOURCE` hoặc tạo declared subsampled variant; không xóa points khỏi dataset gốc. B0/B1 không có multi-view geometry nên bỏ qua keys không áp dụng bằng method schema rõ ràng, không vô tình bật PGSR terms.

B1/B2/Ours common comparison dùng cùng image resolution/data/profile; hard cap và resource decisions phải log giống nhau. Không resize ảnh on-the-fly mà quên K/image/anchor transform. Một variant full-paper không cap có thể yêu cầu GPU lớn hơn, ngoài feasibility đã xác minh.

## 4. Resource và OOM policy

GPU6GB có overhead display/driver/extension allocations; torch budget không bao gồm mọi CUDA allocation. Đo `max_memory_allocated`, `max_memory_reserved` theo process và sampled device-wide VRAM riêng; log sample interval. Count tensors/Adam state và transient densification allocations; không gọi `empty_cache` để khẳng định đủ VRAM.

Nếu OOM: checkpoint cuối hợp lệ giữ nguyên; ghi stage/iteration/counts/resolution/memory. Retry CUDA với config variant mới: giảm patch sample count, Gaussian cap rồi resolution theo nguyên nhân; áp dụng cùng setting cho comparison group. Không silently đổi setting trong một research run hoặc CPU fallback. Chưa chứng minh profile nào fit máy hiện tại; MILo/GVGS full experiments không thuộc cam kết6GB.

## 5. Checkpoints, validation và acceptance

Checkpoint chứa schema/native method, parent hashes, Gaussian tensors, Adam moments/steps, schedulers, active SH, ADC stats, exposure optimizer, RNG state và iteration. Save atomic tới new filename, reopen/validate tensor shapes/finiteness. Resume phải từ checkpoint đồng bộ (không optimizer iteration khác tensor snapshot), parent hashes khớp; load map-to-CPU chỉ deserialization, chuyển CUDA trước compute, không training CPU.

No-grad validation trên validation split; primary test render dùng canonical appearance, không optimize exposure từ test RGB. Save final hoặc validation-selected checkpoint theo preregistered rule; primary comparison dùng fixed30k final, early best chỉ báo phụ. Profile development chỉ chứng minh control flow/numerics, không quality.

Acceptance: real CUDA forward/backward, finite nonzero relevant gradients, density event và optimizer shape consistency, interruption/resume small-run parity trong tolerance đã ghi, checkpoint parse, separate RGB/geometry buffers, resource measurements. Không báo SUCCESS training chỉ vì có PLY hoặc loss giảm ở một ảnh.
