# B0 Original3DGS — thực thi Phase02

Kaggle migration: [setup, derived-checkpoint resume, export/import guide](kaggle_phase02_usage.md). Original checkpoints remain immutable; Linux build identities and resource settings have a separate audit. Kaggle execution is unverified.

Adapter nằm tại `src/gaussian_splatting/`, RGB evaluator tại `src/evaluation/`. Source chính thức tách ở ignored `external/gaussian-splatting`; phiên bản/weights được pin trong [dependency lock](../../configs/training/dependency_lock.json). Không có PGSR/2DGS/GOF training hoặc mesh extraction trong adapter này. Kết quả đo: [Phase02 report](../phase_02_implementation_report.md).

## Environment

```powershell
# Chạy từ repository root. Nếu có ns_3dgs tương thích, script tự tìm base Python.
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/setup_3dgs.ps1

# Hoặc chỉ rõ một Python3.10 có torch2.4.1+cu121 đang hoạt động:
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/setup_3dgs.ps1 -BasePython '<compatible-python.exe>'

.\.tools\b0_env\Scripts\python.exe -m src.gaussian_splatting.runner --config configs/training/b0_smoke.yaml --stage preflight
```

Setup tạo venv project-local với `--system-site-packages`: đọc các dependency đã có từ parent environment tương thích, nhưng cài pytest/extensions vào `.tools/b0_env`, không sửa parent/SfM/base environment. Đây là overlay được ghi rõ, không phải standalone Conda export. Parent versions phải khớp lock; `.tools/b0_build_logs/requirements-resolved.txt` giữ snapshot packages thực tế. Nếu máy khác chưa có parent tương thích, tạo môi trường Python3.10 riêng và cài đúng Torch/TorchVision CUDA12.1 cùng dependencies trong lock, rồi truyền `-BasePython`; không dùng Python3.14/SfM env để thay thế.

Script clone/pin official commit `414b553ef1d3032ea634b7b7a1d121173f36592c` nếu chưa có; chỉ initialize rasterizer, KNN và GLM submodules, không viewer. Existing checkout khác revision bị từ chối, không ghi đè. Build dùng CUDA12.1, MSVC14.29.30133 x64, Ninja, `TORCH_CUDA_ARCH_LIST=8.6`, `MAX_JOBS=2`; chỉ set environment cho process. `-SkipBuild` kiểm tra CUDA/Python và dependency setup mà không rebuild extensions. Không đổi NVIDIA driver, toolkit toàn máy hoặc môi trường đang làm SfM.

Runtime kiểm tra actual checkout/submodule SHAs, native source changes, installed extension paths/hashes và torch2.4.1+cu121. DLL search cho Windows chỉ bổ sung process-local parent `Library/bin` và `DLLs`. Rasterizer/KNN phải ở project venv; CUDA unavailable/mismatch → dừng, không CPU fallback.

## Profiles và commands

| Profile | Resolution | SH max | Train views / initialization | Iterations | Density/resource changes |
| --- | --- | ---: | --- | ---: | --- |
| [Smoke](../../configs/training/b0_smoke.yaml) | 1/4, legacy480×270 | 2 | 12 deterministic train views /20.000 seeded source points | 1.200 | start50, interval50, end700, reset250; cap80.000; chỉ functionality |
| [Low memory](../../configs/training/b0_low_memory.yaml) | 1/4, legacy480×270 | 3 | all197 train views /all76.223 train-init points | 30.000 | Native schedule; cap250.000; lazy CPU camera cache256 |
| [Research quality](../../configs/training/b0_research_quality.yaml) | 1, legacy1920×1080 | 3 | all197 views /all76.223 points | 30.000 | Native schedule, no Gaussian cap; feasibility phải đo riêng trên6GB |

Smaller profiles không là paper-quality equivalent. Seed42, black background, sourceRGB trên CPU, photometric loss coefficient0.2 và optimizer defaults được lưu trong fully resolved config mỗi run. Random camera sampling là stack without replacement của native method; train-view subset chỉ chọn từ train IDs sau frozen split. Smoke vẫn dùng nguồn point colors đã được Phase01 tính từ197 train views, không phải colors chỉ từ12 optimization views. Point subsampling chọn những points có thật, lưu IDs; không sinh points giả.

```powershell
.\.tools\b0_env\Scripts\python.exe -m src.gaussian_splatting.runner --config configs/training/b0_smoke.yaml --stage dataset
.\.tools\b0_env\Scripts\python.exe -m src.gaussian_splatting.runner --config configs/training/b0_smoke.yaml --stage initialize
.\.tools\b0_env\Scripts\python.exe -m src.gaussian_splatting.runner --config configs/training/b0_smoke.yaml --stop-after 550

# Dùng run path thật do launcher in ra; không trỏ tới dataset hoặc run khác config.
.\.tools\b0_env\Scripts\python.exe -m src.gaussian_splatting.runner --config configs/training/b0_smoke.yaml --resume '<smoke-run>'
.\.tools\b0_env\Scripts\python.exe -m src.gaussian_splatting.runner --config configs/training/b0_low_memory.yaml

.\.tools\b0_env\Scripts\python.exe -m src.gaussian_splatting.runner --config configs/training/b0_low_memory.yaml --stage render --run '<trained-run>' --split test
.\.tools\b0_env\Scripts\python.exe -m src.gaussian_splatting.runner --config configs/training/b0_low_memory.yaml --stage evaluate --run '<trained-run>' --split test
```

`--stage evaluate` tự render nếu chưa có bundle; `render` riêng hữu ích khi chỉ cần ảnh. `--stop-after` là controlled interruption ở planned iterations không đổi total schedule/config; run status `INTERRUPTED_FOR_RESUME_TEST`, không gọi complete training. `--resume` yêu cầu checkpoint/config/data/split/native/code identity khớp và tiếp tục sau iteration đã lưu; không nới total iterations ngầm. Các file checkpoint/export/metrics cũ không overwrite. Mỗi run mới có UTCtimestamp+UUID; logs dùng thời gian host, metadata có UTC và display timezone `Asia/Saigon`.

## Data integration và leakage boundaries

Input mặc định là prepared-v1 legacy experiment đã được user cho phép, không là video model. `verify_dataset()` kiểm tra thực tế artifact hashes, geometry/splits và reciprocal train observations trước mỗi stage cần dataset. Actual unmodified native `readColmapSceneInfo(eval=False)` đọc camera/point cloud; adapter map names theo pinned2023 basename/stem semantics, sau đó chọn manifest train/val/test. Không dùng `--eval` native để thay split197/33/33.

Native camera `R` là transpose của COLMAP Rcw theo contract native; `T` giữ tcw. Tcw/Twc/K/size và train-only point positions/colors được đối chiếu metadata. Native `loadCam` chỉ decode ảnh khi cần, resize bằng cùng dimensions thực; principal point centered trong scene này. Alpha/undistortion validity mask hoặc principal point lệch tâm chưa có native loss adapter: dừng có lỗi, không bỏ qua.

Original image data ở CPU, camera view/projection matrices và Gaussian/loss tensors ở CUDA0. CPU camera cache có bound; low-memory profile giữ tối đa256 cameras ở resolution nhỏ, không preload ảnh lên GPU. Evaluation dùng cùng resize/background/color convention. Canonical SfM coordinates không bị normalize lần nữa: native NeRF++ radius từ **selected train camera centers** chỉ định spatial LR và ADC extent, translation không áp vào points/poses. Project reversible normalization vẫn ở metadata, chưa applied; dataset/raw/poses/SfM outputs giữ nguyên.

Fixed-pose **transductive SfM** được công bố: source BA đã dùng held-out images trước Phase01. Downstream gradient targets/sampler/initial appearance chỉ lấy train data; điều này không biến source geometry thành strict train-only reconstruction. Ground-truth geometry không đi vào training/evaluation RGB.

## Native algorithm và engineering boundaries

P02 *3D Gaussian Splatting for Real-Time Radiance Field Rendering*, §4 Eq.(3–6): mean, anisotropic covariance `Σ=R diag(s²) Rᵀ`, quaternion normalized, projected covariance và ordered alpha compositing. Native CUDA renderer thực hiện projection/SH/covariance, tile16×16, sort theo tile/depth và front-to-back compositing; backward tới geometry/opacity/SH. Không CPU point renderer, không detached render để train.

P02 §5.1 Eq.(7): `0.8*mean(abs(render-target)) + 0.2*(1-SSIM)`. Wrapper gọi nguyên `utils.loss_utils.l1_loss/ssim`, FP32; SSIM window11/σ1.5/zero padding5 theo selected code. Chỉ RGB objective, không depth/normal/planarity/PGSR loss, exposure fitting hoặc anti-aliasing additions. Official main ghi các bổ sung2024; revision2023 được pin để tách chúng khỏi B0.

Native `scene/gaussian_model.py` khởi tạo means từ source points; DC `(RGB-0.5)/0.28209479177387814`, higher SH0; identity quaternion; opacity sigmoid⁻¹(0.1); log-scales từ `sqrt(clamp(distCUDA2,1e-7))`, với CUDA3-nearest-neighbor squared-distance mean. Paper mô tả nearest-neighbor covariance initialization; nêu đúng code squared-distance semantics thay vì gọi hai biểu thức là identical. Higher SH band được activate mỗi1.000 iterations theo native train.py, không bật full SH ngay từ iteration1.

Native Adam giữ groups/learning rates: positions0.00016→0.0000016 exponential với native spatial extent; feature0.0025/DC, rest÷20; opacity0.05; scale0.005; rotation0.001; eps1e-15. Native ADC warmup500/end15.000/interval100, opacity reset3.000, gradient threshold0.0002, min-opacity0.005, world/view size rules giữ nguyên cho reference schedule. Clone/split/prune gọi native implementation; lightweight wrappers chỉ đếm actual count transitions. Clone trong selected code copy chính Gaussian; paper mô tả directional movement nhưng không tự thêm behavior không có trong native code.

**ADC chạy trước Adam**, giữ đúng selected train.py; native bỏ Adam ở final iteration. Vì thế30.000 completed iterations tương ứng29.999 Adam steps, checkpoint/log ghi cả hai. Final state/export vẫn là actual optimized parameters; không gọi30.000 Adam updates. `max_gaussians` là resource variant: nếu upper projected final count vượt cap thì defer nguyên ADC event, log `SKIPPED_RESOURCE_CAP`; không đổi threshold âm thầm hoặc prune points tùy tiện. Reference profile null cap không áp rule này. Opacity reset vẫn theo schedule.

Paper §7.1 còn mô tả resolution warmup/upscaling; selected public native code dùng resolution option cố định. Presets này là **official-code variants**, không claim khôi phục nguyên paper's training environment hoặc warmup implementation. Research profile full resolution cũng chưa tự chứng minh equivalent benchmark protocol.

## Checkpoint, artifacts và resource telemetry

```text
outputs/.../research/B0_3DGS/<run>/
  config.yaml / resolved_config.json / run_manifest.json / environment.json
  dataset_integration.json / data_readiness.json / initial_point_ids.npy
  initialization.json / training.jsonl / density.jsonl / validation_*.json
  checkpoints/iteration_*.pth / latest.json
  point_cloud/iteration_*/point_cloud.ply / validation.json
  renders/<split>/iteration_*/{*_render.png,*_reference.png,*_comparison.png,manifest.json}
  metrics/<split>_<iteration>.json
  training_result.json / run.log / invocation_*.json / resume_*.json
```

Checkpoint chứa native capture (all model/Adam/ADC/max-radii/spatial-LR/active-SH state), iteration/Adam count, Python/NumPy/TorchCPU/all CUDA RNG và remaining camera-sampler stack. Scheduler được dựng lại từ cùng locked options và iteration. Save tới filename mới, reopen kiểm tra rồi atomic update pointer/hash. Restore/integrity/row surgery/RNG được test; inference PLY không thay checkpoint. CUDA atomic reductions có thể nondeterministic dù RNG đầy đủ, không claim bitwise resume parity trên mọi hardware.

Gaussian PLY có XYZ/SH/log-scales/rotation/opacity logits và zero normals; native reload rồi render cùng validation camera phải khớp trong tolerance1e-5. PLY này **không là triangle mesh**. Validation dùng val IDs; primary test evaluation chỉ sau completed configuration/checkpoint freeze. Không dùng test score để chọn checkpoint/hyperparameters.

Log allocated/reserved/peak theo process PyTorch riêng với sampled device-wide VRAM0.5s. Resume tạo process/segment mới, nên peak fields trong `training_result.memory` thuộc process hiện tại; dùng history/logs khi tổng hợp nhiều segments. Duration training segment gồm optimizer loop, scheduled validation/checkpoints và final export; dataset qualification/build nằm ngoài segment. Peak không đồng nghĩa tổng tensor bytes vì allocations bên ngoài Torch có thể khác.

OOM hoặc measured reserved budget5GiB vượt ngưỡng → failure report có iteration/error/memory/last checkpoint; giữ run cũ, không CPU fallback và không retry cùng settings vô hạn. Retry phải có config/profile mới ghi deviations, không ghép thành full reference run. Checkpoint files có thể lớn; dung lượng đĩa cần đủ cho model/optimizer states.

## RGB metrics và validation

PSNR per-view rồi mean arithmetic; MSE0 xuất JSON `null`+`psnr_infinite:true`, không invented finite cap. SSIM dùng pinned native implementation. LPIPS theo project spec **AlexNet v0.1**, input `[0,1]→[-1,1]`, CUDA; khác choice VGG của selected upstream metrics.py và được ghi rõ trong report. Hai weight files tải từ official TorchVision/RichZhang sources, cache project-local `.tools/b0_weights`; full SHA-256 được lock và kiểm tra trước reuse. Không weights/LPIPS không chạy được → null+reason/PARTIAL, không ghi0 giả.

Metrics tính từ frozen rounded8-bit PNG bundles để tái tính đúng từ saved artifacts, cùng target resize và sRGB range; không optimized test exposure. Bundle lưu IDs/names/hashes/sizes/resolution/checkpoint+split hashes, missing/corrupt views không bị loại ngầm. Per-view và aggregate có provenance. Comparison PNG là reference trái/render phải, không chỉnh riêng crop/contrast.

```powershell
.\.tools\b0_env\Scripts\python.exe -m pytest -q
python -m ruff check src tests
```

CUDA kernel unit fixture được đánh dấu synthetic-only; real SceneInfo/init/training/checkpoint/render/metrics evidence giữ riêng ở run reports. Native-related tests cần project Python; chạy bằng base Python khác version sẽ skip thay vì gọi CUDA runtime đã verified. B0 export/checkpoint không tự tương thích PGSR; Phase03 phải có adapter/checkpoint contract riêng, không reuse renderer state chỉ vì cùng PLY extension.

## Troubleshooting và verification

Các dòng dưới là hướng dẫn chẩn đoán, không phải lời khẳng định mọi lỗi đã xảy ra trong phiên này.

| Triệu chứng | Căn cứ cần kiểm tra | Khắc phục / xác minh lại |
| --- | --- | --- |
| CUDA unavailable hoặc extension không import được | Project Python, torch version/runtime, DLL search, extension path/hash, actual CUDA tensor test | Chạy setup/preflight đúng venv; không sửa base environment hoặc fallback CPU; kernel fixture phải chạy thật |
| Build/toolkit/compiler/architecture mismatch | nvcc12.1, MSVC14.29 x64, sm86, build stderr/exit trong `.tools/b0_build_logs` | Dùng compatible toolset đã chọn; không dùng `allow-unsupported-compiler` hoặc globally upgrade driver/toolkit để che lỗi; rebuild và CUDA forward/backward test |
| Camera model, pose, image-size mismatch | Native R transpose/Tcw, metadata K/Tcw/Twc, centered principal point, source/target dimensions, masks | Dừng ở dataset gate, sửa adapter/derived experiment có provenance; không sửa poses/model gốc; chạy dataset/initialization tests lại |
| Missing image / empty cloud | Prepared integrity hashes, native point count/PLY names/finite attributes và accepted train subset | Khôi phục đúng immutable source hoặc tạo prepared experiment mới; không sinh camera/point/image giả |
| OOM hoặc growth vượt budget | `failure.json`, exact iteration, allocated/reserved/peak, density transitions, latest checkpoint hash | Giữ checkpoint/run, chọn explicit lower-memory config và comparison label mới; CUDA retry có thay đổi cụ thể, không lặp nguyên failing config |
| NaN/Inf loss/gradients/scales | `training.jsonl`, last valid checkpoint, activated/log parameter ranges, SH/rotation/opacity, native renderer debug | Dừng trước publication; dùng preflight/backward fixture và real initialization để khoanh vùng; không `nan_to_num` loss hoặc chuyển CPU để gọi pass |
| Black render, wrong colors hoặc severe pose/FoV artifacts | Loaded camera matrix/device, visible radii, source point positions, sigmoid opacity, exp scales, SH activation/order, expected RGB range và comparison PNG | So sánh initial/train/held-out view cùng ID; reload PLY/native render parity; không đổi camera convention, exposure hoặc background để làm đẹp test score |
| Checkpoint mismatch / wrong continuation | Config/dataset/split/native/code identity, iteration/Adam counter, RNG và remaining sampler IDs, file checksum | Dùng exact latest compatible checkpoint; không nạp PLY rồi tự coi là resume hoặc sửa identity fields để vượt guard |
| LPIPS unavailable / checksum mismatch | Locked official weights, CUDA, cached file hashes, `lpips_unavailable_reason` | Không dùng random network hoặc ghi0; tải lại đúng nguồn tới cache mới có hash hợp lệ rồi đánh giá frozen bundle; metrics thiếu giữ PARTIAL |

Floaters/blur ở smoke hoặc low-resolution/capped run là quality limitation cần báo, không tự chứng minh implementation sai hoặc scene đã có mesh chính xác. Số đo held-out RGB và visual examples không thay thế geometry evaluation của phase sau.
