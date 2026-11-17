# Phase03 — PGSR CUDA training và Kaggle

Mốc17/11/2026: user đã yêu cầu triển khai và train Phase03. Adapter, environment riêng, CUDA numerical tests, real-data smoke80→resume300 và inference PLY reload đã chạy trên Windows. **Research30000, final evaluation và Blender PGSR còn pending Kaggle**. [Evidence local](../phase_03_local_validation.json) tách rõ các mức kiểm chứng.

## Variant và dữ liệu

Native [PGSR commit de24f1a](https://github.com/zju3dv/PGSR/tree/de24f1a38b350387e8d8fe381b2cd70c1ae946e7) tại `external/PGSR`; plane rasterizer, simple-knn và GLM nằm trong commit. Không dùng renderer B0. Upstream license chỉ cho educational/research/non-profit, modified work phải giữ open-source; giữ attribution. [Dependency lock](../../configs/training/pgsr_dependency_lock.json).

Prepared `legacy_v1/ignatius-legacy-v1-6a3308c22df11122`:263ảnh1920×1080,197train/33val/33test,76.223 train-init points. Metadata SHA `7636dd218be5cdb1bfa10b76b59a40b2b72cfb6129b0bf2d8fac4283a767913b`, split hash `fbd4eb563ba6b5418165ce4d402e04ec123117f56b2fe0c601a35efaad3ba2b4`. Giữ K/Tcw/XYZ và identity world normalization; arbitrary monocular units. SfM ban đầu là transductive pose estimation được công bố ở Phase01. GT geometry không vào trainer. Legacy có lịch sử CPU BA; video vẫn `BLOCKED_GPU_BA`.

`src/pgsr/data.py` bypass native destructive reader để không ghi lại sparse PLY hoặc lọc thêm19.249 points. Native cameras/GaussianModel nhận frozen metadata/train-only points; W2C và resized focal parity được assert. Train UID0…196 index exposure rows, val/test không có row. Neighbors chỉ từ train theo native distance-first/angle gates. CPU image cache bounded, mỗi step chuyển reference+neighbor sang CUDA.

## Objective và overrides

Đây là **official-code guarded variant**, không là exact paper reproduction hoặc Ours. P14 §IV Eq.(1–4) p.5, Eq.(5–9) p.6, Eq.(10) p.7 và Eq.(11–15) p.8; [paper/code differences](../research/pgsr_implementation_notes.md).

Native TNT settings: scale100, single0.015, NCC0.15, geo0.03; radius3=7×7 patch, single/multi activation sau7000, exposure sau1000, opacity cull0.05, abs-grad threshold0.00015, native train visibility trim. App L1 chỉ dùng khi canonical `1-SSIM < 0.5`; SSIM luôn canonical. LNCC native dùng squared correlation.

Shared-resolution research giữ RGB1920×1080 (`resolution:1`) thay vì `-r2` của upstream TNT shell; `ncc_scale:1` giữ grayscale1920 thay vì upscale3840. Seed42/lazy cache là overrides công bố. Cap6M là native setting, không bảo đảm vừa VRAM. Không chạy upstream shell có lệnh xóa output.

`geometry.py` hardening: finite positive depth/nonzero normal/nonparallel masks, valid finite-difference stencil, constant-gradient guard, undefined homography/border/constant-NCC rejection và connected zero khi empty. Giữ valid-sample equations; support khác native unguarded code, B2/Ours sau này phải dùng cùng variant. Single-view giữ **full H×W mean** với invalid pixels đóng góp0; geo/NCC là mean của `detached_weight * residual` trên accepted samples, không chia weight sum. Native depth-normal đã nhân accumulated alpha, so với accumulated plane normal.

Native camera-facing accumulated `N` và positive offset `delta` thỏa `Nᵀx+delta=0`; axial `Z=delta/(-Nᵀray)`. Không normalize N trước depth/homography. `H=Kn(R-tNᵀ/delta)Kr⁻¹` đã kiểm direct projection. Renderer không trả alpha trực tiếp; validity từ finite plane support không là foreground mask hoặc calibrated confidence.

Order: render/loss/backward → screen-space stats → native AbsGS ADC → train visibility trim → opacity reset → Adam (trừ global final) → checkpoint. Final30000 có29999 Adam steps. Không dùng Gaussian PLY hoặc B0 state để resume PGSR.

## Environment và commands

Windows `.tools/pgsr_env` là venv riêng đọc compatible parent Python3.10.20 site packages; PGSR native extensions cài trong chính venv, không đổi B0/SfM. Torch2.4.1+cu121, CUDA12.1, MSVC14.29.30133, sm86 đã forward/backward thật. PyTorch3D0.7.7 commit8965341 cung cấp official pure-Torch CUDA transforms; unused C++ ops không build. Tracked algorithm source sạch; pip-generated `.egg-info` được ghi riêng.

```powershell
.\scripts\setup_pgsr.ps1
.\.tools\pgsr_env\Scripts\python.exe -m src.pgsr.runner --config configs/training/pgsr_smoke.yaml --run outputs/tanks_and_temples/Ignatius/research/B2_PGSR/local-smoke-v1 --stop-after 80
.\.tools\pgsr_env\Scripts\python.exe -m src.pgsr.runner --config configs/training/pgsr_smoke.yaml --run outputs/tanks_and_temples/Ignatius/research/B2_PGSR/local-smoke-v1 --resume
.\.tools\pgsr_env\Scripts\python.exe -m src.pgsr.runner --config configs/training/pgsr_smoke.yaml --run outputs/tanks_and_temples/Ignatius/research/B2_PGSR/local-smoke-v1 --stage evaluate
$env:PGSR_RUN_CUDA_TESTS='1'
.\.tools\pgsr_env\Scripts\python.exe -m pytest -q tests/test_pgsr.py
```

Các smoke commands đã chạy; dùng tên run mới nếu tái lập. `--dry-run` resolve config; `--stage preflight` chạy actual synthetic CUDA unit. [Smoke](../../configs/training/pgsr_smoke.yaml):8train/6000points/res8, geometry activation20, không là fair benchmark. [Low-memory](../../configs/training/pgsr_low_memory.yaml) và [research](../../configs/training/pgsr_research_quality.yaml) là experiments riêng, chưa khẳng định low-memory full scene đã train.

## Full state, inference và Phase04 contract

`pgsr-full-checkpoint-v1` chứa16 native capture entries,7 Adam groups, ADC/radii/abs-gradient accumulators, activeSH, exposure model+Adam, iteration/optimizer steps, Python/NumPy/TorchCPU/CUDA RNG và remaining sampler IDs. New checkpoint parse/finite-check trước atomic latest pointer; chỉ giữ2 files gần nhất trong chính run. Changed source/config/data hoặc hash sai bị từ chối. Cross-OS extension bytes có thể khác nhưng revision/Torch/CUDA/optimization identity phải khớp và CUDA fixture pass; atomics không cam kết bitwise cross-GPU.

Native PLY và float render reference được lưu trước khi process training thoát. Process inference riêng reload và so RGB/depth/normal tolerance1e-5, tránh hai Gaussian stores cạnh optimizer. Test RGB33views canonical, không fit exposure. `rendering_metrics.json` của adapter là float-render diagnostic. **Primary comparison với B0 phải dùng `rendering_metrics_png.json`** từ command dưới: cùng rounded uint8 PNG, same SSIM definitions đã so AST và locked LPIPS weights. Không dùng float diagnostic để so trực tiếp B0 PNG scores.

```powershell
.\.tools\pgsr_env\Scripts\python.exe scripts/pgsr/score_png.py --run outputs/tanks_and_temples/Ignatius/research/B2_PGSR/local-smoke-v1
```

Kaggle capsule version1 giữ evaluator float; sau import cần chạy PNG scorer local trên PNG thật đã tải, không train lại hoặc đổi optimizer. Source của scorer/hash và per-view prediction/reference hashes được ghi riêng. Smoke8train/res8 scores chỉ là chức năng evaluator, không baseline comparison.

`evaluation/geometry_contract.json`: mỗi train-view NPZ có `depth[H,W] float32` axial Z, `normal[H,W,3] float32` unit camera frame, `valid[H,W] bool`, K và Tcw tại resolution export. Invalid0/maskFalse; normal chỉ normalize sau rendering. Quantiles/valid ratios/unit-normal error là diagnostics, không dense geometry accuracy. Phase03 không tạo mesh benchmark hoặc Ours.

## Kaggle Run All

[Private notebook](https://www.kaggle.com/code/vthkhnhhuyn/phase03-pgsr-run-all), [source notebook](../../notebooks/kaggle/phase03_pgsr_run_all.ipynb), [private source dataset](https://www.kaggle.com/datasets/vthkhnhhuyn/ignatius-phase03-pgsr-source-v1). Notebook12sections hash-pin source capsule, reuse existing Phase02 input và chỉ extract/hash `scene/`; không upload lại ảnh hoặc nạp B0 checkpoint.

`scripts/kaggle_pgsr/setup_linux.sh` tạo isolated micromamba Python/CUDA/GCC environment và PGSR native builds, chỉ cuda:0. Numerical/unit + real smoke/resume/reload gates chạy trước research. Train budget18000s dành600s cuối để checkpoint/export; setup/qualification tính ngoài training segment nên session cần thời gian thêm. OOM/resource failure giữ checkpoint, không âm thầm đổi resolution.

`phase03_results/recovery/` sealed trước evaluation, chỉ latest full checkpoint. Final RGB/geometry nằm ở `phase03_results/evaluation/`, sealed riêng và tham chiếu recovery manifest SHA. Evaluator lỗi không làm mất checkpoint. Partial dưới30000 yêu cầu session mới: attach previous notebook output làm Kernel Source, đặt `PREVIOUS` tới `recovery/`, giữ cùng source capsule/config. Output budget18GiB, không nhân đôi multi-GB checkpoint.

## Blender

Blender đã save copy rồi đóng trước CUDA optimization để giải phóng VRAM; không cài lại Blender/addon. B0 review blend và collections giữ nguyên. PGSR chỉ thêm vào scene/collection riêng sau khi validate artifact. Full XYZ/bounded Gaussian preview không thay native render. Hiện Blender PGSR review chưa chạy; Phase04 chỉ nhận handoff sau final validation, chưa thực thi TSDF/mesh.
