# Phase02 — Báo cáo triển khai và thực nghiệm B0 Original3DGS

**Cập nhật17/11/2026:** Kaggle full-resolution đã hoàn tất30000 iterations và được `IMPORTED_FINAL`,3.381.717 Gaussians;33 test views PSNR20,913033 /SSIM0,695186 /LPIPS0,269434. Đã kiểm tra full native PLY inference trên Windows CUDA và tạo Blender review riêng. [Result evidence](phase_02_kaggle_result_import.json), [Blender và Phase03 entry](phase_03_entry_readiness.md). Đoạn mô tả ngày16/11 bên dưới là lịch sử trước khi nhận kết quả Kaggle.

Ngày16/11/2026, timezone `Asia/Saigon`. **B0 low-memory đã hoàn tất training30.000 iterations và evaluation33 held-out views trên CUDA.** Smoke/resume đã đạt; run full-resolution đã dừng bởi resource guard tại7401, checkpoint7000 được giữ, chưa được tính là kết quả hoàn chỉnh. Đây là kết quả local trên legacy experiment đã được user cho phép, không là kết quả video GPU-strict hoặc tái lập benchmark của paper. Không chạy PGSR/2DGS/GOF/meshing.

Các số đo dưới đây lấy từ artifacts thực; [validation summary](phase_02_validation_summary.json) giữ bản JSON, [usage](tech/b0_training_usage.md) giữ contract và lệnh tái lập. Source SfM, raw, camera poses và prepared data được giữ nguyên.

## A. Implementation

| Thành phần | Trách nhiệm đã triển khai |
| --- | --- |
| `src/gaussian_splatting/native.py` | Verify pinned official checkout/submodules, local CUDA extensions và runtime; process-local Windows DLL search |
| `config.py`, `configs/training/` | Ba profiles có fully resolved config; bắt buộc `cuda:0`, không CPU fallback |
| `data.py` | Actual unmodified native `readColmapSceneInfo`, camera/point checks, adapter frozen split, lazy CPU camera LRU |
| `validation.py` | Gaussian parameters/activations/gradients/optimizer-state checks; synthetic CUDA unit fixture tách khỏi scene experiment |
| `checkpoint.py` | Native state, iteration/optimizer count, RNG/sampler, integrity hashes; checkpoint mới + atomic latest pointer |
| `runner.py` | Preflight/data/init/train/resume/render/evaluate CLI; native RGB objective, Adam, ADC, resource guards, PLY export/reload |
| `src/evaluation/` | Held-out render bundles, matched references/comparisons, PSNR/SSIM/LPIPS, per-view và aggregate metadata |
| `scripts/setup_3dgs.ps1` | Project-local environment và official CUDA rasterizer/KNN build |
| `scripts/plot_b0_run.py`, `summarize_b0_run.py` | Figure từ logs thực và tổng hợp evidence/timing với giới hạn interrupted segments |
| `tests/test_gaussian_baseline.py` | Config/metric, real CUDA gradients/Adam, native density state surgery và full checkpoint recovery |

Official repository [graphdeco-inria/gaussian-splatting](https://github.com/graphdeco-inria/gaussian-splatting/tree/414b553ef1d3032ea634b7b7a1d121173f36592c), commit `414b553ef1d3032ea634b7b7a1d121173f36592c`, được giữ riêng trong ignored `external/gaussian-splatting`, không sửa source. Rasterizer `59f5f77e3ddbac3ed9db93ec2cfe99ed6c5d121d`, KNN `44f764299fa305faf6ec5ebd99939e0508331503`, GLM `5c46b9c07008ae65cb81ab79cd677ecc1934b903`; viewer không initialized. [Dependency lock](../configs/training/dependency_lock.json) lưu full pins/weight hashes.

P02 *3D Gaussian Splatting for Real-Time Radiance Field Rendering* §§4–6, Eq.(3–7), là căn cứ representation, alpha compositing, loss và density control; [official project/paper](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/). Adapter gọi native implementation, không tự viết renderer thay thế. B0 chỉ dùng `0.8*L1 + 0.2*(1-SSIM)`, không thêm geometry/depth/normal/exposure/AA. Native ADC chạy **trước Adam** và final iteration không gọi Adam;30.000 iterations tương ứng29.999 optimizer-step invocations. Native clone/split/prune và optimizer-state surgery được giữ nguyên.

Giữ khác biệt paper/code minh bạch: P02 §7.1 mô tả resolution warmup nhưng selected public implementation dùng resolution option cố định; native clone copy center thay vì tự bổ sung dịch chuyển từ mô tả paper. Low-memory cap250k defer nguyên ADC event có log, là resource variant được khai báo. Full-resolution profile không có Gaussian cap. Không gộp các profiles thành một thí nghiệm tương đương.

Git HEAD lúc bắt đầu `c44cb3049bfefa0201bd5ee9936b0fd56185360f`. Own training/evaluation code identity `f48fe731bf951ff77afe60cdb6b7425f7ba74ac4b173ca484e815463fad3ee7a`; working code chưa commit có ZIP+SHA ở `.tools/b0_build_logs/project_source_snapshot_20261116.*` (18 files, archive SHA256 `708029b06dfb2845815e1bf45577ff690db7a44702998f9f83215681d434a960`). Snapshot mới có research CPU cache64 đúng resolved config; snapshot cũ được giữ riêng. Các file thay đổi gồm modules/configs/scripts/tests trên và README, docs index, Phase02/master plans, B0 method contract, training/representation/loss/reproducibility/architecture docs, report và environment/validation JSON. Không sửa implementation/config/tests SfM hoặc báo cáo lịch sử Phase01.

## B. Environment

| Thành phần | Đã xác minh |
| --- | --- |
| OS / GPU | Windows; NVIDIA GeForce RTX3050 6GB Laptop, compute capability8.6, device0 |
| CUDA VRAM | `torch.cuda` báo6.441.926.656bytes; `nvidia-smi`6144MiB |
| Python / Torch / TorchVision | 3.10.20 /2.4.1+cu121 /0.19.1+cu121 |
| Driver / runtime / toolkit | 610.88 /CUDA12.1 /12.1.66 |
| Compiler | MSVC14.29.30133 x64, Windows SDK10.0.26100, Ninja1.13.2 |
| Build flags | `TORCH_CUDA_ARCH_LIST=8.6`, `MAX_JOBS=2`, `DISTUTILS_USE_SDK=1` |
| Environment | `.tools/b0_env`, overlay `--system-site-packages` trên parent compatible chỉ đọc |
| Renderer/KNN | Native CUDA `.pyd` build và forward/backward/initialization chạy thật |

[Environment evidence](phase_02_environment.json) lưu extension hashes và actual six-parameter gradient norms. CUDA tensor forward/backward và native renderer fixture đều có finite nonzero gradients, Adam đổi parameters. Fixture6 Gaussians chỉ là unit evidence; kết quả tái tạo scene lấy từ các run bên dưới. Parent/SfM environment không bị cài đè; image decode/I/O/cache/orchestration ở CPU, Gaussian optimization/render/LPIPS ở CUDA0.

Build issues đã xử lý: chọn MSVC14.29 tương thích CUDA12.1 thay toolset14.51 mặc định; process-local DLL search cho Conda parent; PowerShell here-string cho Python và xử lý native stderr notice dựa trên actual exit code. Build logs và resolved packages ở `.tools/b0_build_logs/`, không có global driver/toolkit replacement.

## C. Training

Input immutable: `datasets/tanks_and_temples/Ignatius/prepared/legacy_v1/ignatius-legacy-v1-6a3308c22df11122`. Có263 ảnh1920×1080, frozen split197 train /33 val /33 test,76.223 train-only initialization points và434.720 train observations. Metadata SHA256 `7636dd218be5cdb1bfa10b76b59a40b2b72cfb6129b0bf2d8fac4283a767913b`; split hash `fbd4eb563ba6b5418165ce4d402e04ec123117f56b2fe0c601a35efaad3ba2b4`.

Legacy source có historical CPU BA (`ba_use_gpu=false`), được dùng trong experiment riêng theo xác nhận user. Không đổi input video mặc định. Protocol **fixed-pose transductive SfM**: source BA từng truy cập held-out images; downstream training loss/sampler/initial appearance chỉ dùng train data. Không gọi đây là strict train-only source reconstruction. GT geometry không được đọc hoặc truyền vào training.

Native loader đã đối chiếu Tcw/Twc/K/sizes/XYZ/colors và frozen IDs. Native NeRF radius6.318395233154297 chỉ điều khiển spatial LR/ADC; không áp translation hoặc normalize XYZ/poses lần nữa. Project normalization metadata giữ unapplied.

| Số đo local | Smoke | Low-memory |
| --- | ---: | ---: |
| Resolution | 480×270 | 480×270 |
| Optimization train views | 12 | 197 |
| Initial / final Gaussians |20.000 /78.582 |76.223 /249.213 |
| SH max |2 |3 |
| Completed iterations |1.200 |30.000 |
| Adam-step invocations |1.199 |29.999 |
| Final sampled train loss |0.0776845887 |0.0588055737 |
| Observed loop/validation/checkpoint/export wall seconds |37.9185474 |**≥5955.7864887** |
| Max recorded Torch peak allocated |105.0625MiB |378.8926MiB |
| Max recorded Torch peak reserved |130.0000MiB |826MiB |
| PLY reload render max-abs difference |0 |0 |

Loss là một sampled training view ở iteration cuối, không phải mean test loss. Low-memory duration là **lower bound**: lần tiếp tục chỉ còn checkpoint25.000, process cũ không chạy và không có clean final timing record. Segment đầu có elapsed log cuối5784.3765911s, segment resume25.001–30.000 ghi171.4098976s; nguyên nhân gián đoạn không được ghi nhận. Tổng loại offline time giữa processes và input qualification/build; không dùng171s làm toàn bộ training time, không suy ra FPS từ duration này. `execution_summary.json` lưu cách cộng và giới hạn.

Low-memory completed6 density events, deferred138 events theo cap đã khai báo; cloned74.908, splitparents100.877, children201.754, pruned2.795, opacity reset4 lần. Cap ảnh hưởng thuật toán ADC và phải giữ khi so sánh cùng profile. Device-wide sampled peak524MiB/316 samples thuộc **segment resume cuối**, có thể gồm process khác; không thay số này cho peak toàn run. Torch peaks là maximum của records qua segments, không phải total CUDA/process peak chính xác.

[Training curves từ actual logs](../outputs/tanks_and_temples/Ignatius/research/B0_3DGS/ignatius-legacy-b0-low-memory_20261115T141109Z_8ed1c6/training_curves.png). Các validation dips ở3000/6000/9000/12000 là diagnostic render ngay sau native opacity reset; không phải final held-out test scores. Memory-peak trace hạ tại resume25000 vì CUDA process peak counters bắt đầu lại; không kết luận model tự giảm memory từ đường này.

Smoke đã stop550→resume551–1200, model/optimizer/RNG/remaining sampler restore hợp lệ. Low-memory đã khôi phục checkpoint25.000 đúng code/config/data identity. CUDA atomic reductions có thể nondeterministic; không tuyên bố bitwise parity chỉ vì RNG state đầy đủ.

Full-resolution reference run `ignatius-legacy-b0-research-quality_20261115T222504Z_9ddaaa`:1920×1080,197 views,76.223 initial points,SH3,no cap,30.000 planned. Pilot500 iterations đạt, checkpoint/native PLY reload đạt; resumed501 để đo actual feasibility. Run dừng BLOCKED_RESOURCE ở7401 theo reserved-memory guard5GiB, không phải full completion hoặc kết luận mọi GPU6GB đều không chạy được. Checkpoint7000 đã kiểm tra và đóng gói; chưa có final test metrics.

## D. Evaluation

| Actual held-out test metrics,33 views | Smoke1200 | Low-memory30000 |
| --- | ---: | ---: |
| PSNR↑ |15.0263114036 |**23.3753929727** |
| SSIM↑ |0.4859111354 |**0.8399258148** |
| LPIPS↓ |0.4174079525 |**0.1150172877** |

Evaluation chỉ sau completion/frozen final checkpoint; không chọn checkpoint bằng test scores, không fit held-out exposure, không exclude ảnh thiếu âm thầm. Render từ reloaded native Gaussian PLY, matched frozen test IDs, correct480×270 size/black background/sRGB. Metrics tính trên frozen rounded8bit PNG predictions/references; mỗi ảnh có values, rồi arithmetic mean33 ảnh. SSIM gọi native implementation; LPIPS dùng bundled `lpipsPyTorch` **AlexNet v0.1 trên CUDA**, input `[0,1]→[-1,1]`, hai official-source weights đã SHA256-lock. Đây là project metric contract; upstream `metrics.py` dùng VGG, vì vậy không gọi scores tương đương upstream paper table.

All33 rendered views và references tồn tại, finite, đúng ID/resolution. Identity/perturbed metric fixture chạy thật, không mock LPIPS. Comparison ảnh000001 đã xem trực tiếp: low-memory tái tạo đúng statue/garden và tốt hơn smoke, còn blur ở vegetation/ground và view-dependent errors. Visual observation không thay quantitative geometry evaluation. Không đo Chamfer/F-score hoặc dựng mesh trong Phase02.

## E. Artifacts và commands

Output root: `outputs/tanks_and_temples/Ignatius/research/B0_3DGS/`.

| Run | Thư mục thực |
| --- | --- |
| Init-only | `ignatius-legacy-b0-smoke_20261115T133608Z_4ef4c4`; `initialize_result.json`SUCCESS, không là trained model |
| Smoke completed | `ignatius-legacy-b0-smoke_20261115T134942Z_346824` |
| Low-memory completed | `ignatius-legacy-b0-low-memory_20261115T141109Z_8ed1c6` |
| Full-resolution reference | `ignatius-legacy-b0-research-quality_20261115T222504Z_9ddaaa` |

Trong low-memory run:

- `checkpoints/iteration_030000.pth`, `latest.json`: full optimization/RNG/sampler checkpoint, khác inference PLY.
- `point_cloud/iteration_030000/point_cloud.ply`, `validation.json`:249.213 native Gaussian splats, reload/render parity0; SHA256 `e92ad58640f8d6fcdca7b3d755e0faa3561b2c5f5d3a8a00623b53990447af48`.
- `renders/test/iteration_030000/`:33 render/reference/comparison triplets và manifest/hash IDs.
- `metrics/test_030000.json`: per-view và aggregate actual scores.
- `training.jsonl`, `density.jsonl`, `validation_*.json`, `run.log`: training/resources/ADC/validation histories.
- `config.yaml`, `resolved_config.json`, `run_manifest.json`, `environment.json`, `dataset_integration.json`, `data_readiness.json`, `invocation_*.json`, `resume_025000.json`: provenance/config/commands.
- `training_result.json`, `execution_summary.json`, `training_curves.png`: final training status, corrected timing evidence và figure từ logs.

Smoke có layout tương tự, final iteration001200 và `resume_000550.json`. Gaussian PLY có native XYZ/SH/log-scales/quaternion/opacity logits, **không là triangle mesh**.

Các lệnh đã chạy bằng `.tools/b0_env/Scripts/python.exe`, từ repository root:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/setup_3dgs.ps1
.\.tools\b0_env\Scripts\python.exe -m src.gaussian_splatting.runner --config configs/training/b0_smoke.yaml --stage preflight
.\.tools\b0_env\Scripts\python.exe -m src.gaussian_splatting.runner --config configs/training/b0_smoke.yaml --stage dataset
.\.tools\b0_env\Scripts\python.exe -m src.gaussian_splatting.runner --config configs/training/b0_smoke.yaml --stage initialize
.\.tools\b0_env\Scripts\python.exe -m src.gaussian_splatting.runner --config configs/training/b0_smoke.yaml --stop-after 550
.\.tools\b0_env\Scripts\python.exe -m src.gaussian_splatting.runner --config configs/training/b0_smoke.yaml --resume outputs/tanks_and_temples/Ignatius/research/B0_3DGS/ignatius-legacy-b0-smoke_20261115T134942Z_346824
.\.tools\b0_env\Scripts\python.exe -m src.gaussian_splatting.runner --config configs/training/b0_smoke.yaml --stage evaluate --run outputs/tanks_and_temples/Ignatius/research/B0_3DGS/ignatius-legacy-b0-smoke_20261115T134942Z_346824 --split test
.\.tools\b0_env\Scripts\python.exe -m src.gaussian_splatting.runner --config configs/training/b0_low_memory.yaml
.\.tools\b0_env\Scripts\python.exe -m src.gaussian_splatting.runner --config configs/training/b0_low_memory.yaml --resume outputs/tanks_and_temples/Ignatius/research/B0_3DGS/ignatius-legacy-b0-low-memory_20261115T141109Z_8ed1c6
.\.tools\b0_env\Scripts\python.exe -m src.gaussian_splatting.runner --config configs/training/b0_low_memory.yaml --stage evaluate --run outputs/tanks_and_temples/Ignatius/research/B0_3DGS/ignatius-legacy-b0-low-memory_20261115T141109Z_8ed1c6 --split test
.\.tools\b0_env\Scripts\python.exe -m src.gaussian_splatting.runner --config configs/training/b0_research_quality.yaml --stop-after 500
.\.tools\b0_env\Scripts\python.exe -m src.gaussian_splatting.runner --config configs/training/b0_research_quality.yaml --resume outputs/tanks_and_temples/Ignatius/research/B0_3DGS/ignatius-legacy-b0-research-quality_20261115T222504Z_9ddaaa
.\.tools\b0_env\Scripts\python.exe scripts/plot_b0_run.py --run outputs/tanks_and_temples/Ignatius/research/B0_3DGS/ignatius-legacy-b0-low-memory_20261115T141109Z_8ed1c6
.\.tools\b0_env\Scripts\python.exe scripts/summarize_b0_run.py --run outputs/tanks_and_temples/Ignatius/research/B0_3DGS/ignatius-legacy-b0-low-memory_20261115T141109Z_8ed1c6
.\.tools\b0_env\Scripts\python.exe -m pytest -q
```

## F. Testing và acceptance gates

Full repository test suite ngày16/11: **62 passed,0 failed,1 skipped**,55.66s. Skip là Windows symlink privilege `WinError1314`, không phải bypass CUDA. Config/PSNR/native SSIM/loss gradients, actual CUDA renderer với gradients của6 groups, clone/split/prune Adam moment shapes, checkpoint checksum/overwrite protection, native restore/RNG/camera sampler đều được kiểm tra. Ruff src/tests/relevant scripts đạt ở lần kiểm tra implementation.

Final artifact recheck: smoke và low-memory latest checkpoint SHA256 khớp; từng run có đủ33 finite/nonblack480×270 RGB predictions và native PLY roundtrip difference0. Ruff cho new source/test/scripts và `git diff --check` đạt. JSON Phase02/dependency lock parse được;138 relative documentation links được kiểm tra, ba link `.agent/skills/preprocessing/` thiếu từ trước không bị sửa trong task Phase02, không có broken link mới.

Real-data evidence tách khỏi unit fixture: actual native loader/initialization/backward; smoke1200+resume+33 held-out metrics; full low-memory30000+checkpoint recovery+PLY native roundtrip+33 held-out metrics. Không dùng synthetic fixture để thay scene training. Full-resolution reference chỉ được ghi theo tiến trình thật.

Protected-file checksum baseline1603 existing source/tests/data-config/dataset/original-output files: **changed0, missing0** ở lần đối chiếu16/11. Chỉ hash original database bytes, không mở SQLite; không đọc GT contents. Evidence `.tools/b0_build_logs/protected_before.json` / `protected_after.json`.

| Acceptance | Evidence / trạng thái |
| --- | --- |
| CUDA environment + rasterizer/KNN build/run |PASS, exact versions/hashes và CUDA unit/real scene |
| Actual prepared dataset/cameras/split |PASS trên legacy separate experiment |
| Init/render/backward/optimizer/ADC |PASS, native modules và finite checks |
| Full checkpoints/resume |PASS, smoke550 và low-memory25000 recovery |
| Planned low-memory training |PASS,30000 iterations/29999 Adam calls |
| Gaussian PLY reload + held-out renders/metrics |PASS,33 test views, all three metrics measured |
| Logs/resources/provenance |PASS, interrupted timing được đánh dấu lower bound |
| Full-resolution uncapped reference |BLOCKED_RESOURCE ở7401; [Kaggle handoff](tech/kaggle_phase02_usage.md) chuẩn bị, chưa train Kaggle |
| Default video sparse model/GPU BA |BLOCKED, ngoài scope Phase02 legacy |

## G. Problems và giới hạn

1. Video vẫn `BLOCKED_GPU_BA`, không có sparse model; legacy source không thay thế kết quả video. Historical CPU BA và transductive geometry được công bố.
2. Low-memory resolution/cap khác uncapped reference; không phải benchmark reproduction, không suy ra SOTA hoặc geometric quality từ PSNR. Chưa có multiseed study/comparison group của Phase06.
3. Full-resolution đã dừng theo resource guard tại7401;5GiB Torch reserved guard/CUDA OOM giữ checkpoint và báo lỗi thật, không CPU fallback. Chưa có final metrics nếu run chưa kết thúc.
4. Low-memory abrupt interruption làm mất exact total duration cuối segment đầu; báo observed lower bound, không bịa timing. Device-wide sample không phải exact process peak; current-segment result không thay all-segment maximum.
5. CUDA atomic reductions cho phép nondeterminism; checkpoint lưu đầy đủ RNG không cam kết bitwise reproducibility mọi GPU.
6. Adapter hiện hỗ trợ scene pinhole centered principal point này. Alpha/validity masks hoặc off-center principal point cần adapter rõ ràng; pipeline từ chối thay vì bỏ qua. Không có camera refinement/exposure optimization.
7. PGSR reader/filter/PLY-write differences từ Phase01 vẫn cần method-specific adapter khi Phase03 được yêu cầu. Vanilla Gaussian checkpoint không tự tương thích PGSR; native Gaussian PLY không chứa triangle faces/depth semantics của surface model.

## H. Next Phase

B0 low-memory đã tạo đủ CUDA training/checkpoint/render/evaluation infrastructure và actual frozen artifacts cho Phase03 entry. Có thể reuse prepared dataset/split/provenance, environment audit, experiment directories, resource logging, metric definitions và evaluator khi protocol/resolution khớp. PGSR phải pin native version, verify riêng camera/initial-point filtering, plane/depth/normal/appearance conventions, renderer gradients và full optimizer state; không chuyển checkpoint bằng cách đổi filename.

Phase03 chưa được bắt đầu. Không dựng mesh/TSDF hoặc train thêm baseline ngoài phạm vi Phase02.

## I. Kaggle publish và runtime verification — 16/11/2026

Đã upload Private Dataset version2 và notebook [Phase02 Ignatius Run All](https://www.kaggle.com/code/vthkhnhhuyn/phase02-ignatius-run-all/edit), GPU T4/Internet/Files only cấu hình sẵn. Kaggle tự giải nén ZIP lồng nhau nên đổi transport sang `.bundle` có SHA256; sealed manifest và original checkpoint không đổi. Notebook có session budget và recovery export trước timeout.

Verification notebook version3 COMPLETE: Python3.10.20/Torch2.4.1+cu121, official rasterizer/simple-knn build đúng native revisions; CUDA fixture forward/backward/optimizer update PASS; actual checkpoint7000/2.233.886 Gaussians restore trên Tesla T4 PASS. Đây là verification, chưa là full-resolution training hoặc final metrics. Local handoff7 tests PASS (hai case session-budget dùng process mô phỏng), notebook schema và server source roundtrip PASS. [Publish/runtime evidence](phase_02_kaggle_publish.json) lưu hashes, environment, migration và scope; logs/JSON tải thật ở `.tools/kaggle_publish/verified_output/`. Full scene training30000, held-out Kaggle metrics và final-result import vẫn chờ thực thi.
