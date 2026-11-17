# Phase02 — chuyển checkpoint sang Kaggle và nhận kết quả

**Kết quả17/11/2026:** notebook train/evaluate đã `COMPLETE`; final bundle145 files đã xác minh và `IMPORTED_FINAL` vào run `b0-kaggle-c55f001d26`, checkpoint30000/3.381.717 Gaussians. Actual native PLY CUDA inference trên Windows khớp PNG Kaggle ở test view1; đã tạo Blender review riêng. Xem [result evidence](../phase_02_kaggle_result_import.json) và [Phase03 entry](../phase_03_entry_readiness.md). Các ghi chú chưa training ở mốc publish16/11 bên dưới là historical evidence.

Notebook đã được upload Private vào tài khoản `vthkhnhhuyn`: [mở editor để Run All](https://www.kaggle.com/code/vthkhnhhuyn/phase02-ignatius-run-all/edit). GPU T4, Internet và [Private Dataset](https://www.kaggle.com/datasets/vthkhnhhuyn/ignatius-phase02-3dgs-resume-input-v1) được cấu hình sẵn. Không cần sửa `INPUT`; notebook tự xác minh archive SHA256 và tìm sealed manifest. Bản source local: [phase02_resume_3dgs.ipynb](../../notebooks/kaggle/phase02_resume_3dgs.ipynb). [Verification report](../phase_02_kaggle_verification.json) ghi evidence local ban đầu; kết quả publish/runtime mới được ghi riêng, không thay lịch sử.

Notebook chính được Quick Save bằng `kaggle kernels push --no-run`; chỉ bắt đầu training khi bấm **Run All**. Phiên [CUDA build/checkpoint verification](https://www.kaggle.com/code/vthkhnhhuyn/phase02-cuda-build-and-checkpoint-verification) chạy riêng, không huấn luyện scene. Kaggle tự giải nén ZIP lồng nhau nên dataset dùng ZIP bytes mang đuôi `.bundle`. Notebook kiểm tra transport hash trước khi giải nén vào `/tmp`, giữ nguyên `source.zip` và toàn bộ sealed input. File upload nằm trong `.tools/kaggle_publish/dataset/`; dữ liệu và credential không commit.

**Đã kiểm chứng trực tiếp trên Kaggle:** verification version3 `COMPLETE`, Python3.10.20/Torch2.4.1+cu121, rasterizer và simple-knn build từ đúng SHA; CUDA unit forward/backward/optimizer update thành công; actual checkpoint7000 với2.233.886 Gaussians restore thành công trên Tesla T4. [Publish/runtime evidence](../phase_02_kaggle_publish.json) lưu environment, hashes, migration và scope. Main notebook saved version1, dataset version2; editor mới đã kiểm tra `Internet on`, `GPU T4 x2`, `Files only`, draft session OFF. Không dùng draft cũ `phase02-original3dgs-resume-run-all` có lỗi ZIP. Full scene training30000 và metrics Kaggle chưa chạy.

Tại kiểm tra ngày 16/11/2026, API trả `timeUsed` khoảng 5394 giây và `totalTimeAllowed` 21600 giây, còn editor hiển thị quota 30 giờ; hai giao diện khác nhau nên không dùng phép trừ này để khẳng định quota tuần còn lại. `SESSION_BUDGET_SECONDS` mặc định 3,5 giờ tính từ cell đầu; cell training dành 20 phút cuối để export/evaluation, gửi SIGINT đến chính process training khi hết budget, đợi process dừng rồi export checkpoint đã publish gần nhất. Không reset optimizer hoặc đổi schedule 30000. Phiên partial được ghi `PARTIAL_SESSION_BUDGET`, không coi là hoàn tất Phase02. Sau khi chạy cần **Save Version có outputs**; quota và tốc độ thực tế có thể buộc resume ở kỳ sau theo mục 3.

Run full-resolution local đã dừng `BLOCKED_RESOURCE` tại iteration7401, không phải completed30000. Checkpoint được công bố gần nhất là7000, SHA256 `bd31ba8fb327a4d767204dd860d9102bb933be8d61edfc338ca30da6aa2911d4`. Không dùng log7400/filename7500 làm checkpoint. Low-memory30000 đã hoàn tất độc lập. Script luôn đọc published `checkpoints/latest.json`, kiểm tra SHA/schema/state rồi chọn file thực tế; không hardcode7000 trong notebook.

## 1. Chuẩn bị trên Windows

Từ repository root, dùng environment đã có; không cần tiếp tục training local:

```powershell
.\.tools\b0_env\Scripts\python.exe scripts/kaggle/prepare_kaggle_input.py --run outputs/tanks_and_temples/Ignatius/research/B0_3DGS/ignatius-legacy-b0-research-quality_20261115T222504Z_9ddaaa --destination outputs/kaggle_handoff/ignatius_phase02_input_v1
```

Destination phải mới; script từ chối overwrite. Bundle đã tạo thật ở đường dẫn trên. File verification bên cạnh bundle ghi manifest SHA256: giữ giá trị này để đối chiếu sau upload. CPU inspection kiểm tra full native capture, Adam moments/steps/shapes, Gaussian tensors, SH degree, spatial extent, ADC/radii/accumulators, RNG và camera sampler. Đây là kiểm tra serialization/state; CUDA restore thật được thực hiện ở cell5 trên Kaggle.

Bundle gồm `scene/`, `checkpoint/`, `config/`, `source.zip`, `manifest.json`. Scene gồm263 images, native sparse model/mirror và các metadata/observations/qualification mà frozen prepared integrity contract yêu cầu; không có video hoặc ground-truth geometry. Checkpoint là full optimization state, khác Gaussian PLY. Source capsule chỉ chứa source/config/helper cần chạy, không chứa papers, outputs, Windows binaries, environments hoặc credential. Manifest giữ dataset/split/source/checkpoint hashes. Code Gaussian đang chạy giữ nguyên, code identity `f48fe731bf951ff77afe60cdb6b7425f7ba74ac4b173ca484e815463fad3ee7a`; source capsule có thêm các helper và SHA riêng, ghi base Git commit rõ ràng.

## 2. Upload Private Dataset và mở Notebook

1. Với bundle hiện tại, dùng link editor đã cấu hình ở đầu tài liệu. Khi tự publish lại, dùng Kaggle CLI đã đăng nhập, giữ dataset Private. Đóng gói sealed input thành ZIP rồi đổi đuôi `.bundle` để Kaggle không tự giải nén `source.zip` bên trong; cập nhật expected transport/manifest SHA trong notebook nếu bundle thay đổi.
2. Import notebook từ GitHub hiện tại hoặc upload file `.ipynb` trong repository nếu cần notebook khác. Attach đúng Private Dataset.
3. Bật Internet để tải đúng dependency/source/metric weights. Chọn GPU (ưu tiên T4 nếu tài khoản được cấp), kiểm tra `nvidia-smi`; một process dùng `cuda:0`, không gộp VRAM hai T4.
4. Notebook đã publish tự tìm `INPUT`. Chỉ sửa `WORK`, `OUTPUT`, `SESSION_TARGET`, `PREVIOUS` và session budget khi thực hiện một phiên khác. Template cấu hình: [kaggle_handoff.json](../../configs/training/kaggle_handoff.json). `WORK` mặc định ở `/tmp`, output ở `/kaggle/working`; bản dataset upload luôn read-only, phần giải nén là derived copy được xác minh.
5. Chạy cell1–5. So sánh manifest SHA với report Windows. Dừng nếu checksum/data/native source/build/restore lỗi. Chỉ dùng checkpoint của chính dự án vì Torch checkpoint chứa pickle.

Setup tạo environment riêng Python3.10.20/Torch2.4.1+cu121/TorchVision0.19.1+cu121, toolkit12.1.1 và GCC12. CUDA binaries build lại trên Linux theo native SHA/submodule pins; host Torch/driver của Kaggle được giữ nguyên. Đây là thay đổi OS/compiler/build đã công bố, không tùy tiện dùng Torch host khác version. [Micromamba official setup](https://mamba.readthedocs.io/en/latest/installation/micromamba-installation.html), [NVIDIA CUDA12.1 Linux/Conda guide](https://docs.nvidia.com/cuda/archive/12.1.1/cuda-installation-guide-linux/index.html). Linux setup đã chạy thành công trong Kaggle verification version3; log tải về ở `.tools/kaggle_publish/verified_output/phase02_setup.log`. Phiên mới vẫn phải vượt setup/smoke/restore checks; không bỏ qua lỗi hoặc fallback CPU.

## 3. Resume và lưu phiên

Cell5 tạo run mới và **derived checkpoint**: giữ nguyên original checkpoint/input. Chỉ đổi identity của extension build và engineering config: run name, checkpoint interval1000, reserved-memory budget `min(12GiB,80% VRAM thực)`. Resolution1920×1080, SH3, means/poses, optimizer/LRs, loss0.2 SSIM, ADC/reset schedule, total30000 và split giữ nguyên. `migration.json` ghi parent SHA/identity, khác biệt config, runtime helper hashes và CUDA environment. Không xóa integrity guard hoặc sửa checkpoint gốc để bypass Windows/Linux hash mismatch.

Restore bằng native `GaussianModel.restore` và project checkpoint loader trên CUDA, kiểm tra tensors/Adam state rồi lưu `cuda_restore.json`. Không gọi initialization từ COLMAP trong nhánh resume. GPU0 RNG/sampler được giữ; CUDA atomic reductions và GPU architecture khác nhau không bảo đảm bitwise equality.

Cell6 là bước duy nhất khởi chạy training dài. `SESSION_TARGET` là iteration tuyệt đối: có thể chọn10000/15000/... lớn hơn checkpoint đã lưu và không quá30000. Full schedule luôn30000; chọn target ngắn chỉ tạo controlled session pause. Native runner dùng status lịch sử `INTERRUPTED_FOR_RESUME_TEST` cho pause; không coi pause là completed training. Save checkpoint mỗi1000 và tại target; giữ hai checkpoint mới nhất trong **run Kaggle mới**, không xóa checkpoint nguồn. Duration trong run là segment Kaggle hiện tại; không gọi đó là tổng thời gian từ iteration1.

Sau mỗi session, **Save Version có outputs** và kiểm tra output đã được lưu. `/kaggle/working` không tự bền vững sau mọi reset/timeout. Cell6 export recovery bundle trong `finally` nếu process vẫn hoạt động; hard termination có thể ngăn export. Chọn session target phù hợp thời gian/quota thực tế của tài khoản, không chờ sát timeout. [Kaggle Notebook docs](https://www.kaggle.com/docs/notebooks).

Để tiếp tục phiên sau: attach **original input dataset** và **output đã lưu của phiên trước**. Đặt `PREVIOUS` đến bundle `phase02_results` (hoặc final bundle). Helper xác minh toàn bộ output, same original-input SHA, dataset/split/source/config và chọn published checkpoint mới hơn original local. Chuẩn bị workspace mới. Không reset iteration, sampler hoặc optimizer. Nếu đã30000, chỉ cần import final result, không resume training thêm.

## 4. Evaluation, export và download

Sau completed30000, cell8 gọi evaluator hiện có trên33 test IDs, xuất rendered/reference/comparison PNG, PSNR/SSIM/AlexNet-v0.1 LPIPS per-view và aggregate, checkpoint/PLY validation, config/environment/resources/logs. LPIPS thiếu weights hoặc lỗi phải báo PARTIAL/null; importer final từ chối metric thiếu. Validation không fit test exposure. Camera coordinates vẫn canonical COLMAP và protocol fixed-pose transductive legacy được giữ nguyên.

Cell6 tạo recovery output trước evaluation. Cell9 xuất bundle có evaluation riêng tên `phase02_results_final`; chọn bundle này khi import final. Partial run dùng `phase02_results`. Outputs có hai checkpoint gần nhất, native Gaussian PLY, logs, manifest/config/migration/environment, render/metrics khi final. Export kiểm tra dung lượng đĩa và budget output18GiB; không upload environments/cache/setup vào bundle. Nếu output budget không đủ, giữ recovery bundle đã có và lưu một phiên riêng, không xóa input.

Save Version rồi tải thư mục bundle từ Output tab; hoặc dùng Kaggle CLI đã đăng nhập:

```powershell
kaggle kernels output OWNER/NOTEBOOK-SLUG -p outputs/kaggle_download
```

Không gửi token trong chat hoặc đặt token trong repository. [Official output command](https://github.com/Kaggle/kaggle-cli/blob/main/docs/kernels.md#kaggle-kernels-output).

## 5. Import về local và Blender

```powershell
.\.tools\b0_env\Scripts\python.exe scripts/kaggle/import_kaggle_results.py --results outputs/kaggle_download/phase02_results_final
```

Importer verify checksums, local parent optimization config, dataset/split, full checkpoint/iteration, metrics33 held-out IDs và finite values, native PLY schema/count/hash/roundtrip evidence. Final và partial được ghi `IMPORTED_FINAL`/`IMPORTED_PARTIAL`, không gắn final cho checkpoint chưa hoàn thành. Tên run riêng dưới output root, không overwrite run cũ. Reimport cùng bundle trả về run hiện có sau recheck, không nhân bản. Conflicting destination hoặc file đã bị sửa phải fail.

`kaggle_import.json` và `blender_handoff.json` ghi receipt, final Gaussian PLY và local comparison candidates. Trong Blender, dùng addon Gaussian Splatting đã cài để import PLY ở handoff; giữ object transform identity để so sánh cùng COLMAP coordinates. PLY là splats, không triangle mesh. Blender/MCP không bắt buộc cho import artifacts; task này không cài addon hoặc thay scene đang có. Sau download/import thật cần native CUDA reload/render validation local nếu muốn xác nhận backend Windows với kết quả Linux; CPU importer không giả vờ đã thực hiện bước đó.

## 6. Kiểm chứng và giới hạn

Packaging/checkpoint inspection trên dữ liệu thật và tests bằng fixtures được báo riêng trong verification JSON lịch sử. [Publish/runtime evidence mới](../phase_02_kaggle_publish.json) ghi Linux solve/build, CUDA fixture và actual full checkpoint restore trên Kaggle, cùng7 tests local. Hai session-budget test dùng process mô phỏng; không là evidence training thật. Notebook chính chưa launch training; không có final metrics Kaggle, chưa import kết quả training mới hoặc triển khai Phase03.
