# Phase01 implementation report — 15/11/2026

**Trạng thái: IMPLEMENTED / acceptance PARTIAL.** Real legacy preparation, geometry validation, project loader và verified resume đã pass. Nhánh video vẫn `BLOCKED_INPUT`; full native scene loaders chưa chạy; PGSR cần adapter vì native filter/PLY writer thay đổi initialization. Chưa bắt đầu Phase02, Gaussian training, mesh extraction hoặc GPU benchmark. Không cài dependency hay chạy lại SfM.

Evidence machine-readable: [validation summary](phase_01_validation_summary.json). Git HEAD `74875ab6b2e3142bc55e672b39c0bbfc9072e236`; code/config/docs đang là working-tree changes, không commit. Implementation fingerprint và fully resolved config nằm trong dataset metadata và summary. [CLI/config/schema/troubleshooting](tech/phase_01_usage.md) là hợp đồng thực thi; reports ngày10/11 và14/11 vẫn giữ nguyên.

## A. Implementation summary

Pipeline thực tế: explicit model discovery/selection → BIN/TXT semantic comparison → camera/image/point/reciprocal-track qualification → derived image/keypoint undistortion → explicit whole-track filters → deterministic split → train-only points/colors/observations → statistics/normalization metadata → independent native-reader checks và readiness.

Parser không thay `src/geometry/sfm/model_io.py`. PyCOLMAP chưa cài; downstream reader mới đọc official little-endian/TXT và được đối chiếu bằng native reader độc lập. Camera/image/point IDs giữ nguyên; chỉ point2D indices là contiguous indices. Rig một camera được xác minh frame/image pose equivalence; rig nhiều sensor bị reject. Không lấy BA synthetic fixtures làm reconstruction thật.

CPU được dùng cho I/O, NumPy geometry diagnostics, hashing và Pillow image processing; mỗi lần decode/warp một ảnh, inverse warp theo tile64rows. Không có neural inference, CUDA training/rendering, CPU fallback từ GPU-required stage hoặc backend GPU mới được xác minh trong Phase01. CUDA evidence của SfM cũ vẫn là historical logs; legacy BA ghi `ba_use_gpu=false`.

## B. Files created/modified

| Nhóm | Files |
| --- | --- |
| Model/geometry validation | `src/data/__init__.py`, `colmap.py`, `geometry.py`, `validation.py` |
| Preparation/data contract | `src/data/config.py`, `splits.py`, `image_processing.py`, `training_data.py`, `preparation.py`, `loader.py` |
| CLI/compatibility | `src/data/cli.py`, `compatibility.py`, `validate_colmap.py`, `prepare_dataset.py`, `inspect_dataset.py`, `verify_compatibility.py` |
| Versioned configs/provenance | `configs/data/ignatius.yaml`, `ignatius_legacy.yaml`, `loader_references.json` |
| Automated tests | `tests/test_data_preparation.py` |
| Tài liệu mới | Report này, `docs/phase_01_validation_summary.json`, `docs/tech/phase_01_usage.md` |
| Tài liệu cập nhật có phạm vi | `README.md`, `docs/README.md`, `docs/architecture/{system_architecture,data_flow}.md`, `docs/tech/data_specification.md`, `docs/plans/{master_plan,phase_01_data_preparation,phase_02_3dgs_baseline}.md` |
| Artifacts ngoài Git | Dataset dẫn xuất, reports/logs mới trong `outputs/.../data_preparation/`, pinned native reader/source/license snapshots nhỏ trong `.tools/phase01_native_readers/` |

`src/geometry/sfm/`, preprocessing, common helpers, `src/pipeline.py`, SfM configs/scripts/tests và raw/model/output gốc không bị sửa. Đối chiếu SHA-256 của **1.276 files tồn tại trước task** không thấy file đổi hoặc mất, gồm source/config/tests cũ, original outputs và SQLite sidecars, raw video và ảnh/frame không phải GT. Ground-truth content không được đọc. Những Git changes/deletions có từ trước task được giữ nguyên; không restore/reset chúng.

## C. COLMAP audit và validation results

Prompt ban đầu ghi SfM hoàn tất; kiểm tra thực tế cho thấy chỉ legacy có scene model. User đã cho phép legacy thành thí nghiệm riêng, không thay nhánh video.

| Nguồn / vị trí từ repository root | Kết quả |
| --- | --- |
| `datasets/tanks_and_temples/Ignatius/raw/Ignatius.mp4` | Có thật, 3.067.060.079 bytes; hash nguồn không đổi |
| `.../preprocessing/extracted_frames/video_v0/`, `selected_frames/video_v0/` | 524 extracted / 96 selected theo manifests hiện có; không trích/chọn lại |
| `outputs/tanks_and_temples/Ignatius/sfm_colmap/video_v0/database.db` | Có thật, 77.455.360 bytes; features/matches đã có, **không sparse scene model** |
| `outputs/tanks_and_temples/Ignatius/sfm_colmap/database.db` | Database legacy 272.183.296 bytes; chỉ inventory/hash, không mở SQLite |
| `outputs/tanks_and_temples/Ignatius/sfm_colmap/sparse/0/` | BIN legacy, 263 registered views / 1 shared intrinsic camera / 85.656 points |
| `outputs/tanks_and_temples/Ignatius/sfm_colmap/txt/` | TXT của cùng model; exact semantic hashes camera/image/point/observation/track bằng BIN |
| `.../sfm_colmap/sparse.ply` | Parse thật đủ85.656 vertices; sparse point cloud, không phải triangle mesh |
| `.../video_v0/audits/...`, `video_smoke_v0/audits/...` | BA probe fixtures được nhận diện diagnostic-only, không chọn làm scene |

Legacy `project.ini` chỉ tới `raw/image_set`, `ba_use_gpu=false`, CERES. Feature backend lịch sử của legacy không được suy từ video GPU logs. BIN/TXT không phải hai reconstruction độc lập. Model/rigs/frames đều parse được; trivial1rig/263frames có pose equivalence. Toàn bộ263 ảnh nguồn decode được và khớp 1920×1080, không missing image hoặc filename collision; registered coverage263/263. Các image→camera associations cụ thể có trong machine-readable reports.

Camera nguồn `SIMPLE_RADIAL`: `f=1159.4932405028942`, `cx=960`, `cy=540`, `k=-0.028378193557882524`. Sau undistortion là `PINHOLE`, không dùng ảnh distorted cùng pinhole parameters.

Các số dưới đây là **đo local**, residual Euclidean trên từng observation, không paper/GT mesh accuracy:

| Tập kiểm tra | Points | Observations | Mean / median / p95 residual (px) | Mean / median track length |
| --- | ---: | ---: | --- | --- |
| Original legacy | 85.656 | 689.049 | 1,139587 / 0,978000 / 2,639945 | 8,044375 / 5 |
| Sau loại whole duplicate tracks, còn distorted | 81.714 | 599.714 | 1,118153 / 0,957559 / 2,600855 | 7,339183 / 5 |
| Accepted undistorted model | 79.928 | 583.184 | 1,134723 / 0,968808 / 2,651701 | 7,296367 / 5 |
| Train-only initialization | 76.223 | 434.720 | 1,136433 / 0,970088 / 2,653059 | 5,703265 / 4 |

Observed positive-depth fraction là1,0 ở các tập đã kiểm tra. Original tracks có **3.942 points chứa nhiều observations từ cùng image**; default reject, legacy explicit `drop_track` loại nguyên track ở derived copy. Sau đó **1.786 points / 2.111 observations** chạm ngoài prepared image bị loại theo explicit `outside_image_tracks: drop_track`. Diagnostic trước duplicate filtering đếm2.229 observations ngoài ảnh; hai số thuộc hai tập khác nhau. Thêm3.705 points không đủ hai train observations bị loại khỏi initialization; không xóa ảnh hoặc pose.

Không suy rằng duplicate finding chứng minh root cause model corruption. Không chọn một observation thuận tiện để giữ track; rejected IDs/reasons được lưu đầy đủ. Nonlinear warp thay hệ pixel residual nên số trước/sau undistortion không phải metric chất lượng trực tiếp tương đương. Residual lớn nhất sau warp4,287879px được giữ và cảnh báo, không âm thầm filter theo warning threshold4px; p95 vẫn vượt qua configured gate10px (project heuristic).

## D. Dataset output và geometry

Dataset đã công bố sau readiness, khoảng1,06GB:

`datasets/tanks_and_temples/Ignatius/prepared/legacy_v1/ignatius-legacy-v1-6a3308c22df11122/`

- [Metadata](../datasets/tanks_and_temples/Ignatius/prepared/legacy_v1/ignatius-legacy-v1-6a3308c22df11122/metadata.json), [splits](../datasets/tanks_and_temples/Ignatius/prepared/legacy_v1/ignatius-legacy-v1-6a3308c22df11122/splits.json), [readiness](../datasets/tanks_and_temples/Ignatius/prepared/legacy_v1/ignatius-legacy-v1-6a3308c22df11122/readiness.json).
- 263 PNG 1920×1080, không resize/crop; `.jpg→.png` có deterministic source-name mapping. Source pixels/files giữ nguyên. Inverse warp thực tế không tạo invalid border masks trong scene này.
- Split primary: **197 train / 33 validation / 33 test**, `k mod8 = 0` test, `=4` val; numeric-aware source names, seed42 được lưu. Image_set không có timestamps/video identity được chứng minh: ghi `null`.
- `qualified_model/` giữ79.928 accepted all-view points; không dùng màu từ đây để initialize Gaussian. `sparse/0/` và byte-identical `sparse/` mirror chứa76.223 train-init points, RGB tính lại từ train PNG, tracks và errors train-only. NPZ có434.720 train observation rows.
- XYZ và source image poses của accepted points giữ nguyên trong canonical COLMAP coordinates. [Actual geometry comparison](../outputs/tanks_and_temples/Ignatius/data_preparation/legacy_v1/geometry_comparison_ignatius-legacy-v1-6a3308c22df11122.json) cho maximum XYZ difference0, quaternion/translation difference0 và keypoint-transform difference0; camera IDs giữ nguyên, point associations chỉ clear, không reassignment.

Normalization lưu riêng, chưa áp vào model: train-camera median center `(0.3202536332, 0.1791153827, -0.0991930852)`, scale `0.1931932016`, inverse scale `5.1761655782`. `X'=s(X-c)`, `t'=s(t+Rc)` và inverse được kiểm thử. Units vẫn monocular arbitrary SfM units, không phải meters; không GT alignment.

Accepted bbox: `[-121.417521, -70.139557, -95.646301]` đến `[153.053025, 32.126706, 100.049275]`, diagonal extent352,263009; bounding-sphere radius195,233689. Đây là thống kê sparse scene, không object extent/accuracy đã kiểm chứng. Far/background points và scene ROI chưa được lọc/tune theo GT. Runtime467,521s là thời gian **tới metadata generation**, không toàn bộ final hashing/readiness và không GPU benchmark/peak-memory measurement.

## E. Commands, tests và kết quả

Các lệnh thực thi từ root:

```powershell
python -m src.data.validate_colmap --config configs/data/ignatius.yaml
python -m src.data.prepare_dataset --config configs/data/ignatius.yaml --dry-run
python -m src.data.validate_colmap --config configs/data/ignatius_legacy.yaml --discover
python -m src.data.prepare_dataset --config configs/data/ignatius_legacy.yaml --dry-run
python -m src.data.prepare_dataset --config configs/data/ignatius_legacy.yaml
python -m src.data.prepare_dataset --config configs/data/ignatius_legacy.yaml --resume
python -m src.data.inspect_dataset --config configs/data/ignatius_legacy.yaml
python -m src.data.verify_compatibility --config configs/data/ignatius_legacy.yaml --references .tools/phase01_native_readers
python -m pytest -q
python -m ruff check src tests
```

Video validation/preparation: `BLOCKED_INPUT` và nonzero exit, không synthetic scene. Legacy validation/dry-run/preparation/inspection/resume: `SUCCESS` hoặc `DRY_RUN` đúng scope. Compatibility command: **FAILED** do PGSR common-initialization mismatch; không che bằng success aggregate. Diagnostic riêng xác minh native filter semantics, xem mụcF.

- Automated suite cuối: **51 passed, 0 failed, 1 skipped**, 27,49s. Skip duy nhất: Windows symlink privilege `WinError1314`; copy/hardlink pass. Không coi skip là pass.
- Ruff toàn `src tests`: **All checks passed**. Existing17 tests và new34 passed/1skipped là test evidence mới của phiên này, không tái dùng số17 của báo cáo10/11.
- Unit fixtures chỉ kiểm tra parser/camera/projection/undistortion/resize/split/normalization/track/data-integrity/resume; không là benchmark thật. Actual original/derived legacy checks, geometry comparison và native readers ở mụcC–F là integration evidence riêng.
- Project lazy loader thực sự decode sample ở cả train/val/test: RGB `[3,1080,1920]`, float32 `[0,1]`, CPU; model/artifact checks bao phủ toàn bộ image files. Không preload images lên GPU.
- Completed resume kiểm tra source/config/relevant-code fingerprint và output hashes, không overwrite dataset; interrupted resume/cache tampering được kiểm tra bằng unit tests. Original input hashes không đổi.

Evidence files mới: [prepare](../outputs/tanks_and_temples/Ignatius/data_preparation/legacy_v1/prepare_20261115T043145Z_a6eeed96/report.json), [resume](../outputs/tanks_and_temples/Ignatius/data_preparation/legacy_v1/prepare_20261115T044134Z_74ffd791/report.json), [inspection](../outputs/tanks_and_temples/Ignatius/data_preparation/legacy_v1/inspect_20261115T044134Z_517afda9/report.json), [compatibility](../outputs/tanks_and_temples/Ignatius/data_preparation/legacy_v1/compatibility_20261115T044134Z_1c5e9879/report.json). Mỗi thư mục còn có exact invocation/resolved config và run log.

## F. Compatibility — verified và unverified

| Consumer | Pinned commit | Kết quả thật |
| --- | --- | --- |
| Project `PreparedDataset` | Working implementation fingerprint trong metadata | Readiness + train/val/test CPU samples pass; frozen manifest được áp dụng |
| Original3DGS reference loader components | `54c035f7834b564019656c3e3fcc3646292f727d` | Actual native binary camera/image/point reader pass: 263 views, 1 intrinsic, 76.223 points; values/IDs/observations exact |
| 2DGS reference loader components | `f3e3b9fa67bbd1c75e05167ff37391d8dab2a678` | Actual native binary reader pass cùng counts và exact values |
| PGSR reference loader components | `de24f1a38b350387e8d8fe381b2cd70c1ae946e7` | Camera/observations parity pass; native point parser tự giữ56.974 points, loại thêm19.249 theo `error<=2px AND track_length>=3`; common-initialization compatibility **ADAPTER_REQUIRED** |
| Full native scene loaders / rasterizers | Cả ba | **NOT_RUN**; thiếu `plyfile`, `diff_gaussian_rasterization`, `simple_knn` trong interpreter hiện tại; không dùng stubs |

[Pinned registry](../configs/data/loader_references.json) có file/source/license hashes, tác giả và URLs; chỉ download source snapshots nhỏ read-only, không clone/build/install các models. Snapshot Original3DGS main này là evidence input reader, không lời khẳng định training mới faithfully tái hiện model2023.

[PGSR diagnostic](../outputs/tanks_and_temples/Ignatius/data_preparation/legacy_v1/pgsr_adapter_diagnostic_ignatius-legacy-v1-6a3308c22df11122.json) xác minh retained XYZ/RGB đúng native filter, không lỗi BIN parsing. Full PGSR scene code còn **luôn ghi lại PLY** do unconditional conversion branch; PLY chuẩn có sẵn không ngăn được việc này. Vì chưa chạy full scene loader, prepared dataset chưa bị writer đó chạm vào. Phase02/03 cần isolate writable native staging và giữ common init/split tường minh; không sửa reported errors/tracks để ép native check pass hoặc tự giảm initialization của mọi baseline.

Native implicit holdout không có ba-way split như project. `apply_native_split` đã có để kiểm tra native camera coverage/init cloud rồi áp manifest, nhưng actual native SceneInfo integration còn untested. FoV-only readers cũng cần adapter nếu principal point lệch tâm hoặc có validity masks; legacy hiện centered và không có invalid border masks. Đúng folder layout một mình không là full compatibility.

## G. Remaining issues / Definition of Done

| Criterion | Status |
| --- | --- |
| Outputs/model audit, BIN/TXT/rig equivalence, images/geometry validation | PASS cho real legacy; video BLOCKED_INPUT |
| Parser/preparation/config/CLI, no source writes | PASS |
| Reproducible dataset, split/statistics/inverse normalization | PASS cho experiment legacy riêng |
| Automated checks và actual real dataset integration | PASS ở mức project adapter/artifact gate; symlink test SKIPPED có lý do |
| Independent Original3DGS/2DGS input-file reader checks | PASS |
| Full downstream loader compatibility | INCOMPLETE; thiếu dependencies và PGSR filter/writer adapter |
| Documentation/Phase02 handoff | Cập nhật bằng artifacts/commands thật |
| Toàn bộ Phase01 / training readiness | **PARTIAL / chưa chấp nhận training-ready hoàn chỉnh** |

Protocol hiện tại là **fixed-pose transductive SfM**: BA/geometry gốc đã truy cập held-out views. Loại test tracks/RGB downstream không đảo ngược việc đó. Strict train-only reconstruction cần qualified source và test-registration protocol riêng, không được thực hiện/rerun trong task này. Temporal overlap của video cũng không biến mất nhờ split random; không claim strict independence.

Video GPU BA vẫn là blocker upstream; không đổi binary/policy hoặc chọn legacy làm video benchmark. Sparse far/background geometry, dense accuracy, CUDA renderer/toolchain/VRAM feasibility và full native training behavior chưa được xác minh ở Phase01. Không có measured PSNR/SSIM/LPIPS/mesh metrics mới.

## H. Next phase

Có artifact legacy thật để **bắt đầu native-adapter/build integration khi user yêu cầu Phase02**, giữ dataset/split/init hashes. Chưa đủ evidence để tuyên bố training-ready toàn bộ: cần full Original3DGS scene integration với manifest, native normalization composition và CUDA forward/backward gate; PGSR cần giải quyết filter/writer riêng trước baseline comparison. Nhánh video phải có qualified sparse model riêng mới handoff được. **Dừng ở Phase01; không train 3DGS, PGSR, 2DGS hoặc model khác.**

## Nguồn đã đối chiếu cho data/camera requirements

- [COLMAP output format](https://colmap.github.io/format.html) và [official binary I/O](https://github.com/colmap/colmap/blob/main/src/colmap/scene/reconstruction_io_binary.cc): little-endian, IDs/indices, camera/image/point/rig/frame layout. Legacy `frames.bin` data count là uint32, khác track length uint64; parser đã đối chiếu source và actual rig poses.
- Schönberger & Frahm, *Structure-from-Motion Revisited*, [bản tác giả](https://demuc.de/papers/schoenberger2016sfm.pdf), §2 và§4.3: camera/point observations, reprojection và cheirality. CVF endpoint trả403 trong phiên này; dùng author-hosted primary PDF, không thêm vào `paper/`.
- P02 Original3DGS local PDF pp.4–5, §3–5: calibrated cameras + sparse initialization, RGB/SH và neighborhood-based scales thuộc Phase02; P01 2DGS pp.3–4 §4: camera/world/projective conventions; P14 PGSR pp.5–6 §IV: camera/plane geometry. Local PDFs được đọc theo [inventory](research/paper_inventory.md), không sửa/chép lại toàn văn.
- Native data readers: [3DGS pinned source](https://github.com/graphdeco-inria/gaussian-splatting/blob/54c035f7834b564019656c3e3fcc3646292f727d/scene/dataset_readers.py), [2DGS pinned source](https://github.com/hbb1/2d-gaussian-splatting/blob/f3e3b9fa67bbd1c75e05167ff37391d8dab2a678/scene/dataset_readers.py), [PGSR reader](https://github.com/zju3dv/PGSR/blob/de24f1a38b350387e8d8fe381b2cd70c1ae946e7/scene/colmap_loader.py), [PGSR scene writer/filter route](https://github.com/zju3dv/PGSR/blob/de24f1a38b350387e8d8fe381b2cd70c1ae946e7/scene/dataset_readers.py). Không dùng folder names thay execution evidence.
