# Phase01 — CLI và prepared-v1 thực tế

Implementation ngày 15/11/2026, chỉ data preparation trên CPU. Không reconstruct SfM, không training hoặc benchmark CUDA. [Data specification](data_specification.md) giữ các quy tắc nghiên cứu; file này mô tả API/artifacts đã triển khai. Kết quả dữ liệu thật: [báo cáo Phase01](../phase_01_implementation_report.md).

## Chạy từ repository root

```powershell
python -m src.data.validate_colmap --config configs/data/ignatius.yaml
python -m src.data.prepare_dataset --config configs/data/ignatius.yaml --dry-run

# Thí nghiệm riêng từ image_set, đã được user cho phép; không thay nguồn video.
python -m src.data.validate_colmap --config configs/data/ignatius_legacy.yaml --discover
python -m src.data.prepare_dataset --config configs/data/ignatius_legacy.yaml --dry-run
python -m src.data.prepare_dataset --config configs/data/ignatius_legacy.yaml
python -m src.data.prepare_dataset --config configs/data/ignatius_legacy.yaml --resume
python -m src.data.inspect_dataset --config configs/data/ignatius_legacy.yaml
python -m src.data.verify_compatibility --config configs/data/ignatius_legacy.yaml --references .tools/phase01_native_readers
```

Video chưa có model nên hai lệnh video trả `BLOCKED_INPUT`, exit code 2; không lấy model legacy/probe thay thế. `--discover` báo cả model thiếu file và đánh dấu BA fixtures `DIAGNOSTIC_NOT_SCENE`; chỉ `source.model_dir` được chọn. `--dataset <relative-path>` bắt buộc khi có nhiều dataset trong configured output root. Mỗi invocation tạo thư mục report mới chứa `invocation.json`, `run.log`, `report.json`; CLI in đường dẫn thực tế. `SUCCESS` của validation/preparation là gate dữ liệu cục bộ, không phải training thành công. Compatibility `PARTIAL` là component checks đã chạy nhưng full scene loader chưa chạy; critical failure trả exit 2.

Không dùng `src.pipeline` cho Phase01 và không thêm keys vào `sfm_ignatius.yaml`. Không cài dependency mới; dùng NumPy, Pillow, PyYAML đã có. Parser đọc COLMAP little-endian BIN/TXT theo [official format](https://colmap.github.io/format.html); không copy toàn bộ official repository vào `src/`. PyCOLMAP chưa có trong interpreter hiện tại. Camera model IDs 0–10 được parse; projection/preparation chỉ hỗ trợ `SIMPLE_PINHOLE`, `PINHOLE`, `SIMPLE_RADIAL`. Model khác và rig nhiều sensor bị reject rõ ràng. Trivial rigs phải có frame/image pose equivalence, không bỏ qua frames file.

## Configuration

Default values/schema ở `src/data/config.py`; unknown keys bị reject. Hai YAML scene dùng cùng framework YAML/pathlib/repository-relative paths với code cũ nhưng schema riêng, không hạ GPU policy. `source` chọn đúng image root/model/encoding, optional paired BIN/TXT export và preprocessing manifest. BIN/TXT phải có exact semantic hashes bằng nhau, kể cả observations/tracks; không fallback TXT khi BIN hỏng. `source.ba_history` là provenance khai báo, được gắn với hash `project.ini` nếu tồn tại; không phải phép xác minh CUDA mới.

`images` chọn `auto|required|never` undistortion, optional `max_width`, `preserve|png|jpg`, JPEG quality. V1 không crop hoặc EXIF autorotate ngầm. `required` bảo đảm output pinhole; source pinhole đã đáp ứng điều kiện. Mặc định legacy giữ 1920×1080 và PNG để không thêm JPEG loss. `output.file_mode` là `copy|hardlink|symlink` cho file không cần biến đổi; ảnh cần undistort/resize/reformat luôn thành file dẫn xuất mới. Link failure được báo, không đổi sang copy ngầm. Prepared data phải dùng như immutable artifacts; khi cần chỉnh ảnh hãy tạo copy/experiment mới, vì hardlink/symlink chia sẻ dữ liệu với nguồn.

`filter.duplicate_tracks` mặc định `reject`, legacy chỉ định `drop_track`: loại toàn track trùng image_id và clear reciprocal associations. `filter.outside_image_tracks` mặc định `reject`; legacy `drop_track` cho point có tracked observation nằm ngoài prepared image sau warp. Không clamp keypoints để che lỗi. Optional `max_point_error_px` là ngưỡng reported source error, không metric test. Tất cả removed IDs/reasons được lưu. Structural errors, missing images, invalid poses hoặc broken references không được chữa bằng outlier filtering. Thresholds là project heuristics, không official benchmark thresholds.

`split.strategy`: primary `interleaved` dùng period8/test0/val4 theo spec; `ratios` áp dụng cho `random` và `temporal_block`, effective ratios luôn ghi trong manifest. Seed chỉ ảnh hưởng random; các cách chia khác vốn deterministic. Video phải có timestamp/frame identity/hash thật từ manifest. Image_set dùng numeric-aware filename order và timestamp/video identity `null`. Mọi split phải nonempty/disjoint/complete, train có ít nhất hai views. `strict_train_only` trả blocker kèm yêu cầu nguồn SfM/test registration được qualify riêng; Phase01 không tái chạy SfM và không nhận một declaration làm bằng chứng independence.

## Artifacts

```text
<dataset_root>/<dataset_name>-<content-hash>/
  images/                         # ảnh undistorted, tên giữ nguyên hoặc đổi suffix có mapping
  masks/                          # chỉ xuất khi inverse warp có pixel không hợp lệ
  qualified_model/                # accepted all-view geometry, không dùng màu ở đây để initialize
  sparse/0/{cameras,images,points3D}.bin
  sparse/0/points3D.ply            # train-only initialization, zero normals
  sparse/{cameras,images,points3D}.bin
  sparse/points3D.ply              # byte-identical mirror cho PGSR pinned loader
  metadata.json
  splits.json
  normalization.json
  qualification.json
  train_observations.npz
  images_progress.json
  integrity.json
  readiness.json
  state.json
```

`metadata.json` chứa source/model/image/manifest hashes, config, implementation fingerprint, UTC timestamp/timezone, actual command/environment, camera matrices per image và scene statistics. `qualified_model` giữ accepted geometry sau các filters đã ghi; source originals không đổi. `sparse` chỉ giữ points có ≥`min_train_views` unique train observations (optional minimum parallax), train-only tracks/colors/recomputed errors. Validation/test images vẫn có poses trong export, nhưng mọi point association được clear; không loại ảnh khỏi frozen split. Original all-view geometry vẫn mang transductive SfM disclosure.

NPZ gồm `image_id`, `point2D_idx`, `point3D_id` int64; `xy[M,2]`, `z[M]`, `residual_px[M]` float64. Chỉ chứa train rows, giữ source keypoint indices. Train RGB được sample bilinear ở transformed observation coordinates rồi lấy mean/round RGB uint8; không đọc held-out RGB trong bước tính màu. Đây chưa là confidence/anchor cache của Phase05.

## Camera/image math và normalization

COLMAP Hamilton WXYZ; `Xc=Rcw Xw+tcw`, `Cw=-Rcw.T tcw`; camera x-right/y-down/z-forward. Camera/point IDs không phải array indices. Metadata `Tcw`, `Twc`, `K` float64; RGB loader `[3,H,W]` float32 trong `[0,1]`, optional validity mask `[1,H,W]` bool trên CPU.

SIMPLE_RADIAL giữ đúng normalized distortion `xd=x(1+k*r²)`. Ảnh pinhole được inverse-warp theo radial source projection, bilinear từng tile64rows; keypoints dùng Newton inversion có kiểm tra convergence/monotonic domain. Derived PINHOLE dùng focal/principal point tương ứng, không bỏ coefficient rồi dùng JPG distorted. COLMAP pixel center đầu tiên `(0.5,0.5)`, Pillow sampling index là `coordinate-0.5`. Resize dùng dimensions thực đã làm tròn: `K'=diag(sx,sy,1)K`, keypoints cũng nhân cùng factors. Config/model/image transform được lưu; không có hidden crop hoặc pose optimization.

Canonical XYZ/poses giữ nguyên scale/coordinate COLMAP. Normalization chỉ lưu downstream transform, fit median và percentile90 radius từ **train camera centers**, `X'=s(X-c)`, `t'=s(t+Rc)`; inverse được lưu. `normalize_geometry()` áp dụng đồng bộ points/poses khi consumer yêu cầu. Không auto-normalize độc lập từng method. Monocular units chưa là meters; không dùng GT để scale/alignment. Phase02 phải disable hoặc compose native transform tường minh.

## Loader và native compatibility

```python
from pathlib import Path
from src.data.loader import PreparedDataset, verify_dataset, apply_native_split

report = verify_dataset(Path("<prepared-directory>"))
train = PreparedDataset(Path("<prepared-directory>"), split="train")
sample = train[0]                 # lazy CPU decode, không preload dataset lên GPU
points = train.initialization_points()
# Khi native dependencies đã có ở Phase02:
# native_scene = actual_native_readColmapSceneInfo(..., eval=False, ...)
# views = apply_native_split(native_scene, directory, method="3dgs")
# fit native normalization từ views['train']; không dùng native implicit split.
```

Ba native readers có folder/basename/split khác nhau. `apply_native_split` kiểm tra complete camera coverage và init pointcloud trước khi chọn train/val/test; actual SceneInfo integration còn phải kiểm thử ở Phase02. FoV-only native loaders không hỗ trợ principal point lệch tâm và không tự nhận validity masks; compatibility gate báo adapter requirement, không gọi folder layout là full compatibility. Loader references/commit/source/license hashes ở [registry](../../configs/data/loader_references.json); source snapshots nhỏ ở ignored `.tools/phase01_native_readers`, không là installed Gaussian implementations. Compatibility CLI không download/import training loop.

**PGSR pinned-source gate:** `read_points3D_binary` tự giữ `error<=2px` và `track_length>=3`; `readColmapSceneInfo` dùng điều kiện `if not os.path.exists(ply_path) or True`, nên luôn ghi lại `sparse/points3D.ply`, kể cả PLY đã tồn tại. Actual native reader vì vậy không giữ đầy đủ common initialization của prepared-v1; compatibility CLI phải báo failure, không sửa error/track values để ép pass. Native camera/point parsing semantics được kiểm tra riêng trong [report](../phase_01_implementation_report.md). Không gọi full PGSR scene reader trực tiếp lên immutable prepared dataset. Phase02/03 adapter cần writable derived staging, frozen split và common initialization có kiểm chứng; `apply_native_split` một mình chưa xử lý việc native writer/filter này.

## Resume và lỗi thường gặp

Completed dataset chỉ reuse khi cache key (source/model/images/config/relevant code) và toàn bộ artifact hashes/camera/split/observations còn đúng. Destination tồn tại mà thiếu `--resume` → lỗi; checksum mismatch → lỗi, không ghi đè. Interrupted stage giữ `.<dataset_id>.partial-*/state.json` với `FAILED`; `--resume` tạo stage mới và chỉ copy ảnh/masks đã có source+output hashes khớp trong progress manifest, không sửa run thất bại cũ. Models/statistics/splits được build lại và revalidate; công bố dataset chỉ bằng rename sau readiness pass. Khi đổi config/code, dataset ID mới, không ghép cache cũ tùy tiện.

Missing model → kiểm tra qualified output thực tế, không chọn BA probe hoặc ground truth; sửa GPU BA thuộc task khác. Duplicate/outside-image tracks → xem rejected IDs, giữ default reject nếu chưa chọn derived policy. `WinError 1314` → dùng `copy` hoặc `hardlink` trong config nếu không có quyền symlink. Full native loader chưa import được → kiểm tra `plyfile`/native extensions trong môi trường Phase02; không tạo stubs để gọi pass.

## Kiểm thử

```powershell
python -m pytest -q
python -m ruff check src tests
```

Fixtures tổng hợp chỉ dùng unit tests: BIN/TXT/truncation, Hamilton poses/projection/inverse, distortion/resize pixel centers, reversible normalization, malformed cameras/points/tracks/images, deterministic splits/temporal ordering, train-only color isolation, links/overwrite, transaction/interrupted resume và hash tampering. Readiness phải chạy thêm trên model thật; không lấy unit fixture metrics làm scene result. Phase02 mới khởi tạo CUDA scales từ native nearest-neighbor distances và SH DC theo RGB2SH; Phase01 chưa có Gaussian optimizer/checkpoint.
