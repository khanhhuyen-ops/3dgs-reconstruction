# Reproducibility và experiment integrity

Áp dụng mọi phase; [evaluation](evaluation_specification.md) và [data lineage](../architecture/data_flow.md) quy định comparability. Phase02 đã có actual B0 smoke/resume và low-memory30.000 iterations/33 held-out metrics; full-resolution reference được ghi riêng theo trạng thái thực ở [report](../phase_02_implementation_report.md). [B0 usage](b0_training_usage.md) mô tả manifests/source/dependency/weight locks; không gộp resource profiles thành paper benchmark. Low-memory bị gián đoạn ngoài process sau checkpoint25.000, nên tổng thời gian được tổng hợp thành observed lower bound từ logs, không dùng thời gian segment resume làm toàn run. Recorded Torch peaks và sampled device-wide VRAM có scope khác nhau.

B0 thực thi với project-local venv đọc parent compatible environment, không overwrite base/SfM. Native sources giữ separate checkout, exact commit/submodules và installed extension hashes. `.tools/b0_build_logs/` giữ build logs, resolved package snapshot, protected-file baseline hashes, CUDA/metric validation và own-source ZIP+SHA snapshot cho working-tree code chưa commit. Mỗi run lưu resolved YAML/JSON, native/environment hashes, dataset/split identity, invocation/logs, model/optimizer/RNG/sampler checkpoints và Gaussian export verification. Giữ original SfM/prepared inputs immutable; các metadata/report updates chỉ ở output run mới hoặc tài liệu liên quan.

## Manifest tối thiểu

| Nhóm | Fields phải lưu trong run tương lai |
| --- | --- |
| Identity | runID, UTCtimestamps và displaytimezone, stage/method/variant/schema, parentrun IDs |
| Source | Git commit, dirtydiff/contenthash, native repository/submodule SHA, license snapshot/reference |
| Input | Video/image/SfM hashes, source_mode, accepted IDs, image transforms/undistortion, pointfilter report |
| Protocol | Split hash, fixed-pose/transductive disclosure, mask/normalization/units, baseline/extractor track |
| Config | Versioned config và fully resolved values, profile, seed, overrides/derivation |
| Environment | Python/package lock, wheel/build/compiler/CUDA/driver, GPU name/VRAM, OS |
| State | Native checkpoint/optimizer/scheduler/ADC/RNG/exposure, iteration và artifact hashes |
| Execution | Exact argument list, cwd relative to root, command returncode, stdout/stderrlogs, stage timings |
| Resources | peakallocated/reserved, sampleddevicewide VRAM với scope/interval, OOMdetails |
| Metrics | Values nullable, definitions/codehash/weights, GT/alignment/threshold/evalmasks, perseed/perview |
| Outcome | Stage status, reason, validationchecks và unresolvedblockers |

Không lưu secrets hoặc absolute personal paths vào versioned configs; external executable resolve qua env/config và log version/hash với path redaction khi share. Timestamp PTS của video không thay bằng filenameindex/FPS. Hash binary/model weights là provenance, không chỉ semanticversion.

## Determinism và resume

Seed42 mặc định thiết kế, seeds43/44 để đánh giá dispersion. Python, NumPy, TorchCPU/CUDA RNG phải save/restore; dataordering và camera sampling deterministic. Ghi nondeterministic CUDA kernel/reduction exceptions; same seed không tự chứng minh bitwise reproducibility. Pin supported determinism settings, không tắt kernel rồi chuyển CPU để né.

Checkpoint atomic tới filemới, validate lại trước cập nhật pointer. Include lastcompletediteration và fulloptimizer state; không overwrite unrelated existingrun. Resume từ inferencePLY không được gọi exactresume. Hashmismatch/corruptcache→fail giải thích, không tự dọn outputs hoặc dùng dataset gần giống.

## Cache keys theo stage

- Prepared dataset: source model/images/manifests + parser/undistortion version + image transforms + accepted ID policy + normalization/split.
- Confidence: dataset/split hash + train-only track extraction/residual/angle algorithm + hyperparameters.
- Training: dataset + method/native SHA/submodules + resolved profile/loss/optimizer/seed + renderer build hash.
- Rendering: checkpoint + cameras/image dimensions + depth/normal definitions + appearance policy.
- Meshing: renderbundle/field evaluator + ROI/voxel/truncation/filter/cleaning + backendversion.
- Evaluation: predictions + GT hashes/unit/alignment/masks/thresholds + evaluator code/weights/samplingseed.

Source-only docs changes không cần recompute geometriccache nếu stage keys không đổi relevantcode; nhưng Gitrevision/dirtyhash vẫn phải lưu trong report. Đổisplit invalidate training/anchors; đổimesher không tự cần retrain.

## Experiments và reporting

Run mới có directory riêng; no partialoutput gọi SUCCESS. `SKIPPED` có reason; `BLOCKED_GPU_BA` giữ nguyên status cũ. Downstream proposedstatuses `BLOCKED_INPUT/BACKEND/RESOURCE/EVALUATION` chưa là existingCLI features. Failedruns/partialcheckpoints được giữ, không exclude thất bại khỏi aggregate ngầm.

Primary method change đi qua decisionrecord gồm problem/source, alternatives, selectedvariant, expectedtest, protocolimpact và evidence. Không cần một filedecision trùng lặp cho từng chỉnh câu; masterplan giữ registry và links. Research hypothesis/proposed values/paper results/measured results phải có label riêng.

Đóng phase cần artifacts tồn tại, nonempty, parse, qualitychecks pass, provenance đầy đủ và requiredtests thực sự chạy. Mockcontrolflow chỉ chứng minh orchestrator; CUDA smoke chỉ chứng minh smallkernel; scene training/evaluation là evidencelevels khác. Báo rõ skipped tests và nguyên nhân.
