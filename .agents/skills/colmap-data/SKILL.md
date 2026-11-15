---
name: colmap-data
description: Validate and adapt existing COLMAP outputs into prepared Gaussian-training datasets in this repository. Use for camera conventions, undistortion, tracks, normalization, splits, and loader changes after SfM; retain the existing video-sfm guidance for SfM execution.
---

# COLMAP data adapter

## Kích hoạt và context

Kích hoạt khi đọc/qualify/chuyển **output đã có** của COLMAP hoặc sửa prepared dataset loader. Đọc root AGENTS, [Phase01](../../../docs/plans/phase_01_data_preparation.md), [data spec](../../../docs/tech/data_specification.md), [audit](../../../docs/repository_audit.md). Paper: P02 §3, P03 §4.3, P14 §IV tại [inventory](../../../docs/research/paper_inventory.md).

## Workflow

1. Xác định request là documentation, implementation hay execution. Task documentation không cho phép chạy SfM/training. Inspect actual source/model và protected paths trước edit.
2. Chọn đúng `source_mode`; default video. `video_v0` đang thiếu model theo audit phải kiểm tra lại; không lấy legacy image_set hoặc diagnostic fixture thay thế. Lưu provenance/BA history.
3. Validate camera model, Hamilton WXYZ/Tcw, IDs/reciprocal tracks, image names/sizes, finite poses, observed positive depth và reprojection residuals.
4. Report duplicate observations cùng image; không sửa raw/model. Policy dẫn xuất/filter chỉ theo explicit dataset config; revalidate sau filter.
5. Undistort sang copy; transform K/keypoints chính xác. Giữ pixel-center convention, normalization/inverse và native adapter transform nhất quán.
6. Freeze splits; derive train-only point appearance/observations/confidence inputs. Công bố transductive SfM nếu all-view poses/points đã dùng held-out RGB trước đó.
7. Validate prepared artifacts và report source changes bằng hashes. Không dùng GT để fix poses, scale hoặc lựa chọn ảnh.

## Tools và commands

Read-only `rg`, Git status/diff, pathlib/PIL headers, existing TXT/PLY readers và hashes phù hợp. SQLite `mode=ro` vẫn có thể cập nhật `database.db-shm`: để giữ nguyên cả sidecars, kiểm tra trên snapshot nhất quán ngoài protected outputs; chỉ dùng `immutable=1` khi chắc chắn database là snapshot tĩnh và không bỏ sót WAL chưa checkpoint. Không dùng immutable tùy tiện trên database đang có writer. Conversion/undistortion CLI chỉ dùng trong implementation/execution scope, tới output mới và với flags binary thực. Muốn thực thi SfM đọc bộ cũ [SKILL-video-sfm](../../../.agent/skills/preprocessing/SKILL-video-sfm.md), [plan](../../../.agent/skills/preprocessing/plan-video-sfm.md), [tech](../../../.agent/skills/preprocessing/tech-video-sfm.md); không tự mở mapper.

## Validation, lỗi và hoàn thành

Tests meaningful: projection before/after resize/undistort/similarity; noncontiguous IDs, duplicate tracks, inverse pose và split isolation. Common failures: distortion bị bỏ, Tcw/Twc đảo, half-pixel hai lần, all-view colors/track errors rò test, sparse/0 mặc định sai source.

Done khi artifacts parse, refs/hashes hợp lệ, source nguyên trạng, loader conventions và required tests được xác minh. Thiếu model/backend ghi blocker và pending stage; không tuyên bố full dataset Done chỉ vì synthetic fixture pass.
