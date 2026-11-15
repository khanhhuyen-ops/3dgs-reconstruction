# Documentation audit — 14/11/2026

Task hoàn thành ở mức hệ thống tài liệu; không implementation hoặc training. File catalog đầy đủ: [docs index](README.md). Evidence trạng thái scene: [repository audit](repository_audit.md). Đây là report validation của phiên hiện tại, không tái sử dụng số “17 passed” từ báo cáo10/11.

## 1. Deliverables và coverage

| Nhóm | Số file | Nội dung |
| --- | ---: | --- |
| Research | 7 | Inventory, review, comparison, math, gaps, proposal và PGSR paper/code notes |
| Architecture | 4 | System, data flow, training, meshing; có Mermaid diagrams |
| Technical specifications | 9 | Stack/environment/data/representation/training/loss/meshing/evaluation/reproducibility |
| Plans | 8 | Master plan +7 phase plans |
| Audit/navigation | 3 | Repository audit, documentation audit, docs index |
| New skills | 6 | COLMAP data, training, geometry, mesh, evaluation, paper research |
| Root updates | 2 | AGENTS routing/current scope và README navigation |

Tổng **37 Markdown mới +2 file Markdown cập nhật**; không thêm runtime config/source/script/test. Thư mục `paper/` đã untracked trước task; không tự add/commit hoặc đổi PDF.

14 local PDF,209trang,14/14 text extraction thành công. Review có20 trường/paper (280 trường), có citations theo local page/section/equation/table. Không có PDF local không đọc được; supplemental CDGS PDF download lỗi TLS đã ghi, chỉ đọc methodology bản HTML, không cộng vào14PDF. SuGaR/VGGT/VGGS không có localPDF và không được khai là full-primary review. Công thức PGSR với dấu overline được xem ảnh trang6; không giả vờ kiểm tra trực quan mọi hình/trang của209pages.

7/7 phase plans có18 mục (126 mục), specific input/output/tasks/dependencies/validation/acceptance/DoD. Hai Development/Research Quality profiles là YAML examples trong Markdown, parse được; không là YAML runtime đã được triển khai hoặc GPU benchmark đã chạy.

## 2. Validation thực hiện

- Read-only filesystem/Git inventory; đọc hướng dẫn cũ, callers/configs và reports. `git diff --check` không báo whitespace errors.
- PyMuPDF extraction đọc được14PDF, ghi page counts và SHA-256; metadata/publication year kiểm tra từ PDF/primary sources thay vì filename.
- Existing TXT parser đọc legacy model; image-name/dimension checks263/263; sparsePLY parse85.656 vertices; recomputed residual diagnostics. Existing strict `model_metrics` **fail** do duplicate same-image tracks: không đổi thành SUCCESS.
- Markdown link audit cuối kiểm tra **231 liên kết nội bộ trong39 file**: **0broken links,0empty files**, code fences cân bằng. Các PDF filenames có ngoặc được percent-encode để tránh renderer/parser ambiguity. Không coi link tồn tại là xác minh nội dung khoa học.
- Coverage checks:14×20 literature fields;7×18 phase sections;6skills có name/description hợp lệ và tên khớp thư mục.
- Skill Creator `quick_validate.py` chạy cho6skills. Lần đầu default Windows cp1252 không đọc được UTF-8; chạy lại `python -X utf8 .../quick_validate.py <skill-dir>`: **6/6 exit0, Skill is valid**. Không sửa validator hoặc cài package.
- YAML parse kiểm tra skill metadata và2configuration profiles; source hashes/protected file stat/hash comparisons; không chạy GPU-heavy tests hoặc toàn bộpytest.

Commands chính: `git status --short`, `git diff --check`, `rg --files`, Python/PyMuPDF/PIL/TXT/PLY/hash diagnostics, `python --version`, `nvcc --version`, `python -X utf8 ...quick_validate.py`. Temporary extraction/validation helpers ở scratch ngoài repo, không trở thành pipeline code. Mermaid được review cấu trúc/flow bằng tay; chưa render bằng Mermaid CLI, không báo visual-render test pass.

## 3. Bảo toàn dữ liệu — kết quả và ngoại lệ

Trước audit lưu metadata của1.279 protected files trong source/scripts/tests/configs/outputs/oldskills/datasets; so hash của274 file gồm source/configs/output files và raw video, dùng stat cho các ảnh/raw khác. **Không có hash mismatch ở code/config/main databases/WAL/BIN/TXT/PLY/logs/raw video**;14PDF hashes vẫn theo inventory. Old SfM docs/skills không sửa theo Git diff. Không có writes vào raw; các ảnh/raw lớn khác kiểm tra bằng stat, không khẳng định đã hash toàn bộ dataset pixels.

**Ngoại lệ đã phát hiện:** mở SQLite với `mode=ro` vẫn cập nhật WAL shared-memory cache. `outputs/tanks_and_temples/Ignatius/sfm_colmap/video_smoke_v0/database.db-shm` đổi hash/mtime; `.../video_v0/database.db-shm` đổi mtime nhưng hash giữ nguyên. Main database/WAL không đổi; không chạy SQL mutations, SfM hoặc mapper. Không có byte-backup sidecar trước đọc nên không giả vờ khôi phục nguyên byte, không ghi đè để che thay đổi. Kết quả reconstruction và dữ liệu logic DB không đổi, nhưng điều kiện “mọi file trong outputs bất biến” có ngoại lệ2cache files này.

Lần audit sau muốn giữ cả cache phải dùng snapshot read-only nhất quán ở scratch hoặc immutable URI khi thật sự là snapshot tĩnh; phải xử lý WAL đúng trước khi đọc, không dùng immutable mù quáng với live writer. Hướng dẫn này đã bổ sung vào skill COLMAP data.

## 4. Consistency review và giới hạn

| Chủ đề | Kết luận audit |
| --- | --- |
| Current state | Tách legacy image_set reconstruction và video `BLOCKED_GPU_BA`; không gộp model cũ vào video |
| Source/runtime truth | Code/paths thực tế được ghi rõ; planned modules/CLI không trình bày như đã có |
| Research integrity | Existing sparse anchors và confidence weighting được ghi prior work; Ours chưa novelty/SOTA claim |
| Baseline identity | B0 không native mesh; B0+TSDF diagnostic; B1/B2/Ours TSDF; B3 opacity/MT; DN-Splatter Poisson |
| Protocol fairness | Same dataset/split/pose/resolution/budget; transductive disclosure; no test exposure fitting; native/common extraction tách |
| Coordinates | COLMAP Tcw/WXYZ, distortion, pixel center, normalization/scale/inverse và plane-sign tests có contract |
| Resource | CUDA strict giữ nguyên; profiles chưa measured; MILo10GB+ reference và Windows build risks được nêu |
| Evidence | Unit/mock/CUDA smoke/scene/official evaluation phân biệt; không có local rendering/mesh benchmark mới |
| Phase ownership | Phase02 sharedRGB validator; Phase03 B2; Phase04 mesh; Phase05 candidate; Phase06 B1/campaign; Phase07 final evaluator/report |

Chưa xác minh CUDA runtime/build mới, checkpoint compatibility thực tế, training feasibility trên6GB, mesh/Blender import, official GT threshold/masks/alignment, hoặc claimed research gains. Một số official links không truy cập được đã ghi rõ (3DGSR code404, TNT benchmark fetch, CDGS PDF TLS); không gọi là read/build success. Không yêu cầu xác nhận để hoàn thiện tài liệu và không tự triển khai phase tiếp theo.

## 5. Handoff

Sau yêu cầu mới: triển khai **Phase01 data preparation** theo [plan](plans/phase_01_data_preparation.md). Full execution trên source video vẫn cần qualified sparse reconstruction; legacy source phải được chọn riêng và xử lý qualification. Sửa GPU BA là nhiệm vụ riêng. Dừng tại documentation-first checkpoint này.
