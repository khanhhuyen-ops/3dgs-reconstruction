---
name: paper-research
description: Audit reconstruction papers, verify equations and official sources, compare implementations, and turn evidence into cited research requirements or novelty assessments. Use for literature review and documentation work without implying model execution.
---

# Paper research

## Kích hoạt và context

Đọc AGENTS, [inventory](../../../docs/research/paper_inventory.md), [review](../../../docs/research/literature_review.md), [gaps](../../../docs/research/research_gaps.md), [master plan](../../../docs/plans/master_plan.md). Chọn papers theo algorithm/request, không load toàn literature nếu chỉ kiểm tra một formula.

## Workflow

1. Inventory PDFs recursively, record path/hash/pagecount/version; inspect existing docs để mở rộng cùng hệ thống. Không đổi raw/PDF/SfM outputs.
2. Dùng available PDF reader, extract toàn văn vào scratch ngoài repo nếu cần. Ghi failures thật; không tự cài dependency nếu scope cấm. Text extraction không đảm bảo hiểu equations/tables: xem ảnh trang khi ký hiệu mất.
3. Review 20 fields theo literature document: metadata/problem/motivation/contributions/method/I/O/architecture/math/representation/optimizer/loss/ADC/training/extraction/datasets/results/compute/limitations/code/application.
4. Cite exact local physical page và section/equation/table; năm filename không thay publication year. Phân biệt paper revision và official code snapshot; pin source commit khi đọc code.
5. Official code verify từ author/project/repository, log inaccessible/redirect rõ; repo truy cập được chưa chứng minh build/backend. Web research dùng primary sources, không lấy search snippets làm full-paper review.
6. Khi đề xuất method, tìm overlapping work với targeted primary searches; sparse anchors/edge/depth/visibility không tự mới. Separate existing method, proposed modification, hypothesis và measured result.
7. Chuyển evidence sang architectural decision/spec/tests/resource gate; unknown ghi `Chưa xác minh`, absent information `Not specified in paper`, method-inapplicable `Not applicable`. Không dùng ba nhãn thay nhau hoặc làm placeholder đại trà.

## Tools và commands

`rg --files`, read-only PDF/text/image tools, primary web sources và metadata/hash readers. Không clone/cài/train/benchmark chỉ vì nghiên cứu paper; nếu cần source dùng read-only pinned files. Không đưa extracted full text hoặc copies dài vào Git docs.

## Validation, lỗi và hoàn thành

Check citation locations/version, equations/units/metric definitions, working internal links, author-vs-code differences và coverage tất cả PDFs được yêu cầu. Common failures: merge table numbers giữa versions, cite survey thay primary, assume mesh mechanism/checkpoint compatibility, call hardware preset measured result.

Done khi inventory/review coverage có evidence, citations truy được, gap/novelty limitations rõ và linked requirements không vượt user scope. Không ghi đã đọc PDF khi chỉ xem abstract hoặc extraction bị lỗi.
