# Research gaps và novelty audit

Trạng thái14/11/2026: chưa có kết quả training/meshing của dự án. Một scene Ignatius chỉ đủ pilot; kết luận tổng quát cần thêm scenes theo protocol riêng, không tự tải dataset.

| Vấn đề | Evidence literature | Điều còn phải kiểm chứng |
| --- | --- | --- |
| RGB đẹp nhưng geometry sai | P02 §7.4; P01 §3; P14 §I | B0/B1/B2 cùng inputs, native/common extraction |
| Sparse geometric supervision | **P03 §4.3 p.6 đã dùng SfM z-depth L2** | Weighting thay vì chỉ thêm sparse depth có giá trị không? |
| Reliability/edge priors | P05 Eq.(4–5), geometry regularization và SfM alignment | Có lợi khi chỉ dùng COLMAP, không thêm neural prior? |
| Visibility consistency | P09 §3.2 Gaussian contribution; P14 §IV.B roundtrip mask | SfM support không phải dense visibility; tác động ngoài textured points? |
| Confidence-aware depth | [CDGS §3.1–3.2](https://arxiv.org/html/2502.14684v1#S3) | Overlap trực tiếp: reprojection weighting và confidence-depth đã tồn tại |
| Thin details/extraction | P07 §3.4; P11 §4/§6.6 | Improvement từ loss hay mesher? Đo cùng extractor |
| Hardware feasibility | P11 §6.1, P09 §4.2 | Có thể đạt quality nào ở cùng6GB/resource budget? Chưa đo |

**Không có bằng chứng đủ để gọi “COLMAP-guided Confidence-aware Geometry Regularization” là đóng góp khoa học mới.** Track length/error/visibility là heuristic reliability cues; kết hợp chúng vào PGSR có thể chỉ là engineering integration. Không dùng từ first/SOTA/novel trong báo cáo khi chưa có novelty review rộng và experiments.

Phạm vi khả thi để nghiên cứu: kiểm tra whether fixed, training-only SfM reliability ranking giúp robust sparse-anchor optimization so với unweighted anchors và error-only weights trên cùng B2. Parallax có thể chống việc nhiều gần-trùng views làm track length tăng giả; chưa phải uncertainty calibration đã chứng minh.

## Falsification và stop criteria

- Weighted anchors không hơn unweighted ở nhiều seeds hoặc làm completeness giảm: không claim đóng góp; báo negative result.
- Lợi ích biến mất khi dùng cùng mesh settings/Gaussian cap: confound đã phát hiện, không giữ claim geometric regularization.
- Chỉ cải thiện chính các anchors: chưa chứng minh dense surface; đo geometry ở held-out observations và mesh evaluator riêng.
- Tăng compute/memory quá budget hoặc không có valid video model: báo blocker/feasibility limit, không đổi source/protocol lặng lẽ.
- Tìm thấy method prior trùng toàn bộ proposed term: giữ thành reproduction/engineering experiment, đổi research question qua decision record.

Novelty search bổ sung đã dùng các từ khóa `Gaussian splatting track length reprojection confidence`, `sparse depth confidence COLMAP geometric regularization` trên nguồn primary. Kết quả này là kiểm tra overlap có mục tiêu, không là bảo đảm exhaustive literature search. Các nguồn không có PDF local không được ghi thành 14-paper measured experiment.
