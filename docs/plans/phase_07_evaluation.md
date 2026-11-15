# Phase 07 — Evaluation and reporting

Trạng thái hiện tại: **PLANNED — chưa triển khai/chạy**. User request tiếp theo quyết định việc thực thi; xem [master plan](master_plan.md). Mọi thông số preset là thiết kế, không benchmark đã đo.

## 1. Phase overview

Hoàn thiện và freeze evaluator, tạo báo cáo nghiên cứu từ comparison artifacts đã lưu. Không sửa training dựa trên test errors.

## 2. Research objective

Đánh giá rendering/geometric/mesh quality với units/alignment/protocol có thể tái lập, phân biệt pilot và scientific validation.

## 3. Scientific references

P02 §7; P01 §6; P14 §V; P05 App.B; P07 §4; P11 §6; [evaluation spec](../tech/evaluation_specification.md).

## 4. Current dependencies

Predictions/checkpoints/meshes từ Phase06; RGB validator dùng chung từ Phase02; official GT metadata/masks/threshold/alignment và evaluator revision đã xác minh. Thiếu official protocol → BLOCKED_EVALUATION cho metric tương ứng.

## 5. Input requirements

Frozen run matrix và trạng thái mọi run; GT file hash và units; evaluation ROI/masks, threshold τ mỗi scene, registration transform; LPIPS weights/code hash; sampling seed/count; metric names và norm definitions.

## 6. Expected outputs

Raw JSON metrics theo view/scene/seed; aggregate CSV/Markdown tables; ablation tables; qualitative figures cùng cách hiển thị; resource/mesh complexity tables; evaluator configs/scripts và limitations/provenance. Chưa chạy thì giá trị là null, trạng thái NOT_RUN.

## 7. Detailed architecture

Evaluator đọc predictions và GT → inverse normalization/alignment đã công bố → fixed masks/sampling → RGB/depth/mesh metrics → aggregation/figures → review claims. Dữ liệu GT không quay về optimizer hoặc training caches.

## 8. Mathematical requirements

PSNR mỗi view; pinned SSIM/LPIPS; unsquared Euclidean CD=0.5(Acc+Comp); Precision/Recall/F1 theo threshold; normal/depth metrics chỉ khi có GT phù hợp. Ghi rõ metric units/scale và Sim(3) matrices; không dùng threshold 5cm của indoor thay TNT threshold.

## 9. Implementation tasks

Hoàn thiện GT protocol reader, tích hợp official evaluator hoặc generic diagnostic có nhãn rõ; sample theo triangle area; kiểm tra alignment/metric fixtures; aggregation có failed counts; mean/std theo seeds; export tables/figures cùng cameras/crops; đối chiếu claims với evidence từ artifacts.

## 10. Ordered execution steps

1) Verify input/protocol hashes. 2) Synthetic metric/alignment tests. 3) Evaluate one frozen run to validate units/masks. 4) Execute all frozen matrix without tuning. 5) Aggregate with missing/failed counts. 6) Produce qualitative/static figures. 7) Review claims/provenance và final report.

## 11. Technical dependencies

Versioned RGB metrics/LPIPS weights; mesh NN/sampling tools/evaluator build; official TNT protocol availability. No automatic dependency download/training unless phase scope includes it; expensive learned metric requires CUDA.

## 12. Hardware and resource constraints

Stream views/sample chunks; GPU required learned metric on CUDA; CPU parsing and declared geometric diagnostics allowed. Report evaluator runtime separately; LPIPS model loaded separately from trainer/mesher.

## 13. Validation strategy

Fixture plane/translated mesh với distance biết trước; unit conversion và Sim(3) inverse; identity PSNR/SSIM/LPIPS; empty/invalid cloud phải báo lỗi; deterministic area sampling và mask edge cases; tái tạo bảng từ raw metrics. Không trình bày số đo mock fixture thành kết quả scene thật.

## 14. Evaluation metrics

PSNR/SSIM/LPIPS; Acc/Comp/CD/P/R/F1τ; depth/normal accuracy khi đủ GT; time/VRAM; faces/vertices/components/file bytes; toàn bộ seeds và failed counts. Kiểm định H0/H1 kèm dispersion thực đo.

## 15. Risks and mitigation

Rủi ro: thiếu GT units/ROI/threshold, alignment ưu ái từng baseline, báo training-view PSNR thành NVS, chọn crops/seeds thuận lợi hoặc loại scene thất bại. Giảm bằng metric definitions rõ, frozen protocol, failure tables đầy đủ; công bố giới hạn một scene chỉ là pilot.

## 16. Acceptance criteria

Mọi metric truy được prediction/evaluator/config/GT; chỉ gọi official score khi đã xác minh official protocol. Cùng units/split/masks/alignment policy trong comparison group; giữ kết quả âm/failed; không claim novelty/SOTA thiếu evidence.

## 17. Definition of Done

Báo cáo/bảng/hình tái tạo được từ frozen artifacts; citations/links hợp lệ, phân biệt measured/paper/proposed, ghi rõ blockers chưa xử lý. Thiếu GT protocol thì chưa hoàn thành full scientific evaluation.

## 18. Next phase dependencies

Không tự mở VGGT/SLAM/SDF/MILo phase mới. Quyết định nghiên cứu tiếp theo dựa trên kết quả đã review và yêu cầu user; bảo toàn documentation và experiments hiện có.

