# Báo cáo phân tích huấn luyện và kiểm toán thực nghiệm tái tạo cảnh 3D

Ngày kiểm toán: **22/11/2026**. Đối tượng: repository 3DGS Mesh Reconstruction, cảnh Ignatius. Báo cáo này được tổng hợp từ artifact đã tồn tại; không tiến hành huấn luyện lại, tiếp tục huấn luyện, suy luận mô hình hoặc tạo mesh trong đợt kiểm toán. Số liệu đo trước đây được phân biệt với các thống kê mới tính từ log. [Danh mục thí nghiệm](../metadata/experiment_registry.json), [kiểm toán checkpoint](../metadata/checkpoint_audit.csv) và [bảng so sánh](../comparisons/model_comparison.csv) là nguồn dữ liệu định lượng chính.

## 1. Tổng quan quá trình thực nghiệm

Pipeline thực tế gồm preprocessing/SfM, chuẩn bị dữ liệu COLMAP, huấn luyện Gaussian và đánh giá ảnh RGB. Nhánh video monocular vẫn bị chặn ở GPU Bundle Adjustment; không có bằng chứng hoàn tất tái tạo video GPU-strict. Các kết quả huấn luyện sử dụng **thí nghiệm legacy image_set riêng đã được cho phép**, không phải đầu ra thành công của nhánh video. SfM legacy có lịch sử CPU BA; việc này được công bố thay vì gán nhãn GPU-strict cho toàn bộ pipeline.

Kiểm toán phân biệt 8 bản ghi thí nghiệm sau khi hợp nhất các bản sao có log nội dung giống hệt: 5 lịch trình đã hoàn thành, 2 bị gián đoạn và 1 trường hợp chưa đủ bằng chứng về các bước huấn luyện. Trạng thái COMPLETED của smoke chỉ có nghĩa hoàn thành lịch trình kiểm thử ngắn. Sự tồn tại của checkpoint không được dùng làm tiêu chí duy nhất cho việc hoàn thành.

| Experiment ID | Model | Observed iteration / target | Status | Loss rows |
| --- | --- | --- | --- | --- |
| b0-kaggle-c55f001d26 | B0_Original3DGS | 30000/30000 | COMPLETED | 231 |
| ignatius-legacy-b0-low-memory_20261115T141109Z_8ed1c6 | B0_Original3DGS | 30000/30000 | COMPLETED | 302 |
| ignatius-legacy-b0-research-quality_20261115T222504Z_9ddaaa | B0_Original3DGS | 7400/30000 | INTERRUPTED | 76 |
| ignatius-legacy-b0-smoke_20261115T133608Z_4ef4c4 | B0_Original3DGS | 0/1200 | UNKNOWN | 0 |
| ignatius-legacy-b0-smoke_20261115T134942Z_346824 | B0_Original3DGS | 1200/1200 | COMPLETED | 26 |
| kaggle-pgsr-geometry-smoke | B2_PGSR | 300/300 | COMPLETED | 30 |
| local-smoke-v1 | B2_PGSR | 300/300 | COMPLETED | 30 |
| pgsr-kaggle-c269454133 | B2_PGSR | 21298/30000 | INTERRUPTED | 212 |

Các alias tải từ Kaggle và đường dẫn gốc được giữ trong registry. B0 Kaggle là continuation từ checkpoint7000 của local research run, được xác nhận bằng parent SHA và input manifest; hai bản ghi này không là hai independent replications. Nhánh local còn có log đến7400 nhưng continuation xuất phát từ state7000, nên không tự ghép các điểm loss7001–7400 của hai nhánh thành một chuỗi duy nhất. Không sao chép thêm dataset hoặc checkpoint để xây dựng báo cáo. Danh mục [artifact](../metadata/artifact_index.csv) ghi các tài liệu, cấu hình camera, mô hình COLMAP, PLY và metadata liên quan.

## 2. Môi trường và cấu hình huấn luyện

Log môi trường xác nhận RTX3050 Laptop6GiB cho các run local và Tesla T4 cho các run Kaggle. Runtime đã dùng Python3.10.20, Torch2.4.1+cu121; đây là thông tin lấy từ artifact, không phải một phép benchmark môi trường mới. B0 dùng native Graphdeco commit414b553ef1d3032ea634b7b7a1d121173f36592c. B2 dùng PGSR commitde24f1a38b350387e8d8fe381b2cd70c1ae946e7, plane rasterizer riêng và simple-knn tương ứng; không tráo checkpoint giữa hai phương pháp. [Nguồn và entry points](../metadata/implementation_inventory.json).

Prepared legacy scene có263 ảnh, chia197train/33val/33test và76.223 điểm khởi tạo train-only theo metadata đã lưu. Metadata SHA256 là7636dd218be5cdb1bfa10b76b59a40b2b72cfb6129b0bf2d8fac4283a767913b; split hash làfbd4eb563ba6b5418165ce4d402e04ec123117f56b2fe0c601a35efaad3ba2b4. Seed cấu hình là42. Poses cố định, hệ tọa độ COLMAP và arbitrary monocular scale được giữ nguyên. Do SfM ban đầu tiếp cận ảnh held-out khi ước lượng pose, protocol là fixed-pose transductive, không phải camera recovery hoàn toàn train-only.

Research B0/B2 có RGB1920×1080; B0 low-memory dùng divisor4. Smoke PGSR dùng divisor8,8train views và6000 điểm ban đầu; không thể xem là benchmark toàn cảnh. Các thông số đầy đủ nằm trong `configuration` của từng bản ghi registry. GT geometry không được dùng làm giám sát huấn luyện trong các adapter đã kiểm tra.

## 3. Các mô hình đã triển khai

**Original3DGS (B0)** có trainer, CUDA renderer, full-state resume và evaluator RGB. Research Kaggle và low-memory local đã hoàn thành30000 iterations; các profile được báo riêng.

**PGSR (B2)** có adapter train-only, loss hình học, exposure và full-state checkpoint. Local và Kaggle smoke hoàn thành300 iterations. Research Kaggle dừng có kiểm soát tại21298/30000 do session budget, còn8702 iterations chưa chạy. Recovery có3.733.679 Gaussian và optimizer_steps21298. Trạng thái session Kaggle COMPLETE được kiểm tra lại chỉ phản ánh notebook đã kết thúc, không đồng nghĩa mô hình đã hoàn tất30000 iterations.

**2DGS (B1)** chưa có project trainer/configured experiment/checkpoint được phát hiện. Tài liệu hoặc danh sách baseline không chứng minh đã sử dụng2DGS. GOF và Ours vẫn là kế hoạch; không đưa kết quả giả định của chúng vào bảng measured results.

## 4. Hàm mất mát của từng mô hình

### Original3DGS

Theo `src/gaussian_splatting/runner.py`, `l1` là mean absolute RGB error; `structural=1-SSIM`. Objective là `loss=(1-lambda_dssim)*l1+lambda_dssim*structural`, với lambda_dssim0,2 trong cấu hình hiện có. Log `l1`, `dssim` là các thành phần **chưa nhân trọng số**; `loss` đã tổng hợp. Không phát hiện loss depth, normal hoặc geometry riêng trong B0. Không diễn giải `dssim` thành SSIM similarity hoặc nửa của1-SSIM.

### PGSR

Theo `src/pgsr/geometry.py`, total gồm `rgb + 100*flat + 0.015*single + 0.03*geometric + 0.15*photometric` trong research profile. `flat` là trung bình minimum activated scale trên Gaussian nhìn thấy. `single` so L1 giữa native depth-normal đã alpha-weight và accumulated plane normal, nhân edge weight rồi lấy **mean toàn H×W** sau masked-zero. `geometric` là mean trên accepted pixels của detached exp(-roundtrip_error) nhân roundtrip pixel error. `photometric` là weighted native squared-correlation LNCC trên patch hợp lệ; radius3 tương ứng7×7 trong selected code variant.

Single/multi-view được bật khi iteration>7000 trong research, và>20 trong smoke. Exposure được bật sau1000 ở research, sau10 ở smoke; app L1 chỉ dùng khi canonical structural error<0,5, còn SSIM dùng canonical RGB. Exposure là biến đổi nhánh RGB và nhóm tham số tối ưu, **không phải additive exposure loss độc lập**. `raw`/`weighted`/`coefficients` được lưu riêng. L1 và SSIM thành phần của PGSR **NOT RECORDED**; không được khôi phục giả từ tổng RGB.

Mapping chuẩn giữ `loss_normal=raw.single`, `loss_geometry=raw.multi_geo`, `loss_multiview=raw.multi_ncc`, `loss_regularization=raw.flatten`; không nhập các term khác nhau thành một loss hình học. Chi tiết biến, định nghĩa, trọng số và activation nằm ở từng `loss_definitions.json`. Zero do lịch chưa kích hoạt là quan sát thật; null/ô trống là thiếu dữ liệu. Không suy lịch sử loss từ checkpoint cuối.

## 5. Phân tích quá trình hội tụ

Đã phục hồi 907 quan sát loss từ JSONL và các console records bổ sung khi cần. Không phát hiện TensorBoard event files trong inventory. CSV lưu nguồn và số dòng; console làm tròn được phân biệt với JSONL đủ độ chính xác. Không nội suy các iterations không được log. Wall-clock tuyệt đối không có trong phần lớn loss rows; elapsed time theo segment được lưu riêng, không giả thành UTC.

| Thí nghiệm | Số quan sát | Giá trị đầu ghi nhận | Giá trị cuối ghi nhận | Cực tiểu ghi nhận | Iteration cực tiểu |
| --- | --- | --- | --- | --- | --- |
| b0-kaggle-c55f001d26 | 231 | 0.109641 | 0.078443 | 0.043991 | 22900 |
| ignatius-legacy-b0-low-memory_20261115T141109Z_8ed1c6 | 302 | 0.376780 | 0.058806 | 0.041059 | 17900 |
| ignatius-legacy-b0-research-quality_20261115T222504Z_9ddaaa | 76 | 0.361307 | 0.096965 | 0.096965 | 7400 |
| ignatius-legacy-b0-smoke_20261115T133608Z_4ef4c4 | 0 | NOT RECORDED | NOT RECORDED | NOT RECORDED | NOT RECORDED |
| ignatius-legacy-b0-smoke_20261115T134942Z_346824 | 26 | 0.257724 | 0.077685 | 0.075975 | 850 |
| kaggle-pgsr-geometry-smoke | 30 | 40.325825 | 0.672668 | 0.662114 | 290 |
| local-smoke-v1 | 30 | 40.325741 | 0.674523 | 0.663385 | 290 |
| pgsr-kaggle-c269454133 | 212 | 9.975446 | 0.194697 | 0.083884 | 6700 |

Các giá trị cuối bảng là **last recorded losses**, không phải giá trị tại checkpoint cuối khi checkpoint rơi giữa hai kỳ log. Với PGSR research, loss cuối ghi ở21200, trong khi checkpoint hoàn tất21298;98 iterations cuối không có loss row được phục hồi. Cực tiểu chỉ là minimum sampled training objective; không tự chọn thành best reconstruction hoặc best validation model. Các thống kê mean/std/min/max và số missing cells có trong từng `loss_summary.json`; missing-cell count không đồng nghĩa số iterations không chạy.

PGSR có quá trình giảm nhanh flattening term, sau đó kích hoạt các term hình học nên objective thay đổi thành phần theo lịch. Cực tiểu total được ghi tại6700, trước khi geometry được bật sau7000; không chọn checkpoint6700 làm best geometric model dựa riêng con số này. Các dao động sau activation còn phụ thuộc camera/neighbor được lấy mẫu. Vì vậy, một mức tăng total tại activation không đủ chứng minh phân kỳ; ngược lại, flatten loss nhỏ không chứng minh surface accuracy. Các scheduled-inactive zeros được giữ trong thống kê toàn bộ dữ liệu, không diễn giải thành geometry hoàn hảo.

![Các thành phần loss PGSR research](../figures/training/pgsr-kaggle-c269454133_components.png)

Hình dùng raw observations và thang symlog được ghi rõ, không smoothing. Chỉ số [figure index](../figures/figure_index.json) liên kết đủ các hình PNG/SVG/PDF với nguồn. Có31 nhóm hình, mỗi nhóm ba định dạng; các biểu đồ thiếu nguồn như validation LPIPS hoặc mesh quality được ghi trong danh sách omissions thay vì dựng dữ liệu.

## 6. Đánh giá chất lượng tái tạo

Rendering metrics đã ghi trên33test views của B0 full-resolution là PSNR20,913033dB, SSIM0,695186 và LPIPS0,269434. B0 low-memory có PSNR23,375393dB, SSIM0,839926 và LPIPS0,115017 **tại độ phân giải khác**. Đây là số đo lịch sử đã đối chiếu JSON nguồn, không phải metrics được tính lại bằng suy luận trong kiểm toán này.

PGSR research chưa có final test evaluation được xác minh ở21298. Các metric PGSR smoke chỉ kiểm tra chức năng evaluator ở resolution divisor8. B0 dùng rounded uint8 PNG; PGSR float diagnostic không được trộn trực tiếp với PNG protocol. Tất cả test records, periodic validation và training-view scores được tách. Không fit exposure từ ảnh test.

![Rendering diagnostics](../figures/comparisons/rendering_quality.png)

Không có GT alignment, ROI, unit và threshold đủ để báo official TNT geometry score. Chamfer Distance, dense depth error và F-score được để trống, không thay bằng sparse SfM fit hoặc một loss geometric trong training.

## 7. So sánh 3DGS, 2DGS và PGSR

| Experiment | Resolution divisor | Iterations | PSNR (dB) | SSIM | LPIPS | Protocol |
| --- | --- | --- | --- | --- | --- | --- |
| b0-kaggle-c55f001d26 | 1 | 30000 | 20.913033 | 0.695186 | 0.269434 | rounded_uint8_png |
| ignatius-legacy-b0-low-memory_20261115T141109Z_8ed1c6 | 4 | 30000 | 23.375393 | 0.839926 | 0.115017 | rounded_uint8_png |
| ignatius-legacy-b0-research-quality_20261115T222504Z_9ddaaa | 1 | 7400 | NOT RECORDED | NOT RECORDED | NOT RECORDED | NOT RECORDED |
| ignatius-legacy-b0-smoke_20261115T133608Z_4ef4c4 | 4 | 0 | NOT RECORDED | NOT RECORDED | NOT RECORDED | NOT RECORDED |
| ignatius-legacy-b0-smoke_20261115T134942Z_346824 | 4 | 1200 | 15.026311 | 0.485911 | 0.417408 | rounded_uint8_png |
| kaggle-pgsr-geometry-smoke | 8 | 300 | 9.896437 | 0.225452 | 0.550352 | float_render_diagnostic |
| local-smoke-v1 | 8 | 300 | 9.826858 | 0.222708 | 0.556722 | rounded_uint8_png |
| pgsr-kaggle-c269454133 | 1 | 21298 | NOT RECORDED | NOT RECORDED | NOT RECORDED | NOT RECORDED |
| B1_2DGS_NOT_TRAINED | N/A | N/A | NOT RECORDED | NOT RECORDED | NOT RECORDED | NOT TRAINED |

Đây là bảng mô tả, không phải xếp hạng công bằng giữa ba phương pháp. 2DGS chưa train; PGSR research chưa hoàn tất và chưa có final RGB score; các smoke budgets và image resolutions khác nhau. B0 low-memory score cao hơn full-resolution không chứng minh thuật toán/profile đó tốt hơn trong cùng protocol. Pairwise decisions về split/resolution/metric definition/iterations có trong [comparability audit](../comparisons/comparability_audit.json). Không so trực tiếp total loss của RGB-only B0 với PGSR gồm các geometric regularizers có trọng số lớn.

## 8. Đánh giá hiệu quả tính toán

| Experiment | Logged-segment seconds (lower bound) | Reported duration (scope in registry) | GPU | Gaussians |
| --- | --- | --- | --- | --- |
| b0-kaggle-c55f001d26 | 8848.235 | 8908.292 | Tesla T4 | 3381717 |
| ignatius-legacy-b0-low-memory_20261115T141109Z_8ed1c6 | 5953.923 | 171.410 | NVIDIA GeForce RTX 3050 6GB Laptop GPU | 249213 |
| ignatius-legacy-b0-research-quality_20261115T222504Z_9ddaaa | 2967.191 | 179.596 | NVIDIA GeForce RTX 3050 6GB Laptop GPU | 2379021 |
| ignatius-legacy-b0-smoke_20261115T133608Z_4ef4c4 | NOT RECORDED | NOT RECORDED | NVIDIA GeForce RTX 3050 6GB Laptop GPU | UNKNOWN |
| ignatius-legacy-b0-smoke_20261115T134942Z_346824 | 36.064 | 37.919 | NVIDIA GeForce RTX 3050 6GB Laptop GPU | 78582 |
| kaggle-pgsr-geometry-smoke | 13.952 | 14.299 | Tesla T4 | 12899 |
| local-smoke-v1 | 25.967 | 26.623 | NVIDIA GeForce RTX 3050 6GB Laptop GPU | 12699 |
| pgsr-kaggle-c269454133 | 17294.433 | 17457.767 | Tesla T4 | 3733679 |

Thời gian logged-segment là tổng cực đại elapsed đã ghi theo segment, chỉ là **cận dưới quan sát được**. Đặc biệt, report cuối của low-memory B0 ghi171,409898giây cho segment bắt đầu25001, không phải toàn bộ30000 iterations; cận dưới từ log là5953,923giây. B0 Kaggle dùng checkpoint7000 làm đầu vào nên thời gian segment Kaggle không bao gồm local stages trước đó. PGSR report ghi17457,766866giây cho session21298 iterations, bao gồm overhead theo scope đã lưu; không đánh đồng với kernel-only time.

Peak Torch allocated/reserved và sampled device-wide VRAM được phân biệt trong registry. Earlier unreported peaks vẫn UNKNOWN; không coi peak của segment cuối là whole-run peak. Peak system RAM, standardized inference FPS và mesh extraction time không có nguồn đủ để báo. Không so tốc độ RTX3050 với T4 như một controlled method benchmark.

![Tài nguyên theo scope đã ghi](../figures/comparisons/resources.png)

## 9. Phân tích chất lượng mesh

Danh mục project geometry không phát hiện predicted triangle mesh: các PLY đã đọc header là point cloud/Gaussian cloud, không phải surface mesh; ground-truth scan được phân loại riêng. File Gaussian PLY không chứng minh đã có mesh kín. Không có số liệu có căn cứ về connected components, watertightness, surface area hoặc số faces của reconstructed surface. Những metric này là **NOT MEASURED**, không phải0. Chưa có project mesh-extraction entry point được kiểm chứng; native meshing scripts trong third-party source không là evidence đã chạy workflow.

## 10. Hạn chế và vấn đề tồn tại

Các giới hạn chính gồm nghiên cứu một cảnh/seed, transductive poses, lịch sử log thưa, thiếu PGSR L1/SSIM riêng, thiếu validation LPIPS, thiếu matched-resolution three-method baseline và chưa có GT geometry protocol. PGSR export trạng thái EXPORTED_PENDING_SEPARATE_RELOAD; không nâng lên final inference/Phase04 readiness nếu chưa có bước kiểm chứng tương ứng.

Kiểm toán restricted CPU đã đọc58 checkpoint/weight files,53 unique SHA256;56 tệp theo full-state schema của dự án (bao gồm unit fixtures) qua finite/layout/Adam/RNG checks và2 pretrained/other weights đọc được. Các fixture không được tính thành nghiên cứu huấn luyện cảnh thật. Có5 checkpoint copies byte-identical; original flat PGSR recovery vẫn ESSENTIAL do yêu cầu layout, các bản trùng còn lại không bị xóa khi chưa có approval. Equality theo hash không đồng nghĩa redundancy về chức năng. CUDA restore không chạy lại trong đợt kiểm toán; structural validity không tự chứng minh CUDA kernel/ABI trên máy tương lai.

PGSR canonical import đã chuyển config/identity vào `configs/` và training logs/results vào `logs/`, không khớp flat layout mà current CLI yêu cầu. Original flat `phase03_results/recovery/run` vẫn còn và là artifact cần bảo toàn. Không âm thầm sửa sealed manifest hoặc đổi identity để gọi là resume thành công. Best validation candidate chỉ dựa các validation records đã có; checkpoint availability được kiểm riêng, không suy từ training minimum.

Trước cleanup, inventory ghi18926 files với52500485363logical bytes. Cleanup chỉ xóa127 file bytecode/tool-cache đã kiểm chứng, tổng953633logical bytes;0 model/data/log artifacts bị xóa. 897 candidate copies, tổng tiềm năng12342451848bytes, được giữ nguyên trong [approval list](../cleanup/approval_required.csv). Hash equality không loại bỏ các dependency theo đường dẫn hoặc sealed-bundle membership. Con số tiết kiệm là tổng độ dài file thực tế đã unlink, không giả là physical free-space gain khi có concurrent writes. [Manifest thao tác](../cleanup/cleanup_manifest.json) ghi đầy đủ guard và kết quả.

## 11. Kết luận và định hướng cải tiến

Kết quả đủ căn cứ hiện nay là baseline B0 có các run30k và RGB evaluation, cùng PGSR đã thực thi optimization/hình học và lưu recovery21298. Chưa có2DGS hoặc reconstructed mesh benchmark. Các bằng chứng này cho phép mô tả pipeline thực thi, loss dynamics và giới hạn tài nguyên, nhưng chưa hỗ trợ tuyên bố phương pháp vượt trội hoặc SOTA.

Ưu tiên tiếp theo là bảo toàn recovery PGSR, xử lý layout handoff có kiểm chứng, và chỉ khi có authorization mới tiếp tục30000 cùng final export/evaluation. Sau đó tích hợp2DGS bằng thí nghiệm riêng, thống nhất resolution/splits/metric protocols, xác lập GT alignment/mesh evaluator và mở rộng nhiều seed/cảnh. Nên tăng khả năng quan sát bằng logging các RGB subterms/exposure diagnostics và UTC/segment metadata trong **run tương lai có version**, không sửa lịch sử hoặc giả bổ sung dữ liệu cho các run đã hoàn tất.

Tài liệu thực thi và giới hạn tái lập: [reproducibility guide](../../docs/EXPERIMENT_REPRODUCIBILITY.md). Dữ liệu đầy đủ: `research/metrics/<experiment_id>/loss_history.csv`, `loss_definitions.json`, `loss_summary.json` và `validation_history.csv`. Mọi trị số trong bảng có thể truy về source paths/line numbers hoặc JSON được liệt kê; các giá trị thiếu được công bố rõ.
