"""Generate English developer summaries and the formal Vietnamese academic report."""
from __future__ import annotations

from collections import Counter
import csv
from pathlib import Path

from discover import ROOT, RESEARCH, read_json


def fmt(value, digits: int = 6) -> str:
    return "NOT RECORDED" if value is None or value=="" else f"{float(value):.{digits}f}"


def table(headers: list[str], rows: list[list]) -> str:
    return "| "+" | ".join(headers)+" |\n| "+" | ".join("---" for _ in headers)+" |\n"+"\n".join("| "+" | ".join(str(v) for v in row)+" |" for row in rows)


def main() -> None:
    runs=read_json(RESEARCH/"metadata/experiment_registry.json")
    cleanup=read_json(RESEARCH/"cleanup/cleanup_manifest.json",{})
    discovery=read_json(RESEARCH/"metadata/discovery.json")
    implementation=read_json(RESEARCH/"metadata/implementation_inventory.json")
    figures=read_json(RESEARCH/"figures/figure_index.json")
    with (RESEARCH/"comparisons/model_comparison.csv").open(encoding="utf-8") as stream:
        comparison=list(csv.DictReader(stream))
    with (RESEARCH/"metadata/checkpoint_audit.csv").open(encoding="utf-8") as stream:
        checkpoints=list(csv.DictReader(stream))
    statuses=Counter(r["training_status"] for r in runs)
    losses=[]
    for run in runs:
        summary=read_json(RESEARCH/"metrics"/run["experiment_id"]/"loss_summary.json")["components"]["loss_total"]
        losses.append([run["experiment_id"],summary["valid_observations"],fmt(summary["initial_recorded_value"]),fmt(summary["final_recorded_value"]),fmt(summary["minimum"]),summary["iteration_of_minimum"] if summary["iteration_of_minimum"] is not None else "NOT RECORDED"])
    run_table=table(["Experiment ID","Model","Observed iteration / target","Status","Loss rows"],[[r["experiment_id"],r["model"],f"{r['actual_observed_completed_iteration']}/{r['target_iterations']}",r["training_status"],r["loss_observations"]] for r in runs])
    score_table=table(["Experiment","Resolution divisor","Iterations","PSNR (dB)","SSIM","LPIPS","Protocol"],[[r["experiment_id"],r.get("resolution_divisor","") or "N/A",r.get("completed_iterations","") or "N/A",fmt(r.get("psnr")),fmt(r.get("ssim")),fmt(r.get("lpips")),r.get("metric_protocol","") or "NOT TRAINED"] for r in comparison])
    comparison_md="# Recorded model comparison\n\nAudit date: 2026-11-22. These are previously recorded results; no model was trained or evaluated during this audit.\n\n"+score_table+"\n\n"
    comparison_md+="There is no completed, matched-protocol three-method comparison. B1 2DGS was not trained. B2 research stopped at21298/30000 and has no recorded final RGB evaluation. B0 full-resolution and low-memory use different resolutions; their scores must not be interpreted as a controlled superiority result. Development smoke scores are diagnostics.\n\n"
    comparison_md+="PGSR total loss includes weighted geometric terms; B0 total loss is RGB-only. Never rank their total objectives numerically. Empty fields mean unavailable, not zero. Test and periodic validation results are separate. See [pairwise audit](comparability_audit.json) and [registry](../metadata/experiment_registry.json).\n\n"
    comparison_md+="Runtime columns are summed maximum elapsed values from observed log segments, hence lower bounds. GPU models and iteration windows differ. Reported last-segment durations and memory scopes remain in the registry. No FPS, system-RAM peak, dense geometry error or reconstructed mesh quality was recovered.\n"
    (RESEARCH/"comparisons/model_comparison.md").write_text(comparison_md,encoding="utf-8")
    status="# Research status\n\nVerified audit date: 2026-11-22. This compact status supersedes execution claims in older plans; historical reports remain unchanged.\n\n"+run_table+"\n\n"
    status+="**Implemented/trained methods:** B0 Original3DGS and B2 PGSR. **2DGS has not been used in a project training run.** GOF, Ours and project mesh extraction are not implemented/verified. Completed smoke schedules are not completed research baselines.\n\n"
    status+="**PGSR:** original Kaggle notebook reports session COMPLETE, but its saved model is PARTIAL_SESSION_BUDGET at21298/30000,3733679 Gaussians. Full-state CPU integrity is verified. No audit training/resume/inference was launched. The canonical reorganized folder is not directly resume-compatible; preserve the original flat `phase03_results/recovery/run` and its seal. A future continuation needs explicit authorization.\n\n"
    status+="**B0:** full-resolution Kaggle30k has33 test views (PSNR20.913033, SSIM0.695186, LPIPS0.269434). Low-memory30k has33 test views at resolution divisor4 (23.375393/0.839926/0.115017). Those resolutions differ. Local full-resolution training stopped after logged7400, with published checkpoint7000.\n\n"
    status+="**Evidence:** "+str(sum(r["loss_observations"] for r in runs))+" recovered loss rows;58 checkpoint/weight files,53 unique hashes;56 restricted-load project-format full-state checks (including unit fixtures) and2 pretrained/other weight reads. No TensorBoard events found. No reconstructed predicted triangle mesh was found in indexed project geometry; no GT accuracy/Chamfer/F-score was measured.\n\n"
    status+="**Limits:** PGSR standalone L1/SSIM and additive exposure-loss histories were not recorded; validation LPIPS histories are absent. Raw-video SfM remains GPU-BA blocked; successful legacy image-set experiments do not change that. Single-seed, single-scene, transductive fixed-pose experiments do not support SOTA claims.\n\n"
    status+="**Next steps, not executed:** approve a PGSR continuation if desired; repair/qualify the run-layout handoff without modifying the original seal; finish30k and validate export/evaluation; integrate B1 2DGS only under a separate training request; establish mesh/GT protocol before geometry claims.\n\n"
    status+="Indexes: [experiments](../research/metadata/experiment_registry.json), [loss histories](../research/metrics/), [comparison](../research/comparisons/model_comparison.md), [cleanup approval list](../research/cleanup/approval_required.csv), [Vietnamese report](../research/reports/bao_cao_phan_tich_huan_luyen.md).\n"
    (ROOT/"docs/RESEARCH_STATUS.md").write_text(status,encoding="utf-8")
    approval=[r for r in cleanup.get("candidates",[]) if r["proposed_action"]=="REQUIRE_USER_APPROVAL"]
    original=cleanup.get("baseline",discovery)
    removed=cleanup.get("removed_count",0)
    saved=cleanup.get("logical_bytes_removed",0)
    efficiency=table(["Experiment","Logged-segment seconds (lower bound)","Reported duration (scope in registry)","GPU","Gaussians"],[[r["experiment_id"],fmt(r["training_duration_seconds"],3),fmt(r["reported_training_seconds"],3),r["gpu"] or "UNKNOWN",r["final_gaussians"] if r["final_gaussians"] is not None else "UNKNOWN"] for r in runs])
    loss_table=table(["Thí nghiệm","Số quan sát","Giá trị đầu ghi nhận","Giá trị cuối ghi nhận","Cực tiểu ghi nhận","Iteration cực tiểu"],losses)
    report=f'''# Báo cáo phân tích huấn luyện và kiểm toán thực nghiệm tái tạo cảnh 3D

Ngày kiểm toán: **22/11/2026**. Đối tượng: repository 3DGS Mesh Reconstruction, cảnh Ignatius. Báo cáo này được tổng hợp từ artifact đã tồn tại; không tiến hành huấn luyện lại, tiếp tục huấn luyện, suy luận mô hình hoặc tạo mesh trong đợt kiểm toán. Số liệu đo trước đây được phân biệt với các thống kê mới tính từ log. [Danh mục thí nghiệm](../metadata/experiment_registry.json), [kiểm toán checkpoint](../metadata/checkpoint_audit.csv) và [bảng so sánh](../comparisons/model_comparison.csv) là nguồn dữ liệu định lượng chính.

## 1. Tổng quan quá trình thực nghiệm

Pipeline thực tế gồm preprocessing/SfM, chuẩn bị dữ liệu COLMAP, huấn luyện Gaussian và đánh giá ảnh RGB. Nhánh video monocular vẫn bị chặn ở GPU Bundle Adjustment; không có bằng chứng hoàn tất tái tạo video GPU-strict. Các kết quả huấn luyện sử dụng **thí nghiệm legacy image_set riêng đã được cho phép**, không phải đầu ra thành công của nhánh video. SfM legacy có lịch sử CPU BA; việc này được công bố thay vì gán nhãn GPU-strict cho toàn bộ pipeline.

Kiểm toán phân biệt {len(runs)} bản ghi thí nghiệm sau khi hợp nhất các bản sao có log nội dung giống hệt: {statuses.get('COMPLETED',0)} lịch trình đã hoàn thành, {statuses.get('INTERRUPTED',0)} bị gián đoạn và {statuses.get('UNKNOWN',0)} trường hợp chưa đủ bằng chứng về các bước huấn luyện. Trạng thái COMPLETED của smoke chỉ có nghĩa hoàn thành lịch trình kiểm thử ngắn. Sự tồn tại của checkpoint không được dùng làm tiêu chí duy nhất cho việc hoàn thành.

{run_table}

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

Đã phục hồi {sum(r['loss_observations'] for r in runs)} quan sát loss từ JSONL và các console records bổ sung khi cần. Không phát hiện TensorBoard event files trong inventory. CSV lưu nguồn và số dòng; console làm tròn được phân biệt với JSONL đủ độ chính xác. Không nội suy các iterations không được log. Wall-clock tuyệt đối không có trong phần lớn loss rows; elapsed time theo segment được lưu riêng, không giả thành UTC.

{loss_table}

Các giá trị cuối bảng là **last recorded losses**, không phải giá trị tại checkpoint cuối khi checkpoint rơi giữa hai kỳ log. Với PGSR research, loss cuối ghi ở21200, trong khi checkpoint hoàn tất21298;98 iterations cuối không có loss row được phục hồi. Cực tiểu chỉ là minimum sampled training objective; không tự chọn thành best reconstruction hoặc best validation model. Các thống kê mean/std/min/max và số missing cells có trong từng `loss_summary.json`; missing-cell count không đồng nghĩa số iterations không chạy.

PGSR có quá trình giảm nhanh flattening term, sau đó kích hoạt các term hình học nên objective thay đổi thành phần theo lịch. Cực tiểu total được ghi tại6700, trước khi geometry được bật sau7000; không chọn checkpoint6700 làm best geometric model dựa riêng con số này. Các dao động sau activation còn phụ thuộc camera/neighbor được lấy mẫu. Vì vậy, một mức tăng total tại activation không đủ chứng minh phân kỳ; ngược lại, flatten loss nhỏ không chứng minh surface accuracy. Các scheduled-inactive zeros được giữ trong thống kê toàn bộ dữ liệu, không diễn giải thành geometry hoàn hảo.

![Các thành phần loss PGSR research](../figures/training/pgsr-kaggle-c269454133_components.png)

Hình dùng raw observations và thang symlog được ghi rõ, không smoothing. Chỉ số [figure index](../figures/figure_index.json) liên kết đủ các hình PNG/SVG/PDF với nguồn. Có{len(figures['figures'])} nhóm hình, mỗi nhóm ba định dạng; các biểu đồ thiếu nguồn như validation LPIPS hoặc mesh quality được ghi trong danh sách omissions thay vì dựng dữ liệu.

## 6. Đánh giá chất lượng tái tạo

Rendering metrics đã ghi trên33test views của B0 full-resolution là PSNR20,913033dB, SSIM0,695186 và LPIPS0,269434. B0 low-memory có PSNR23,375393dB, SSIM0,839926 và LPIPS0,115017 **tại độ phân giải khác**. Đây là số đo lịch sử đã đối chiếu JSON nguồn, không phải metrics được tính lại bằng suy luận trong kiểm toán này.

PGSR research chưa có final test evaluation được xác minh ở21298. Các metric PGSR smoke chỉ kiểm tra chức năng evaluator ở resolution divisor8. B0 dùng rounded uint8 PNG; PGSR float diagnostic không được trộn trực tiếp với PNG protocol. Tất cả test records, periodic validation và training-view scores được tách. Không fit exposure từ ảnh test.

![Rendering diagnostics](../figures/comparisons/rendering_quality.png)

Không có GT alignment, ROI, unit và threshold đủ để báo official TNT geometry score. Chamfer Distance, dense depth error và F-score được để trống, không thay bằng sparse SfM fit hoặc một loss geometric trong training.

## 7. So sánh 3DGS, 2DGS và PGSR

{score_table}

Đây là bảng mô tả, không phải xếp hạng công bằng giữa ba phương pháp. 2DGS chưa train; PGSR research chưa hoàn tất và chưa có final RGB score; các smoke budgets và image resolutions khác nhau. B0 low-memory score cao hơn full-resolution không chứng minh thuật toán/profile đó tốt hơn trong cùng protocol. Pairwise decisions về split/resolution/metric definition/iterations có trong [comparability audit](../comparisons/comparability_audit.json). Không so trực tiếp total loss của RGB-only B0 với PGSR gồm các geometric regularizers có trọng số lớn.

## 8. Đánh giá hiệu quả tính toán

{efficiency}

Thời gian logged-segment là tổng cực đại elapsed đã ghi theo segment, chỉ là **cận dưới quan sát được**. Đặc biệt, report cuối của low-memory B0 ghi171,409898giây cho segment bắt đầu25001, không phải toàn bộ30000 iterations; cận dưới từ log là5953,923giây. B0 Kaggle dùng checkpoint7000 làm đầu vào nên thời gian segment Kaggle không bao gồm local stages trước đó. PGSR report ghi17457,766866giây cho session21298 iterations, bao gồm overhead theo scope đã lưu; không đánh đồng với kernel-only time.

Peak Torch allocated/reserved và sampled device-wide VRAM được phân biệt trong registry. Earlier unreported peaks vẫn UNKNOWN; không coi peak của segment cuối là whole-run peak. Peak system RAM, standardized inference FPS và mesh extraction time không có nguồn đủ để báo. Không so tốc độ RTX3050 với T4 như một controlled method benchmark.

![Tài nguyên theo scope đã ghi](../figures/comparisons/resources.png)

## 9. Phân tích chất lượng mesh

Danh mục project geometry không phát hiện predicted triangle mesh: các PLY đã đọc header là point cloud/Gaussian cloud, không phải surface mesh; ground-truth scan được phân loại riêng. File Gaussian PLY không chứng minh đã có mesh kín. Không có số liệu có căn cứ về connected components, watertightness, surface area hoặc số faces của reconstructed surface. Những metric này là **NOT MEASURED**, không phải0. Chưa có project mesh-extraction entry point được kiểm chứng; native meshing scripts trong third-party source không là evidence đã chạy workflow.

## 10. Hạn chế và vấn đề tồn tại

Các giới hạn chính gồm nghiên cứu một cảnh/seed, transductive poses, lịch sử log thưa, thiếu PGSR L1/SSIM riêng, thiếu validation LPIPS, thiếu matched-resolution three-method baseline và chưa có GT geometry protocol. PGSR export trạng thái EXPORTED_PENDING_SEPARATE_RELOAD; không nâng lên final inference/Phase04 readiness nếu chưa có bước kiểm chứng tương ứng.

Kiểm toán restricted CPU đã đọc{len(checkpoints)} checkpoint/weight files,53 unique SHA256;56 tệp theo full-state schema của dự án (bao gồm unit fixtures) qua finite/layout/Adam/RNG checks và2 pretrained/other weights đọc được. Các fixture không được tính thành nghiên cứu huấn luyện cảnh thật. Có5 checkpoint copies byte-identical; original flat PGSR recovery vẫn ESSENTIAL do yêu cầu layout, các bản trùng còn lại không bị xóa khi chưa có approval. Equality theo hash không đồng nghĩa redundancy về chức năng. CUDA restore không chạy lại trong đợt kiểm toán; structural validity không tự chứng minh CUDA kernel/ABI trên máy tương lai.

PGSR canonical import đã chuyển config/identity vào `configs/` và training logs/results vào `logs/`, không khớp flat layout mà current CLI yêu cầu. Original flat `phase03_results/recovery/run` vẫn còn và là artifact cần bảo toàn. Không âm thầm sửa sealed manifest hoặc đổi identity để gọi là resume thành công. Best validation candidate chỉ dựa các validation records đã có; checkpoint availability được kiểm riêng, không suy từ training minimum.

Trước cleanup, inventory ghi{original.get('file_count',0)} files với{original.get('total_bytes',0)}logical bytes. Cleanup chỉ xóa{removed} file bytecode/tool-cache đã kiểm chứng, tổng{saved}logical bytes;0 model/data/log artifacts bị xóa. {len(approval)} candidate copies, tổng tiềm năng{sum(r['bytes'] for r in approval)}bytes, được giữ nguyên trong [approval list](../cleanup/approval_required.csv). Hash equality không loại bỏ các dependency theo đường dẫn hoặc sealed-bundle membership. Con số tiết kiệm là tổng độ dài file thực tế đã unlink, không giả là physical free-space gain khi có concurrent writes. [Manifest thao tác](../cleanup/cleanup_manifest.json) ghi đầy đủ guard và kết quả.

## 11. Kết luận và định hướng cải tiến

Kết quả đủ căn cứ hiện nay là baseline B0 có các run30k và RGB evaluation, cùng PGSR đã thực thi optimization/hình học và lưu recovery21298. Chưa có2DGS hoặc reconstructed mesh benchmark. Các bằng chứng này cho phép mô tả pipeline thực thi, loss dynamics và giới hạn tài nguyên, nhưng chưa hỗ trợ tuyên bố phương pháp vượt trội hoặc SOTA.

Ưu tiên tiếp theo là bảo toàn recovery PGSR, xử lý layout handoff có kiểm chứng, và chỉ khi có authorization mới tiếp tục30000 cùng final export/evaluation. Sau đó tích hợp2DGS bằng thí nghiệm riêng, thống nhất resolution/splits/metric protocols, xác lập GT alignment/mesh evaluator và mở rộng nhiều seed/cảnh. Nên tăng khả năng quan sát bằng logging các RGB subterms/exposure diagnostics và UTC/segment metadata trong **run tương lai có version**, không sửa lịch sử hoặc giả bổ sung dữ liệu cho các run đã hoàn tất.

Tài liệu thực thi và giới hạn tái lập: [reproducibility guide](../../docs/EXPERIMENT_REPRODUCIBILITY.md). Dữ liệu đầy đủ: `research/metrics/<experiment_id>/loss_history.csv`, `loss_definitions.json`, `loss_summary.json` và `validation_history.csv`. Mọi trị số trong bảng có thể truy về source paths/line numbers hoặc JSON được liệt kê; các giá trị thiếu được công bố rõ.
'''
    path=RESEARCH/"reports/bao_cao_phan_tich_huan_luyen.md"
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(report,encoding="utf-8")
    summary=table(["Run","Rows","Initial recorded total","Last recorded total","Minimum recorded total","Iteration of minimum"],losses)
    (RESEARCH/"comparisons/loss_summary.md").write_text("# Recorded loss summary\n\nTotals have different objective definitions and cannot be ranked across methods.\n\n"+summary+"\n",encoding="utf-8")
    print(summary)
    print("Formal Vietnamese report: "+path.relative_to(ROOT).as_posix())


if __name__=="__main__":main()
