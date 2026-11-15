# Evaluation specification — proposed protocol v1

Nguồn: P02 §7 (held-out rendering), P01 §6, P14 §V, P05 App.B, P07 §4 và P11 §6. Đây là protocol dự án, chưa có benchmark result. [Data/split](data_specification.md) xác định fixed-pose **transductive SfM**, không strict train-only camera recovery.

## 1. Comparability key và budgets

Comparison group phải cùng `dataset_id`, accepted image IDs, resolution/transforms, K/poses/normalization, split hash, mask/color-space, evaluation code/weights, GT units/alignment/thresholds và hardware profile. Chỉ thay method hoặc ablation được nghiên cứu.

B0/B1/B2 bắt buộc; B3 optional phải ghi `SKIPPED`/blocker nếu không khả thi, không loại scene thất bại rồi báo mean như đủ scenes. Ours giữ B2 base settings. Research30k iterations là common proposed budget, training time/Gaussian count report thêm; iteration equal không đồng nghĩa FLOPs equal. Full native/paper budget track riêng; capped6GB profile là resource-constrained comparison, không exact paper reproduction.

Final checkpoint30k cho primary table; validation-best diagnostic nếu preregister. Seeds42/43/44 đề xuất, cùng initialization point selection và base camera sampling khi phù hợp. Report perseed/perview/perscene, mean±std; một pilot scene không chứng minh statistically general improvement hoặc SOTA. Không đặt quality acceptance target bằng paper metrics.

## 2. Held-out RGB

Test frames chỉ được đọc ở evaluator sau config/checkpoint freeze. Validation dùng cho tuning; không fit exposure từ test RGB hoặc dùng test image làm neighbor. Raw canonical appearance là primary; test-time exposure-fitted score nếu muốn là separate diagnostic.

Evaluate RGB trong `[0,1]`, cùng sRGB convention; không chuyển linear một baseline. Common image/mask/background, fullimage primary trừ dataset officialmask có nguồn. PSNR theo mỗi view:

```math
\mathrm{MSE}=\frac{1}{3|V|}\sum_{p\in V,c}(C_{pc}-I_{pc})^2,\quad
\mathrm{PSNR}=-10\log_{10}\mathrm{MSE}.
```

MSE0→infinity ghi JSON-compatible policy tường minh, không clamp để tăng/giảm score tùy ý. Aggregate arithmetic mean per-view PSNR, không đổi thành PSNR của pooledMSE mà dùng cùng tên.

SSIM: pin implementation, gaussianwindow11×11/σ1.5, data_range1, channelreduction/borderpolicy rõ; nếu chọn officialbaseline evaluator khác phải cùng choice cho cả group. MaskedSSIM không áp mask bằng blackfill rồi coi equivalent; crop/validwindow rule fixed. LPIPS: proposed AlexNet variant, official package/version+weight hash, RGB map `[0,1]→[-1,1]`, same padding/mask policy. Không download LPIPS trong phiên này; thiếu weights→metric chưa đo, không0.

## 3. Geometry/mesh metrics

Ignatius GT hiện chỉ được inventory1file, chưa xác minh unit/alignment/masks/official scene threshold. [TNT benchmark website](https://www.tanksandtemples.org/benchmark/) không truy cập được bằng web tool trong audit. **Official TNT score hiện chưa đủ prerequisites**, không lấy threshold5cm của DN-Splatter indoor áp vào Ignatius.

Trước official evaluation phải có: GT hash/unit, official evalsceneROI/masks, source/model registration transform, per-scene threshold, evaluator revision và sampling/outlier rules. Thiếu một thông tin ảnh hưởng metric → `BLOCKED_EVALUATION`, giữ mesh và report lý do. Có thể báo generic diagnostic riêng khi định nghĩa đầy đủ, không gắn label officialTNT.

Generic diagnostic đề xuất: sample200k points trên predicted triangle area, seeded; GT là mesh thì area-sample200k, GT là scan pointcloud thì giữ/raw hoặc downsample deterministic với policy/hash rõ. Không sample uniformly theo vertices vì mesh tessellation sẽ bias. Định nghĩa Euclidean nearest-neighbor distances `d_P(p)=min_g||p−g||₂`, `d_G(g)=min_p||g−p||₂`:

```math
\mathrm{Accuracy}=\operatorname{mean}_{p\in P}d_P(p),\quad
\mathrm{Completion}=\operatorname{mean}_{g\in G}d_G(g),\quad
\mathrm{CD}_{sym,unsquared}=\tfrac12(\mathrm{Accuracy}+\mathrm{Completion}).
```

Đây là mean **unsquared Euclidean** distance; không tự gọi là squaredChamfer hoặc component-wise L1norm. Lưu metricname/norm/squared flag/unit. `Precisionτ=mean_P[d_P<τ]`, `Recallτ=mean_G[d_G<τ]`, `F1τ=2PR/(P+R)`; P+R0→0, **emptyP/G là invalid metric**, không đánh đồng0. Report raw0..1; nếu bảngpercent thì nhân100 và ghi `%`. Thresholds chỉ có nghĩa trong declared units. Accuracy không là classification accuracy; Completion không thay Recall.

Normal consistency cần oriented/unoriented choice: dùng absolute dot cho unoriented diagnostic hoặc signed dot khi orientation đã verified; ghi nearest-neighbor rule và symmetry. Depth metrics (AbsRel/RMSE/normalangular error) chỉ khi có valid GT depth/normal protocol, không báo bằng sparseSfM residual như denseGT accuracy. Sparse anchor fit là diagnostic khác.

## 4. Scale, alignment và masks

Đảo project normalization về SfMcoordinates trước registration. Monocular scale arbitrary; metric mm/meters chỉ sau declared Sim(3) registration to GT trong evaluator. Prefer một frozen source-SfM→GT transform dùng chung cho các methods cùng source. Không refine mỗi method bằng ICP khác config để tăng scores. Nếu officialprotocol cho perprediction alignment thì dùng đúng cùng algorithm/budget, report pre/post alignment và transforms.

GT alignment chỉ phục vụ evaluation; không truyền transform/GT points vào initialization, confidence, loss, frame selection, meshingROI tuning. Official evalmask/visibilitycrop cũng chỉ evaluator. Không kết luận better background geometry khi GT chỉ có foreground (P07 §4.2, P11 §6).

## 5. Native/common mesh và performance

Native extractor track và commonTSDF track separate tables theo [comparison](../research/method_comparison.md). Report raw và cleaned mesh stats: vertices/faces/components, boundary/nonmanifold edges, area/bounds/filebytes. Cleaning/decimation/holes protocol pin, không tự giảm meshes để đạt metric.

Performance: wallclock riêng preparation/training/rendering/fusion/extraction/evaluation; CUDA events cho kernels khi phù hợp, synchronization trước timedsection. FPS có resolution, camera path, warmup, renderer và numberframes; không dùng network time làm wholepipelineFPS. Measured peak allocated/reserved khác device-wide sampling; báo units/scope/sample interval. So sánh B0/Ours phải gồm overhead confidence/precompute, không chỉ optimizerloop.

## 6. Reporting/acceptance

Các bảng measured ban đầu không có số, metrics `null` và status `NOT_RUN` trong report tương lai. Paper tables ở literature có nguồn riêng, không ghép thành kết quả local. Report từng failed/skippedrun và sốsuccess/total, không quietly dropOOM. Qualitative camera/crops/color mapping identical, dùng predetermined views và artifact links.

Acceptance evaluator: synthetic distance/scale fixtures, empty/invalidinput handling, deterministic sampling, mask/alignment roundtrip, known PSNR/identitySSIM/LPIPS checks, rawmetric JSON có provenance; final reports reproducible từ frozen artifacts. Chỉ sau đó mới gắn experimentallyvalidated result. Không có thresholdSOTA hoặc acceptancequality giả trong documentation.
