# Literature review

Nguồn PDF, hash, phiên bản và official code: [inventory](paper_inventory.md). `p.` luôn là trang vật lý PDF local. Mỗi hàng nêu vị trí nguồn; các khuyến nghị tích hợp ở hàng 20 là **đánh giá thiết kế**, chưa phải kết quả thực nghiệm. Số liệu paper không phải benchmark trên RTX 3050 hoặc Ignatius video hiện tại. Công thức và cách tích hợp chi tiết: [mathematical foundations](mathematical_foundations.md).

## P01

**2D Gaussian Splatting for Geometrically Accurate Radiance Fields** — Huang, Yu, Chen, Geiger, Gao; SIGGRAPH 2024, local arXiv v3 2025. Nguồn chính: p.1; §4–6 pp.4–8, appendix pp.10–13.

| # / Trường | Phân tích có nguồn |
| --- | --- |
| 1. Bibliography | Năm công bố 2024; DOI 10.1145/3641519.3657428 (p.1), không đổi thành 2025 vì revision. |
| 2. Problem | Gaussian volume không xác định một surface nhất quán giữa các view (§3 p.3). |
| 3. Motivation | Kết hợp tốc độ splatting và geometry của surfels, tránh cần dense GT normals (pp.2–4). |
| 4. Contributions | Oriented disks; perspective-correct ray-splat intersection; depth distortion và normal consistency (§1, §4–5). |
| 5. Methodology | Optimize disks từ sparse SfM và posed RGB; regularize geometry rồi fuse depths (§4–6). |
| 6. I/O | Input posed RGB + sparse points; output radiance field, depth/normal, triangle mesh (§4–6). |
| 7. Architecture | Tangent-plane parameterization → ray-plane intersection → tile rasterization → ADC (§4 pp.4–5). |
| 8. Mathematics | Eq.(4–6): local plane/Gaussian; Eq.(10): intersection; Eq.(13–16): geometry losses (pp.4–5). |
| 9. Representation | Mean, quaternion/tangent basis, **2** positive scales, opacity, SH; không phải covariance 3D invertible (§4.1). |
| 10. Optimization | Photometric + distortion + normal loss; custom CUDA gradients (§5–6.1). |
| 11. Losses | Distortion Σᵢⱼwᵢwⱼ\|zᵢ−zⱼ\|; normal Σᵢwᵢ(1−nᵢᵀN); geometry weights 1000 bounded/100 unbounded và 0.05 (§5). |
| 12. Density | ADC từ 3DGS, projected 3D-center gradient approximation, threshold 0.0002, opacity pruning 0.05 mỗi 3000 steps (§6.1 p.5). |
| 13. Training | 15k/30k comparisons; DTU 800×600 từ 1600×1200; một RTX3090 (§6.1–6.2 pp.5–6). |
| 14. Extraction | Training-view depth → Open3D TSDF; voxel 0.004, truncation 0.02 trong paper (§6.1). Đơn vị không tự chuyển sang mét của Ignatius. |
| 15. Evaluation | DTU CD; TNT F1; Mip-NeRF360 PSNR/SSIM/LPIPS (§6). Table 3 có **training-view** PSNR, không gọi đó NVS test. |
| 16. Quantitative | Table 1 p.6: DTU mean CD 0.80 tại 30k, 10.9 min; Table 2: TNT Ignatius F1 0.51 trong bản local. PGSR trích baseline khác version, không hợp nhất. |
| 17. Compute | RTX3090, CUDA renderer (§6.1); peak training VRAM cho RTX3050: Not specified in paper. |
| 18. Limitations | Unobserved regions và các vật liệu khó vẫn gây lỗi; sorting/intersection approximation còn hạn chế (§7/limitations p.8). |
| 19. Code | Official hbb1 repository đã kiểm tra liên kết; chưa build local. |
| 20. Application | B1 riêng; dùng cùng dataset contract, giữ TSDF native và common extraction track tách biệt. Không load PGSR checkpoint vào 2DGS. |

## P02

**3D Gaussian Splatting for Real-Time Radiance Field Rendering** — Kerbl, Kopanas, Leimkühler, Drettakis; TOG 42(4), 2023. Nguồn chính: §§3–7 pp.4–10, algorithm/derivatives và per-scene tables pp.11–14.

| # / Trường | Phân tích có nguồn |
| --- | --- |
| 1. Bibliography | SIGGRAPH/TOG 2023, DOI 10.1145/3592433 (p.1). |
| 2. Problem | NVS chất lượng cao nhưng neural volume rendering chậm (§1). |
| 3. Motivation | Explicit unstructured primitives thích ứng với scene, fast CUDA rasterization (§3). |
| 4. Contributions | Anisotropic Gaussians; interleaved density control; visibility-aware differentiable rasterizer (§1, §3). |
| 5. Methodology | SfM points → Gaussian initialization → alternating optimization/ADC → NVS (§3–5). |
| 6. I/O | Calibrated static-scene RGB và sparse SfM; output Gaussian field và rendered RGB. Mesh không thuộc output native (§3). |
| 7. Architecture | Parameter store, project covariance, cull/sort tiles, alpha blend; custom backward (§4–6). |
| 8. Mathematics | Eq.(3–7): alpha compositing, Gaussian, projected covariance, Σ=RSSᵀRᵀ và photometric loss (pp.3–5). |
| 9. Representation | Mean[3], scale[3], quaternion[4], opacity[1], SH coefficients; sigmoid opacity/exponential scale (§4–5.1). |
| 10. Optimization | Gradient descent framework; Adam implementation, position LR decay; random training view sampling (§5, §7.1, algorithm p.11). |
| 11. Losses | (1−λ)L1+λLD-SSIM, λ=0.2 (§5.1 Eq.7). Không thêm depth loss vào B0 rồi gọi original. |
| 12. Density | Clone small high-gradient Gaussians, split large ones, prune low opacity/oversized splats; mỗi 100 iterations sau warmup (§5.2). |
| 13. Training | 7k/30k evaluation; initialization scale từ 3 nearest points; SH activated progressively (§5.1, §7.1). |
| 14. Extraction | **Not applicable: original method không định nghĩa mesh extraction**. B0+depth→TSDF của dự án là diagnostic extension, không là original paper feature. |
| 15. Evaluation | Mip-NeRF360, TNT Truck/Train, Deep Blending; held-out RGB PSNR/SSIM/LPIPS, FPS, storage, time (§7.1–7.2). |
| 16. Quantitative | Table 1 p.8: Mip-NeRF360 30k PSNR 27.21, SSIM 0.815, LPIPS 0.214; training 41m33s. Không phải Ignatius. |
| 17. Compute | A6000 cho reported experiments (§7.1 p.8); “Mem” trong Table 1 là model storage, không là peak training VRAM. |
| 18. Limitations | Elongated/splotchy splats, popping, thiếu view; renderer sorting approximation (§7.4 p.10). |
| 19. Code | graphdeco-inria reference repository; main đã có thay đổi sau 2023, cần pin faithful revision. |
| 20. Application | B0 xác lập rendering và tài nguyên; geometry/mesh phải gắn nhãn extension riêng để giữ tính trung thực baseline. |

## P03

**3DGSR: Implicit Surface Reconstruction with 3D Gaussian Splatting** — Xiaoyang Lyu, Yang-Tian Sun, Yi-Hua Huang, Xiuzhe Wu, Ziyi Yang, Yilun Chen, Jiangmiao Pang, Xiaojuan Qi; TOG 43(6), Article 198, December 2024 (pp.1,4).

| # / Trường | Phân tích có nguồn |
| --- | --- |
| 1. Bibliography | DOI 10.1145/3687952; local revision metadata 2025 không thay năm journal. |
| 2. Problem | Radiance Gaussians không bảo đảm surface geometry (§1–2). |
| 3. Motivation | SDF liên tục bổ sung geometry trong free space, GS giữ chi tiết appearance (§4 pp.4–5). |
| 4. Contributions | Tight/loose GS–SDF coupling; volumetric depth/normal consistency (§4.1–4.2). |
| 5. Methodology | Hash-grid+MLP SDF và 3DGS cùng optimize, rendered geometry supervision (§4). |
| 6. I/O | Posed RGB; real scenes dùng SfM points/observations; output SDF mesh và GS radiance field (§4.3, p.6). |
| 7. Architecture | SDF branch + GS branch; sparse surface alignment và dense ray regularization (Fig.4 p.4). |
| 8. Mathematics | SDF zero level Eq.(6), SDF→opacity Eq.(7), orientation Eq.(9), geometry Eq.(11–15). |
| 9. Representation | 3DGS cùng neural SDF; loose coupling default để giảm conflict appearance/geometry (§4.1 p.5). |
| 10. Optimization | Joint differentiable optimization; dense ray samples regularize vùng không có Gaussian (§4.2–4.3). |
| 11. Losses | Photometric; rendered depth/normal; alignment; Eikonal. Weights 1.0/0.1/0.0001/0.1 (§4.3). **Sparse SfM z-depth L2 đã có** (p.6). |
| 12. Density | Kế thừa adaptive density control 3DGS (§4.3 p.5). Chi tiết mọi ngưỡng: Not specified in paper tại mục này. |
| 13. Training | 30k iterations; 1024 SDF rays/step; DTU 800×600, TNT 1920×1080 (§4.3–5.1). |
| 14. Extraction | Marching Cubes trên zero level SDF; Poisson là ablation so sánh, không tuyến chính (Fig.4, §5.3). |
| 15. Evaluation | NeRF synthetic, DTU, TNT; PSNR, CD, F1 (§5.1). |
| 16. Quantitative | Table 4 p.8: DTU training 30 min, inference 175 FPS; Table 1 p.7: synthetic loose-coupling PSNR 33.86. |
| 17. Compute | Single RTX3090, custom CUDA, tiny-cuda-nn (§4.3 pp.5–6). Không suy ra chạy được trên 6GB. |
| 18. Limitations | Unbounded scenes/hash-grid, noisy real poses, transparency (§6 p.9). Eq.(10) surface projection sign phải đối chiếu convention, không copy máy móc. |
| 19. Code | URL được tác giả công bố hiện trả 404; không tuyên bố đã audit official implementation. |
| 20. Application | **Đối chứng tính mới bắt buộc**: sparse SfM anchors không phải đóng góp mới. Joint SDF branch là nghiên cứu độc lập, chưa chọn vì chi phí/build. |

## P04

**A survey on surface reconstruction based on 3D Gaussian splatting** — Zheng Xu, Gang Chen, Feng Li, Lingyu Chen, Yuanhang Cheng; PeerJ Computer Science 11:e3034, 2025; DOI 10.7717/peerj-cs.3034 (p.1).

| # / Trường | Phân tích có nguồn |
| --- | --- |
| 1. Bibliography | Published 05/08/2025, 29 trang, tác giả và DOI ở p.1. |
| 2. Problem | Khó theo dõi nhanh các nhánh Gaussian surface reconstruction (§Introduction pp.1–3). |
| 3. Motivation | Taxonomy chung để phân biệt scene representation, optimization và extraction (pp.3–4). |
| 4. Contributions | Tổng hợp ba stage, datasets, applications, challenges; không là algorithm training mới (Fig.1, pp.3–4). |
| 5. Methodology | Literature search CVF, arXiv, Web of Science; taxonomy theo intermediate representations (Survey methodology p.5). |
| 6. I/O | Input literature; output taxonomy/review, không scene checkpoint (pp.3–5). |
| 7. Architecture | Static/dynamic representation, Gaussian optimization, static/deformable extraction (pp.6–14). |
| 8. Mathematics | Background Gaussian/covariance/blending Eq.(1–3) pp.4–5; phải dẫn original paper khi thực thi. |
| 9. Representation | Phân biệt points, volumes, Gaussians, hybrids (pp.5–8). |
| 10. Optimization | Tổng hợp geometric regularization và priors (pp.8–11); không optimizer duy nhất. |
| 11. Losses | SDF alignment, depth/normal và multi-view constraints được phân loại; không coi cùng một objective (pp.9–11). |
| 12. Density | Không đề xuất densification/pruning riêng; Not applicable cho survey (pp.8–11). |
| 13. Training | Not applicable; không có cấu hình training thống nhất để tái hiện toàn survey. |
| 14. Extraction | Poisson, TSDF, implicit zero-level, tetrahedral opacity surfaces (pp.12–14). |
| 15. Evaluation | Dataset overview DTU, TNT, indoor/outdoor và synthetic (pp.14–17). |
| 16. Quantitative | Dataset summary ở Table 2 p.14; không có benchmark chạy trong repo này. Không chuyển số tổng hợp thành measured result. |
| 17. Compute | Not applicable cho survey; hardware training chung: Not specified in paper. |
| 18. Limitations | High-resolution geometry, scalability, illumination/material và incomplete observation (Challenges pp.20–21). Một số attribution trong prose cần kiểm tra primary source. |
| 19. Code | DOI/publisher được xác minh; không có single official trainer cho survey. |
| 20. Application | Dùng làm taxonomy/gap checklist; không lấy citation nhầm “GOF (Gropp et al. 2020)” ở p.9 thay primary GOF của Yu et al. |

## P05

**DN-Splatter: Depth and Normal Priors for Gaussian Splatting and Meshing** — Turkulainen, Ren, Melekhov, Seiskari, Rahtu, Kannala. Local preprint 2024; [WACV 2025 publication](https://ieeexplore.ieee.org/abstract/document/10943388). Nguồn kỹ thuật: §4–5 pp.3–8, appendix pp.12–20.

| # / Trường | Phân tích có nguồn |
| --- | --- |
| 1. Bibliography | Tác giả p.1; năm filename là preprint, publication WACV 2025. |
| 2. Problem | Indoor ít texture khiến photometric-only GS geometry kém (§1). |
| 3. Motivation | Sensor/monocular cues hỗ trợ ill-posed regions (§1, §4). |
| 4. Contributions | Gradient-aware depth loss, Gaussian normal supervision/smoothing và practical meshing (§4). |
| 5. Methodology | Align priors vào SfM scale → supervise depth/normal → backproject oriented points (§4.1–4.5). |
| 6. I/O | RGB/cameras + sensor hoặc monocular depth/normals; output GS và Poisson mesh (§4). |
| 7. Architecture | gsplat renderer output RGB/depth/normal, frozen prior inference, regularized optimizer (§3–4). |
| 8. Mathematics | Eq.(4): edge-weighted log-L1 depth; Eq.(5): affine mono-depth/SfM fit; Eq.(6–12): normals/smoothing/final loss (pp.4–5). |
| 9. Representation | 3DGS flattened via smallest scale; shortest axis defines normal (§4.2). |
| 10. Optimization | ADC-compatible GS training; priors offline/frozen; normal geometry gradients update scale/rotation (§4.2–4.4). |
| 11. Losses | Lrgb + 0.2 Ldepth + Lscale + 0.1 Lnormal + 0.1 Lsmooth; log residual weighted by exp(−∇I) (§4.1–4.4). |
| 12. Density | Kế thừa ADC; bản baseline Splatfacto default cần version rõ (§3, App.A.1). |
| 13. Training | 30k, gsplat 1.0.0; thường initialize 1M sensor points; Omnidata normals (§5 p.5). Không lấy sensor protocol áp cho monocular video. |
| 14. Extraction | Depth/normal backprojection → oriented point set → Poisson; mặc định 2M points, octree depth 9; appendix có biến thể 5M (§4.5, §5, App.A/C). |
| 15. Evaluation | MuSHRoom 6 scenes, ScanNet++ 2 scenes; RGB/depth/mesh metrics, F1 threshold 5cm (§5, App.B). |
| 16. Quantitative | Table 1 p.6, sensor-depth variant: MuSHRoom F1 0.9243; C-L1 0.0216. Không so trực tiếp TNT F1 threshold khác. |
| 17. Compute | Reported training trên RTX4090 (Table 2 p.6); 1M initialization có thể vượt budget 6GB, chưa benchmark ở đây. |
| 18. Limitations | Chất lượng phụ thuộc prior/sensor, normal convention và loss weighting; indoor evidence không tự mở rộng thành outdoor guarantee (§5–6). |
| 19. Code | maturk/dn-splatter; repo hiện gồm cả AGS-Mesh, phải pin đúng phương pháp. |
| 20. Application | Edge-aware geometric regularization và SfM alignment là existing methods. Chưa thêm monocular model vào Ours vì cần isolate contribution và VRAM. |

## P06

**SurfaceSplat: Connecting Surface Reconstruction and Gaussian Splatting** — Zihui Gao, Jia-Wang Bian, Guosheng Lin, Hao Chen, Chunhua Shen; ICCV 2025. Nguồn: §3 pp.3–5; §4 pp.5–8.

| # / Trường | Phân tích có nguồn |
| --- | --- |
| 1. Bibliography | Tác giả/title p.1; CVF proceedings PDF 2025, printed pp.28525–28534. |
| 2. Problem | Sparse views: SDF giữ global shape, GS giữ appearance nhưng thiếu dense geometry (§1). |
| 3. Motivation | Hai representation có thế mạnh bổ sung nhau (§1–3). |
| 4. Contributions | Mesh-initialized GS; GS synthetic view expansion để refine SDF; cyclic optimization (§3). |
| 5. Methodology | Voxurf coarse mesh → clean/sample → GS → additional rendered views → Voxurf fine stage (§3.2–3.4). |
| 6. I/O | Sparse RGB/cameras và monocular normal prior; mesh và GS output (Fig.2 p.3). |
| 7. Architecture | SDF voxel grid+MLPs và separate GS stage; **không PGSR loss plug-in** (§3). |
| 8. Mathematics | Eq.(3–4): SDF objectives/normal prior; Eq.(6): backprojection; Eq.(7–12): expanded views/cycle (pp.4–5). |
| 9. Representation | Voxurf SDF grids và full 3DGS (§3.1). |
| 10. Optimization | Sequential training stages, thường một cycle; synthetic views bổ sung train, không trở thành held-out GT (§3.4). |
| 11. Losses | Photometric+SDF TV/smoothness; normal L1 từ Metric3D, λnormal=0.05 (§3.2, §4.1 pp.6–7). |
| 12. Density | GS initialization từ surface-visible depth samples cộng sparse points cho background; chi tiết ADC riêng: Not specified in paper (§3.2). |
| 13. Training | Coarse grid 96³/10k; GS 7k; fine grid 256³/20k; khoảng một giờ/cycle (§4.1 pp.6–7). |
| 14. Extraction | Marching Cubes trên SDF; largest component cleaning trong paper phục vụ object-centric coarse stage (§3.2). |
| 15. Evaluation | MobileBrick 10 views, DTU 5 views; CD/F1 và foreground/full-image PSNR (§4.1). |
| 16. Quantitative | Table 2 p.6: 5-view DTU mean CD 1.45mm; Table 1: MobileBrick CD 9.7mm. Khác dense-view PGSR protocol. |
| 17. Compute | Multi-stage grids/prior models có overhead; peak training VRAM: Not specified in paper. Không hứa 6GB. |
| 18. Limitations | GS rendering có thể đưa self-reinforcing errors vào SDF; thiếu true observation vẫn là constraint. Đánh giá này là inference từ cycle (§3.4), không claim tác giả chứng minh. |
| 19. Code | aim-uofa/SurfaceSplat đã xác minh official link. |
| 20. Application | Background cho sparse-view direction; không đưa synthetic view expansion hoặc largest-component-only vào main Ignatius pipeline. |

## P07

**Gaussian Opacity Fields: Efficient Adaptive Surface Reconstruction in Unbounded Scenes** — Zehao Yu, Torsten Sattler, Andreas Geiger; SIGGRAPH Asia/TOG 2024. Local arXiv:2404.10772v2 (p.1).

| # / Trường | Phân tích có nguồn |
| --- | --- |
| 1. Bibliography | 2024, title/authors/project ở p.1; không nhầm opacity field với 2DGS disk. |
| 2. Problem | Explicit Gaussians không có continuous surface; fixed voxel extraction khó scale (§1). |
| 3. Motivation | Tạo implicit opacity trực tiếp từ cùng Gaussian field để extraction thích ứng scene (§3). |
| 4. Contributions | Ray-Gaussian evaluation; opacity field; normal definition; adaptive tetrahedral extraction (§3.1–3.4). |
| 5. Methodology | RGB+SfM → regularized ray-based GS → opacity level set extraction (§3). |
| 6. I/O | Posed images/sparse points → GS + foreground/background mesh (§3–4). |
| 7. Architecture | Custom ray-based CUDA renderer và opacity evaluator; Delaunay backend; marching tetrahedra (§3–4). |
| 8. Mathematics | Per-ray opacity Eq.(9), minimum over views Eq.(10); distortion/normal Eq.(11–13); §3.4 level-set search. |
| 9. Representation | Anisotropic 3D Gaussians, view-conditioned ray contribution; opacity field không phải SDF (§3.2). |
| 10. Optimization | Geometry regularized GS; improved pixel-gradient densification (§3.3). |
| 11. Losses | RGB + depth distortion + normal consistency; **detach blend weights trong distortion** để tránh floaters (§3.3 p.4). |
| 12. Density | Improved gradient aggregation; opacity prune 0.05 thay 0.005 để kiểm soát count; stop densification 15k (§3.3, §4.1). |
| 13. Training | 30k; distortion 1000 bounded/100 unbounded, normal 0.05; appearance model riêng cho TNT (§4.1 p.6). |
| 14. Extraction | Gaussian center+8 box corners → CGAL Delaunay → opacity 0.5 level-set → marching tetrahedra với binary search; filter long invalid cells (§3.4 pp.5–6). |
| 15. Evaluation | DTU, TNT surface; Mip-NeRF360 NVS; không có background GT nên foreground-only geometry metric (§4). |
| 16. Quantitative | Table 3 p.7: Mip-NeRF360 outdoor PSNR 24.82, indoor 30.79; không so trực tiếp mean đã gộp của paper khác. |
| 17. Compute | NVIDIA A100 (§4.1); Delaunay CPU CGAL có bottleneck, bicycle khoảng 8 min (§5 p.10). Đây không là all-GPU pipeline. |
| 18. Limitations | Triangulation/memory/mesh size và occlusion/material ambiguities (§5). Marching Tetrahedra ở đây khác MILo train-in-loop. |
| 19. Code | autonomousvision/gaussian-opacity-fields official; chưa compile Windows. |
| 20. Application | B3 optional separate pipeline. Đổi extraction trên B0 không biến B0 thành full GOF; cần khai báo renderer, loss và adapter compatibility. |

## P08

**Gaussian splatting for surface reconstruction: A survey** — Vitor Matias, Daniel Perazzo, Vinicius da Silva, Alberto Raposo, Luiz Velho, Afonso Paiva, Tiago Novello; Computers & Graphics 141 (2026) 104782, DOI 10.1016/j.cag.2026.104782. Available online 14/10/2026 (p.1).

| # / Trường | Phân tích có nguồn |
| --- | --- |
| 1. Bibliography | Publication 2026; “SIBGRAPI 2025 Tutorials” là special-section label, không phải năm bài hiện tại (p.1). |
| 2. Problem | Photometric fidelity không đảm bảo surface recovery (§1). |
| 3. Motivation | Kết nối rendering formulation, surface regularization, extraction và fair comparisons (§1–2). |
| 4. Contributions | Taxonomy, mathematical review và comparative tables; không trainer mới (§3–6). |
| 5. Methodology | Review explicit/hybrid/implicit representations và extraction families; comparison theo protocols (pp.4–12). |
| 6. I/O | Literature và comparison evidence → survey/taxonomy; không Gaussian checkpoint. |
| 7. Architecture | Depth-map fusion; SDF coupling; mesh-in-loop; opacity fields (Table 2 p.12). |
| 8. Mathematics | Volume-rendered depth/normals, Gaussian regularization, opacity fields (§2–4 pp.2–9). Dẫn original khi chọn formulation. |
| 9. Representation | 2DG/3DG, quadrics, fields, tetrahedra và hybrids (Table 2). |
| 10. Optimization | Phân biệt post-hoc extraction và joint mesh/field optimization (§4). |
| 11. Losses | Photometric, distortion, normal, geometric priors và coupled-field terms (§2–4). Không cộng tất cả vào một “survey method”. |
| 12. Density | Không đề xuất ADC duy nhất; Not applicable cho survey. |
| 13. Training | Comparison dùng nhiều codebase/hardware; không một config chung cho mọi bảng (§6 pp.10–12). |
| 14. Extraction | TSDF, Poisson, Marching Cubes, opacity-based tetrahedra và mesh-in-loop (§4.3 pp.8–9). |
| 15. Evaluation | DTU/TNT, foreground protocol, CD/F-score; rendering evidence khác geometry (§5–6). |
| 16. Quantitative | Table 3 p.12 có PGSR 0.52 CD/30.5 min/RTX4080; **không thay** 0.49 của local PGSR paper bằng số này. |
| 17. Compute | Table 3 chỉ rõ nhiều GPU khác nhau; không chuyển runtime sang RTX3050 bằng tỷ lệ FLOPs. |
| 18. Limitations | Bảng tổng hợp/code revisions/protocol differences; taxonomy có thể rút gọn sai meshing của method cụ thể. DN-Splatter phải theo primary Poisson. |
| 19. Code | Publisher DOI xác minh; không single official training model. |
| 20. Application | Kiểm tra ranh giới representation/extractor và cách báo cáo provenance; không dùng survey thay mọi primary source. |

## P09

**GVGS: Gaussian Visibility-Aware Multi-View Geometry for Accurate Surface Reconstruction** — Mai Su, Qihan Yu, Zhongtao Wang, Yilong Li, Chengwei Pan, Yisong Chen, Guoping Wang, Fei Zhu; local arXiv:2601.20331v3, 02/04/2026 (p.1). Không tự gán venue chưa xác minh.

| # / Trường | Phân tích có nguồn |
| --- | --- |
| 1. Bibliography | Author list có Fei Zhu ở bản local v3; index abstract cũ không đủ để thay metadata (p.1). |
| 2. Problem | Depth-based visibility phụ thuộc depth chính xác, trong khi depth supervision lại phụ thuộc visibility (§1). |
| 3. Motivation | Dùng Gaussian contribution để mở rộng co-visible supervision (§3.2). |
| 4. Contributions | GVMV geometry consistency và quadtree-calibrated monocular depth QDC (§3). |
| 5. Methodology | Accumulate per-Gaussian contributions ở neighbor; project visibility weights về reference; calibrate depth blockwise (§3.2–3.3). |
| 6. I/O | Calibrated RGB, SfM init, Depth Anything V2 priors → GS, depths và TSDF mesh (§3–4). |
| 7. Architecture | PGSR base + visibility-aware renderer + progressive depth calibration (Fig.2 p.4). |
| 8. Mathematics | Eq.(5–8): contribution/opacity/weighted roundtrip residual; Eq.(9–10): median/MAD affine depth fit và loss (pp.4–5). |
| 9. Representation | PGSR-style planar 3DGS; contribution statistics gắn Gaussian IDs, không sparse COLMAP visibility (§3). |
| 10. Optimization | End-to-end joint losses, cached calibration updates khi đổi quadtree level (§3.3). |
| 11. Losses | Existing RGB/scale/single-view/multi-view terms + GVMV/QDC; Eq.(8) dùng exp(−φ)+λOᵣ (§3.4). |
| 12. Density | Retain PGSR hyperparameters; independent novel density policy: Not specified in paper (§4.2). |
| 13. Training | DTU 30k, QDC 7k–25k, split every 5k; TNT 60k; τ=1e−4, visibility λ=0.5 (§4.2 p.6). |
| 14. Extraction | TSDF fusion (§4.2). |
| 15. Evaluation | DTU standard 15 scans, TNT 6 scenes; DTU no explicit GT alignment; TNT QGS protocol (§4.1). |
| 16. Quantitative | §4.3 p.6, Tables 1–2 p.8: mean DTU CD 0.49mm, TNT F1 0.53; TNT 60k differs 30k baseline budget. |
| 17. Compute | Experiments dùng **4 RTX A6000** (§4.2); chưa xác định mỗi run dùng đồng thời 4 card hay phân phối runs, không suy đoán. |
| 18. Limitations | Specular/transparent surfaces (§limitations p.9); Gaussian visibility vẫn là model estimate, không là GT occlusion. |
| 19. Code | URL trong PDF redirect về sumai-1998/GVGS; chưa chạy hoặc đối chiếu toàn source. |
| 20. Application | Visibility-weighted geometry đã tồn tại; Ours phải phân biệt fixed SfM observation support và learned Gaussian visibility, so sánh theo cùng iterations. |

## P10

**MeshSplat: Generalizable Sparse-View Surface Reconstruction via Gaussian Splatting** — Hanzhi Chang, Ruijie Zhu, Wenjie Chang, Mulin Yu, Yanzhe Liang, Jiahao Lu, Zhuoyuan Li, Tianzhu Zhang; AAAI 2026, preprint 2025 (p.1; official project/code registry).

| # / Trường | Phân tích có nguồn |
| --- | --- |
| 1. Bibliography | Local proceedings printed pages 2716 onward; phân biệt MeshSplat với MeshSplatting hoặc surface per-scene methods. |
| 2. Problem | Surface reconstruction từ cực ít views, cần cross-scene generalization (§1–2). |
| 3. Motivation | 2DGS bridge từ NVS supervision tới geometric priors (§1). |
| 4. Contributions | Pixel-aligned 2DGS prediction; weighted Chamfer depth regularization; uncertainty-guided normal prediction (§3). |
| 5. Methodology | Two posed RGB → CNN/Transformer → plane-sweep cost volume → Gaussian heads (§3.1–3.3). |
| 6. I/O | Hai RGB + projection matrices; output 2HW Gaussians, depth/normal, NVS, mesh (§3.1 Eq.1). |
| 7. Architecture | Cost-volume refinement U-Net, normal prediction network, Gaussian head (Fig.3 p.4). |
| 8. Mathematics | Softmax depth expectation Eq.(3); WCD Eq.(5–7); angular NLL Eq.(8); final objective Eq.(9–10). |
| 9. Representation | Pixel-aligned **2D** GS, rotation from normal head, uncertainty κ; khác free per-scene parameter store (§3.3). |
| 10. Optimization | Feed-forward network training nhiều scenes, inference generalization (§3.6–4.1). |
| 11. Losses | MSE+LPIPS photometric, confidence-weighted Chamfer, angular NLL với Omnidata pseudo normals (§3.4–3.6). |
| 12. Density | Gaussian count từ pixels; per-scene clone/split ADC không phải kiến trúc mô tả. |
| 13. Training | Re10K 256², 200k steps, batch12; ScanNet 512×384, 75k, batch4 (§4.1 p.5). |
| 14. Extraction | Inference GS → 2DGS-style TSDF fusion; baselines cũng dùng TSDF cho comparison (§3.6, §4.2). |
| 15. Evaluation | Re10K/ScanNet training; zero-shot ScanNet/Replica; CD/F1, depth và normal metrics (§4). |
| 16. Quantitative | Table 4 p.7 ScanNet ablation: CD 0.3748 baseline → 0.2606 full; không tự gọi đơn vị mm khi bảng không ghi. |
| 17. Compute | Exact peak VRAM/GPU model cho cấu hình này: Not specified in paper. Training hàng chục nghìn scenes không là smoke phù hợp 6GB. |
| 18. Limitations | Textureless depth discontinuity, không reconstruct vùng không quan sát (§5 p.7). |
| 19. Code | HanzhiChang/MeshSplat từ project-page Code link. |
| 20. Application | Background cho sparse-view generalization; không đưa vào main baseline per-scene budget vì dữ liệu/pretraining khác. |

## P11

**MILo: Mesh-In-the-Loop Gaussian Splatting for Detailed and Efficient Surface Reconstruction** — Antoine Guédon, Diego Gomez, Nissim Maruani, Bingchen Gong, George Drettakis, Maks Ovsjanikov; TOG 44(6), SIGGRAPH Asia 2025; DOI 10.1145/3763339 (p.1).

| # / Trường | Phân tích có nguồn |
| --- | --- |
| 1. Bibliography | December 2025; PDF typesetting có provisional Article 1, không ép thành citation article number đó. |
| 2. Problem | Post-hoc volume→surface conversion mất detail và tạo mesh rất dày (§1). |
| 3. Motivation | Geometry loss trên mesh cần backpropagate về Gaussian trong training (§3). |
| 4. Contributions | Gaussian pivots, differentiable extraction, bidirectional consistency, anti-erosion/interior regularization (§3–5). |
| 5. Methodology | Selected Gaussians spawn tetra vertices → signed scalars → differentiable marching tetrahedra → mesh render losses (§4–5). |
| 6. I/O | Posed RGB/SfM → GS và lightweight surface mesh (§3). |
| 7. Architecture | RaDe-GS base, Delaunay update, GPU MT, differentiable mesh rendering (§3, §6). |
| 8. Mathematics | Mesh vertex interpolation Eq.(2); depth/normal consistency Eq.(5–7); erosion/interior Eq.(8–11) (pp.5–7). |
| 9. Representation | Gaussian pivots với **9 learnable signed scalars** mỗi pivot, không nhất thiết true metric SDF (§4.2 p.5). |
| 10. Optimization | Gradients từ mesh vertices qua signed scalars và pivot positions/covariance; topology updates rời rạc (§4.3). |
| 11. Losses | Volumetric photometric/normal + mesh depth log-L1/normal + anti-erosion + interior BCE (§5). |
| 12. Density | Aggressive early densification; importance sampling chọn pivots; base prune/dense retain hai chế độ (§4.1, App.B). |
| 13. Training | App.B pp.13–14: densify 3k, introduce mesh 8k, tổng 18k; refresh triangulation 500 iterations. |
| 14. Extraction | In-loop differentiable Marching Tetrahedra; không TSDF và không GOF opacity0.5 thuần túy (§4.2–4.3). |
| 15. Evaluation | TNT F1, DTU CD, Mip-NeRF360/DeepBlending mesh-based NVS và complexity (§6). |
| 16. Quantitative | Table 2 p.6: mean TNT F1 base RaDe-GS 0.47, base GOF 0.49; biến thể dense 0.49. |
| 17. Compute | §6.1 p.9: base **10GB training VRAM**, dense tới17GB, RTX4090 24GB. Số của paper, không phép đo local. |
| 18. Limitations | In-loop extraction overhead, phụ thuộc initial Gaussian distribution (§6.7 p.11). Không cam kết watertight mọi input. |
| 19. Code | Anttwo/MILo official; chưa build/local run. |
| 20. Application | Research branch độc lập, không đưa vào mandatory 6GB roadmap. Có thể tham khảo mesh complexity và field compatibility. |

## P12

**MASt3R-SLAM: Real-Time Dense SLAM with 3D Reconstruction Priors** — Riku Murai, Eric Dexheimer, Andrew J. Davison; CVPR 2025 (p.1).

| # / Trường | Phân tích có nguồn |
| --- | --- |
| 1. Bibliography | CVF CVPR2025 paper; local 11 trang. |
| 2. Problem | Monocular dense SLAM ổn định với camera model không biết trước (§1). |
| 3. Motivation | Two-view learned geometry tốt nhưng inconsistent scale và drift (§2–3). |
| 4. Contributions | Efficient matching, tracking/fusion, loop closure, second-order global optimization (§3). |
| 5. Methodology | MASt3R pair predictions → pointmap matching → Sim(3) tracking → graph optimization (§3.1–3.5). |
| 6. I/O | Monocular image stream, calibration optional; output poses và dense pointmaps (§1, §3). |
| 7. Architecture | Frontend tracker/fusion; retrieval/loop/backend; learned pairwise priors (Fig.3 p.4). |
| 8. Mathematics | Sim(3) update Eq.(1); ray alignment Eq.(3,6); Gauss-Newton Eq.(7); weighted fusion Eq.(8). |
| 9. Gaussian representation | Not applicable: method không optimize Gaussian scene. |
| 10. Optimization | Robust IRLS/Gauss-Newton; pose graph loop constraints và filtering canonical pointmaps (§3). |
| 11. Losses | Geometric/ray residuals và robust weights; không GS photometric trainer (§3.3–3.5). |
| 12. Density | Not applicable: không Gaussian densification/pruning. |
| 13. Training config | Dùng pretrained MASt3R; không huấn luyện GS từ Ignatius. Largest side512 ở evaluation (§4 p.6). |
| 14. Surface extraction | Không đề xuất native triangle mesh extractor trong main pipeline; dense pointmaps không mặc định mesh. |
| 15. Evaluation | TUM RGB-D, 7-Scenes, EuRoC; ATE, reconstruction accuracy/completion/CD và speed (§4). |
| 16. Quantitative | Table 4 p.8: matching+feature variant 2ms, system14.9 FPS; đây là tác giả đo, không đồng nghĩa full network inference2ms. |
| 17. Compute | Intel i9-12900K + RTX4090 (§4 p.6). |
| 18. Limitations | Global optimization chưa refine toàn geometry; learned biases/drift, model assumptions (§5 p.8). |
| 19. Code | rmurai0610/MASt3R-SLAM official. |
| 20. Application | Background về scale/visibility; không thay COLMAP, không thêm learned pose branch vào task hiện tại. |

## P13

**ORB-SLAM3: An Accurate Open-Source Library for Visual, Visual-Inertial and Multi-Map SLAM** — Carlos Campos, Richard Elvira, Juan J. Gómez Rodríguez, José M. M. Montiel, Juan D. Tardós; IEEE Transactions on Robotics 2021, DOI 10.1109/TRO.2021.3075644. PDF p.1 là copyright cover, title/abstract ở p.2.

| # / Trường | Phân tích có nguồn |
| --- | --- |
| 1. Bibliography | Journal 37(6), 2021; local arXiv v2 cover 23/04/2021. |
| 2. Problem | Robust visual/visual-inertial SLAM và multimap recovery (§I–II). |
| 3. Motivation | Long-term associations và reuse maps khi tracking mất (§I–III). |
| 4. Contributions | MAP inertial init, Atlas, improved place recognition, multimap merging (§III–VI). |
| 5. Methodology | ORB features → tracking/local mapping → loop closing/Atlas merge → BA (§III–VI). |
| 6. I/O | Mono/stereo/RGB-D, optional IMU; output trajectories/sparse maps (§III). Không có IMU trong monocular video hiện tại. |
| 7. Architecture | Parallel tracking, mapping, loop/merge threads và Atlas (Fig.1, §III). |
| 8. Mathematics | Visual reprojection Eq.(3), MAP Eq.(4), positive-scale update Eq.(10) (pp.7–8). |
| 9. Gaussian representation | Not applicable. |
| 10. Optimization | Robust local/global BA, inertial MAP; loops/merge optimize graph (§V–VI). |
| 11. Losses | Mahalanobis visual/inertial residuals, Huber on visual correspondences (§V p.7). |
| 12. Density | Keyframe/map-point insertion/culling; không Gaussian ADC (§III). |
| 13. Training config | Not applicable: conventional SLAM, không neural training. Vocabulary và calibration là dependencies khác. |
| 14. Surface extraction | Not applicable: sparse SLAM map không tự thành dense triangle mesh. |
| 15. Evaluation | EuRoC, TUM-VI; ATE, scale drift, repeated runs/runtime (§VII). |
| 16. Quantitative | §VII.D p.14: tracking30–40 frames/s, mapping3–6 keyframes/s; authors report medians over10 runs (§VII p.11). |
| 17. Compute | Intel i7-7700 3.6GHz/32GB, **only CPU** (§VII p.11). Không là evidence CUDA SfM. |
| 18. Limitations | Inertial observability/initialization có thể khó khi chuyển động yếu (§V.B p.8); không assume absolute scale từ mono-only. |
| 19. Code | UZ-SLAMLab/ORB_SLAM3 official. |
| 20. Application | Tài liệu nền về tracks/gauge/camera models; không chuyển pipeline sang SLAM. |

## P14

**PGSR: Planar-based Gaussian Splatting for Efficient and High-Fidelity Surface Reconstruction** — Danpeng Chen, Hai Li, Weicai Ye, Yifan Wang, Weijian Xie, Shangjin Zhai, Nan Wang, Haomin Liu, Hujun Bao, Guofeng Zhang; TVCG2024 theo official repository. Local PDF17 trang; cite nội dung theo bản local, không ép thông số bản code vào paper.

| # / Trường | Phân tích có nguồn |
| --- | --- |
| 1. Bibliography | Tác giả/title p.1; publication label TVCG2024 tại official registry; local manuscript không ghi đủ volume/pages, không tự bổ sung. |
| 2. Problem | GS optimized RGB không bảo đảm accurate/multi-view-consistent surface (§I). |
| 3. Motivation | Flattened primitive cần depth definition phù hợp hình học để geometric constraints có nghĩa (§IV.A). |
| 4. Contributions | Planar representation, unbiased plane depth, single-/multi-view regularization, exposure compensation (§IV). |
| 5. Methodology | SfM init → planar GS optimize → render plane geometry → fuse TSDF (§IV–V). |
| 6. I/O | Static posed RGB+sparse points; output GS, plane distance/normal/depth và mesh (§IV). |
| 7. Architecture | 3DGS base; min-scale flatten; distance/normal rasterization; normal-from-depth/neighbor warps; exposure model (Fig.4 p.4). |
| 8. Mathematics | Eq.(1–4): flatten/plane depth; (5–11): single/multi-view; (12–15): exposure/final training objective (pp.5–8). |
| 9. Representation | Flattened **3D Gaussians**, shortest-axis normal; không swap thành 2DGS renderer (§IV.A). |
| 10. Optimization | Geometry constraints sau initialization; AbsGS densification; exposure LR0.001 (§V p.8). |
| 11. Losses | Paper λflatten100; single0.01, multi-photo0.2, multi-geometry0.05; RGB mix0.2 (§IV.D). Code khác: xem [notes](pgsr_implementation_notes.md). |
| 12. Density | AbsGS cited trong implementation details (§V); code còn có trim/pruning behavior phải pin. |
| 13. Training | 30k; NCC11×11 trong paper; exposure chỉ TNT; RTX4090 (§IV.B–V pp.7–8). |
| 14. Extraction | Render training-view depths → TSDF → isosurface mesh (§V p.8). Không suy ra Gaussian PLY là triangle mesh. |
| 15. Evaluation | Mip-NeRF360 NVS; DTU15 scans CD(mm); TNT6 scenes F1 (§V). |
| 16. Quantitative | Table II p.9: DTU PGSR mean0.49mm, PGSR(DS)0.53mm; Table III Ignatius F1=0.80, mean0.50. **Chỉ paper results**. |
| 17. Compute | RTX4090, reported DTU1h/full, 0.6h DS; TNT1.2h (Tables II–III). Peak training VRAM: Not specified in paper. |
| 18. Limitations | Missing viewpoints, reflective/mirror surfaces, floaters (§VI p.10). Plane-depth “unbiased” không bảo đảm mixture nhiều surfaces/near-parallel rays chính xác. |
| 19. Code | Official zju3dv/PGSR; read-only source audit ở commit ghi trong [notes](pgsr_implementation_notes.md). |
| 20. Application | B2 và nền Ours vì có sẵn geometry signals; kiểm soát Windows build, memory, plane convention và existing regularizers trước cải tiến. |

## Kết luận literature

PGSR là lựa chọn kiến trúc đề xuất cho geometry branch, dựa trên compatibility với posed RGB/SfM và explicit regularization; **tính khả thi trên GPU6GB chưa được xác minh**. B0/B1 giữ vai trò đối chiếu representation, B3 GOF tách field/extraction. Hybrid SDF, MILo và feed-forward sparse-view methods không được coi là checkpoint-compatible extensions.

Ours chưa có experimentally validated result. Sparse SfM depth đã có ở P03 p.6, prior alignment/edge weighting ở P05 §4, visibility ở P09 §3. Nguồn bổ sung [CDGS §3](https://arxiv.org/html/2502.14684v1#S3) còn dùng reprojection-error weights khi align depth và confidence-dependent regularization. Vì vậy tên hướng ban đầu không đủ chứng minh novelty. Xem [research gaps](research_gaps.md) và [proposed method](proposed_method.md).
