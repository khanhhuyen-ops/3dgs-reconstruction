# Paper inventory — 14/11/2026

Quét đệ quy `paper/`: **14 PDF, 209 trang; 14/14 mở và trích xuất được văn bản bằng PyMuPDF hiện có**. Không cài dependency, không sửa PDF, không thêm paper tải ngoài vào `paper/`. PDF page là chỉ số vật lý bắt đầu từ 1, tránh nhầm số trang appendix hoặc watermark. Metadata lấy từ nội dung trang đầu; năm trong filename không được dùng thay năm xuất bản.

Đã đọc các phần phương pháp, thí nghiệm, giới hạn và supplemental liên quan để thiết kế; không có OCR thất bại. Các bảng/công thức sử dụng được đối chiếu đúng vị trí; Eq. (6) PGSR được xem ảnh trang 6 vì trích xuất dấu overline khó đọc. Đây là literature review có phạm vi; không phải xác minh mọi phép suy diễn của tác giả hoặc chạy tái hiện paper. Text trích xuất và ảnh kiểm tra chỉ ở thư mục tạm ngoài repo.

| ID | PDF gốc | Trang | Vai trò | Phiên bản/năm |
| --- | --- | ---: | --- | --- |
| [P01 — 2DGS](literature_review.md#p01) | [2D Gaussian Splatting (2024).pdf](../../paper/2D%20Gaussian%20Splatting%20%282024%29.pdf) | 13 | Required baseline B1 | 2024 SIGGRAPH; local arXiv v3 22/02/2025 |
| [P02 — 3DGS](literature_review.md#p02) | [3D Gaussian Splatting for Real-Time Radiance Field Rendering.pdf](../../paper/3D%20Gaussian%20Splatting%20for%20Real-Time%20Radiance%20Field%20Rendering.pdf) | 14 | Core foundation / B0 | 2023 TOG 42(4), doi:10.1145/3592433 |
| [P03 — 3DGSR](literature_review.md#p03) | [3DGSR Implicit Surface Reconstruction with 3D Gaussian Splatting.pdf](../../paper/3DGSR%20Implicit%20Surface%20Reconstruction%20with%203D%20Gaussian%20Splatting.pdf) | 12 | Optional improvement / novelty check | 2024 TOG 43(6), Article 198 |
| [P04 — Xu et al. survey](literature_review.md#p04) | [A survey on surface reconstruction based on 3dgs.pdf](../../paper/A%20survey%20on%20surface%20reconstruction%20based%20on%203dgs.pdf) | 29 | Background literature | 2025 PeerJ Computer Science 11:e3034 |
| [P05 — DN-Splatter](literature_review.md#p05) | [DN-Splatter (2024).pdf](../../paper/DN-Splatter%20%282024%29.pdf) | 20 | Geometry reference / novelty check | 2024 preprint; WACV 2025 |
| [P06 — SurfaceSplat](literature_review.md#p06) | [Gao_SurfaceSplat_Connecting_Surface_Reconstruction_and_Gaussian_Splatting_ICCV_2025_paper.pdf](../../paper/Gao_SurfaceSplat_Connecting_Surface_Reconstruction_and_Gaussian_Splatting_ICCV_2025_paper.pdf) | 10 | Optional independent research | 2025 ICCV |
| [P07 — GOF](literature_review.md#p07) | [Gaussian Opacity Fields (2024).pdf](../../paper/Gaussian%20Opacity%20Fields%20%282024%29.pdf) | 15 | Optional baseline B3 | 2024 SIGGRAPH Asia / TOG |
| [P08 — Matias et al. survey](literature_review.md#p08) | [Gaussian splatting for surface reconstruction A survey.pdf](../../paper/Gaussian%20splatting%20for%20surface%20reconstruction%20A%20survey.pdf) | 15 | Background / comparison protocols | 2026 Computers & Graphics 141:104782 |
| [P09 — GVGS](literature_review.md#p09) | [GVGS Gaussian Visibility-Aware Multi-View Geometry for Accurate Surface Reconstruction.pdf](../../paper/GVGS%20Gaussian%20Visibility-Aware%20Multi-View%20Geometry%20for%20Accurate%20Surface%20Reconstruction.pdf) | 11 | Geometry reference / novelty check | 2026 arXiv:2601.20331v3, 02/04/2026 |
| [P10 — MeshSplat](literature_review.md#p10) | [MeshSplat Generalizable Sparse-View Surface Reconstruction via Gaussian Splatting.pdf](../../paper/MeshSplat%20Generalizable%20Sparse-View%20Surface%20Reconstruction%20via%20Gaussian%20Splatting.pdf) | 9 | Sparse-view background | 2025 preprint; AAAI 2026 |
| [P11 — MILo](literature_review.md#p11) | [MILo Mesh-In-the-Loop Gaussian Splatting for Detailed and Efficient Surface Reconstruction.pdf](../../paper/MILo%20Mesh-In-the-Loop%20Gaussian%20Splatting%20for%20Detailed%20and%20Efficient%20Surface%20Reconstruction.pdf) | 15 | Advanced extraction / resource reference | 2025 SIGGRAPH Asia / TOG |
| [P12 — MASt3R-SLAM](literature_review.md#p12) | [Murai_MASt3R-SLAM_Real-Time_Dense_SLAM_with_3D_Reconstruction_Priors_CVPR_2025_paper.pdf](../../paper/Murai_MASt3R-SLAM_Real-Time_Dense_SLAM_with_3D_Reconstruction_Priors_CVPR_2025_paper.pdf) | 11 | Background; no SfM replacement | 2025 CVPR |
| [P13 — ORB-SLAM3](literature_review.md#p13) | [ORB-SLAM3 (2021).pdf](../../paper/ORB-SLAM3%20%282021%29.pdf) | 18 | Background; no SfM replacement | 2021 IEEE T-RO |
| [P14 — PGSR](literature_review.md#p14) | [pgsr.pdf](../../paper/pgsr.pdf) | 17 | Required implementation reference / B2 | 2024 TVCG; local 17-page manuscript |

## Dấu vết PDF

SHA-256 để gắn analysis với đúng bản local; không thay số liệu local bằng bảng ở website/phiên bản paper khác.

| ID | SHA-256 | Text chars |
| --- | --- | ---: |
| P01 | `f3091d687bef0e79ac774743e31678c3deed91c35df7d2b75ae392de5ff8f0c7` | 86220 |
| P02 | `f4c0c9f27e2d02b0265017f66049d471eb66cb50203bc918f51ba07d20cbbe11` | 110404 |
| P03 | `369a61c793063711cc39abaec58b9aadc23c9d40ccee675703ed84c3f012534d` | 80678 |
| P04 | `9f394f65e8b03f9f8e1db3ba4dd12ecce4a04b9cfd04a5e5ce19ead625bca9cd` | 128694 |
| P05 | `7804ce23f23bbe74be606c9ff9d002c218f18396afb0b9fa72269b5f9e959f37` | 99641 |
| P06 | `511b222dd4697b3a9c9fec7f6d54ddf9af795c677b2bd079a7f1f117a6c9ec1a` | 57136 |
| P07 | `00528437b6a36520e4eaca42532aad8d56bc5d883e38eeb1a609ad3c714b2143` | 93627 |
| P08 | `7fe04c358dcf68ec11274956c590aa632496e15b0ec4fdce0a92e3ef50a6bdbc` | 129742 |
| P09 | `2d0e80e784c881c4543b3dd5a87df65aefc4ff8683ab96d1e9d6a0e580eed23f` | 60636 |
| P10 | `2a31362c63972d8ede301376a4de9e12bf355d45db9e31999bf0ca59f23141ac` | 56390 |
| P11 | `e22db09e608af23d288503f65875431799e79e27c26f37da1c33ab09bb3875bf` | 103326 |
| P12 | `d0b520a8623b42e8421522bd0a3f4b6297692b0c88ca1ec6e251a9d7dd2d7d24` | 69450 |
| P13 | `4d119517be7565c9651fbf1e36a330d605b7a534d738558858c96feca7b99156` | 148049 |
| P14 | `92cb2a52ce951029b6dc4aeaf5acb1265cb3e384954ebf9ce81d9c9f18e04b9a` | 79854 |

## Tài liệu được yêu cầu nhưng không có PDF local

- *Structure-from-Motion Revisited* (Schönberger & Frahm, CVPR 2016): kiểm tra [CVF](https://openaccess.thecvf.com/content_cvpr_2016/html/Schonberger_Structure-From-Motion_Revisited_CVPR_2016_paper.html); [PDF chính thức](https://openaccess.thecvf.com/content_cvpr_2016/papers/Schonberger_Structure-From-Motion_Revisited_CVPR_2016_paper.pdf). Chỉ dùng làm nền tảng/nguồn định dạng COLMAP, không tính vào 14 PDF local.
- SuGaR, VGGT, VGGS: không có PDF trong cây `paper/` tại audit. SuGaR được phân biệt ở mức nguồn liên quan trong các PDF đã đọc; chưa có full primary-paper review local. VGGT/VGGS không thuộc pipeline được triển khai. Không tuyên bố đã đọc chúng như PDF local.
- TSDF: [Curless & Levoy 1996](https://graphics.stanford.edu/papers/volrange/) là nguồn bổ sung cho weighted distance fusion; đây là nguồn ngoài inventory.
- **CDGS** (Zhang, Wysocki, Urban & Jutzi, 2025, arXiv:2502.14684v1): bổ sung từ [bản HTML tác giả](https://arxiv.org/html/2502.14684v1), §3.1 Eq. (6), §3.2 Eq. (9–16). Đã xem methodology để kiểm tra overlap; tải PDF bằng Python gặp `CERTIFICATE_VERIFY_FAILED`, nên không tính là PDF đã đọc. Confidence weighting và reprojection weighting đã xuất hiện trước Ours.

## Implementation registry

Xác minh liên kết read-only ngày audit; truy cập được không có nghĩa đã build, license phù hợp mọi mục đích hoặc đã pin version. Trước cài đặt phải lưu commit, submodule SHAs, license và toolchain. Riêng PGSR đã đọc source snapshot commit `de24f1a38b350387e8d8fe381b2cd70c1ae946e7`; không clone hoặc chạy source.

| Paper | Nguồn tác giả / implementation | Evidence và giới hạn |
| --- | --- | --- |
| 3DGS | [graphdeco-inria/gaussian-splatting](https://github.com/graphdeco-inria/gaussian-splatting) | Reference repository; main hiện tại không đồng nghĩa bản 2023 |
| 2DGS | [hbb1/2d-gaussian-splatting](https://github.com/hbb1/2d-gaussian-splatting) | Tác giả; giữ renderer/checkpoint riêng |
| PGSR | [zju3dv/PGSR](https://github.com/zju3dv/PGSR) | Official; [paper/code comparison](pgsr_implementation_notes.md) |
| DN-Splatter | [maturk/dn-splatter](https://github.com/maturk/dn-splatter) | Hiện chung với AGS-Mesh; cần chọn đúng revision |
| GOF | [autonomousvision/gaussian-opacity-fields](https://github.com/autonomousvision/gaussian-opacity-fields) | Official; CUDA + tetrahedral dependencies chưa kiểm chứng Windows |
| 3DGSR | [URL tác giả công bố](https://arxiv.org/abs/2404.00409) | `github.com/CVMI-Lab/3DGSR` trả 404 trong web audit; **official code chưa xác minh truy cập được** |
| SurfaceSplat | [aim-uofa/SurfaceSplat](https://github.com/aim-uofa/SurfaceSplat) | Official; độc lập PGSR |
| MILo | [Anttwo/MILo](https://github.com/Anttwo/MILo) | Official; không đủ bằng chứng phù hợp VRAM 6GB |
| GVGS | [sumai-1998/GVGS](https://github.com/sumai-1998/GVGS) | URL `GVGScode/GVGS` trong PDF redirect về đây; author list local v3 khác abstract index cũ |
| MeshSplat | [project page](https://hanzhichang.github.io/meshsplat_web/) → [HanzhiChang/MeshSplat](https://github.com/HanzhiChang/MeshSplat) | Code link từ project tác giả; feed-forward cross-scene training |
| MASt3R-SLAM | [rmurai0610/MASt3R-SLAM](https://github.com/rmurai0610/MASt3R-SLAM) | Official; chỉ background |
| ORB-SLAM3 | [UZ-SLAMLab/ORB_SLAM3](https://github.com/UZ-SLAMLab/ORB_SLAM3) | Official; CPU reference, không thay SfM strict |
| Xu survey | [PeerJ](https://doi.org/10.7717/peerj-cs.3034) | Survey, không có một training implementation thống nhất |
| Matias survey | [publisher](https://doi.org/10.1016/j.cag.2026.104782) | Survey, phân biệt số liệu tổng hợp với experiment của survey |

Quy ước review: `Not specified in paper` chỉ dùng khi thông tin không được nêu cho trường đang xét; `Not applicable` cho thành phần không thuộc loại paper đó; `Chưa xác minh` cho kiểm chứng môi trường/code chưa thực hiện. Không biến ba trạng thái này thành một.

