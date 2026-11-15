# PGSR: paper và official implementation

Đối chiếu ngày14/11/2026, **chỉ đọc**, chưa compile/chạy. Paper: [P14](literature_review.md#p14), đặc biệt pp.5–9. Code snapshot: `de24f1a38b350387e8d8fe381b2cd70c1ae946e7`.

Nguồn pinned: [train.py](https://github.com/zju3dv/PGSR/blob/de24f1a38b350387e8d8fe381b2cd70c1ae946e7/train.py), [arguments](https://github.com/zju3dv/PGSR/blob/de24f1a38b350387e8d8fe381b2cd70c1ae946e7/arguments/__init__.py), [loss utilities](https://github.com/zju3dv/PGSR/blob/de24f1a38b350387e8d8fe381b2cd70c1ae946e7/utils/loss_utils.py). Đây là revision đã kiểm tra, không khẳng định là revision tạo bảng trong paper.

| Thành phần | Paper local | Code snapshot |
| --- | --- | --- |
| Geometry weights | single0.01, NCC0.2, geometry0.05 | defaults0.015/0.15/0.03 |
| Single/multi activation | Không nêu lịch đủ chi tiết ở §IV.D | sau7000 |
| Patch | 11×11, p.7 | radius3 → 7×7 |
| Single-view edge factor | Eq.(6) p.6 có dấu overline, power5 | complement normalized gradient, power2, detached |
| Multi-view residual | Forward/backward homography Eq.(9) | depth reprojection roundtrip; detached exp(−error) weighting, threshold1px |
| Seed | Không nêu seed cụ thể | module seed22 |
| Pixel sample budget | Không nêu | default102400 |

Không gọi hai cột tương đương. B2 mặc định nghiên cứu sẽ là **official-code variant** tại SHA này; `paper-faithful` nếu triển khai phải có config/experiment riêng. Baseline comparison dùng cùng project seed, override khác default phải được công bố.

`get_img_grad_weight` chia cho gradient range không có epsilon trong snapshot. Ảnh constant là một **failure case cần test**, chưa phải lỗi runtime đo trên Ignatius. Hardening numerical phải được version hóa và áp dụng giống B2/Ours. CUDA renderer và camera transforms cần roundtrip test; không suy dấu homography từ tên biến. Xem [mathematical foundations](mathematical_foundations.md).

Official training còn dùng shell commands kiểu Unix để backup source, custom extensions, app model và visibility trimming. Windows adapter phải giải quyết đường dẫn/toolchain và lưu state đầy đủ; không copy nguyên trainer vào `src/`. Chưa có bằng chứng extension hỗ trợ Python3.14/torch cu128/Toolkit12.1 của máy hiện tại.

Các giá trị `main` hoặc README sau này có thể đổi. Khi implementation bắt đầu: kiểm tra SHA/submodules, chọn variant, lập configuration diff, build trong environment riêng, chạy gradient smoke rồi mới training. Source code được đọc để phân biệt semantics; không coi là cho phép thực thi trong phiên tài liệu.
