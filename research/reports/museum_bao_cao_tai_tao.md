# Báo cáo tái tạo cảnh Museum

## Dữ liệu và giao thức

Thí nghiệm sử dụng video đơn mắt `datasets/tanks_and_temples/museum/Museum.mp4` thuộc tập kiểm thử nâng cao. Video H.264 có 17.115 khung hình, độ phân giải 3840 × 2160, tốc độ 30000/1001 fps và thời lượng 571,0705 giây. SHA-256 của video là `f407eca1d989587c5c350fce42156c235b6e17dedc08c1013fa739d1088d83f2`. Mọi dữ liệu đầu vào và sản phẩm của Museum được quản lý riêng với Ignatius. [Nguồn chuẩn Tanks and Temples](https://tanksandtemples.org/download/).

Bộ dữ liệu `museum-video-v1-04702c7617547b96` chứa 159 ảnh đã hiệu chỉnh méo tại 1280 × 720 và 27.435 điểm khởi tạo chỉ từ các góc nhìn huấn luyện. Phân hoạch cố định gồm 119 ảnh huấn luyện, 20 ảnh xác thực và 20 ảnh kiểm thử. Các tham số ngoại chuẩn biểu diễn phép biến đổi thế giới sang camera theo quy ước COLMAP Hamilton WXYZ. Đơn vị SfM thuộc thang đo đơn mắt tùy ý, và phép chuẩn hóa được ghi trong metadata. Ước lượng camera có tính chuyển dẫn vì SfM dùng toàn bộ tập khung hình đã chọn.

## Tiền xử lý và tái tạo thưa

| Công đoạn | Kết quả quan sát | Thời gian |
| --- | --- | --- |
| Trích xuất theo PTS gốc | 1.142 ảnh ứng viên ở 2 fps, chiều rộng tối đa 1280 | 150,78 s |
| Chọn khung hình trên CUDA | 160 ảnh, khoảng cách lớn nhất giữa hai ảnh được chọn là 7,007 s | Xấp xỉ 55 s, gồm kiểm tra và xuất dữ liệu |
| COLMAP SIFT CUDA | 820.393 mô tả và điểm đặc trưng | 6,76 s |
| Ghép tuần tự và kiểm tra hình học | 729 cặp được xác minh, 212.021 tương ứng inlier | 58,30 s |
| Ánh xạ COLMAP/CASPAR | 159/160 ảnh đăng ký, tỷ lệ 99,375% | 161,51 s |
| Mô hình thưa dẫn xuất | 31.447 điểm, độ dài track trung bình 4,077, sai số chiếu lại trung bình 0,528 px, p95 1,392 px | Xuất và kiểm tra độc lập sau ánh xạ |
| Chuẩn bị dữ liệu huấn luyện | Loại 275 track đi ra ngoài ảnh sau hiệu chỉnh méo, giữ 31.172 điểm ở mô hình dùng chung | 67,70 s |

Phiên bản COLMAP sẵn có dừng ở mã lỗi Windows `0xC0000409` trong phép thử bundle adjustment. Bản CUDA/CASPAR 4.2.1 thay thế giảm sai số trên mô hình kiểm tra từ 3,018 px xuống 0,000067 px. Lần ánh xạ có tùy chọn mặc định kích hoạt nhánh đăng ký structure-less và gọi Ceres. Lần chạy nghiêm ngặt tắt nhánh đó, xác minh nhật ký chỉ dùng CASPAR cho BA và giữ nguyên kết quả của các lần thử trước. Mô hình thưa gốc bị loại do 289 track chứa quan sát trùng trong một ảnh. Bản dẫn xuất loại bỏ toàn bộ các track này, giữ nguyên ngoại chuẩn, nội chuẩn và tọa độ XYZ của các điểm còn lại.

VRAM lấy mẫu trên toàn thiết bị đạt 968 MiB ở bước SIFT, 462 MiB ở bước ghép và 218 MiB ở bước ánh xạ. Tensor chọn khung hình đạt 24 MiB bộ nhớ PyTorch dự trữ. Các số đo có phạm vi khác nhau và không biểu diễn đỉnh cấp phát chính xác của từng tiến trình.

## Huấn luyện và kết xuất

Hai phương pháp có bộ huấn luyện trong dự án là B0 Original3DGS và B2 PGSR. B1 2DGS, B3 GOF và phương pháp đề xuất vẫn ở phạm vi kế hoạch. Kiểm tra CUDA tổng hợp của B0 đã thực hiện raster hóa, lan truyền ngược và một cập nhật optimizer. Kiểm tra tổng hợp của PGSR đã xác thực đạo hàm sai phân hữu hạn, xử lý NCC trên ảnh hằng và cổng bù phơi sáng. Kết quả các cấu hình chạy thử và cấu hình nghiên cứu được ghi riêng để tránh so sánh giữa các độ phân giải và tập ảnh tối ưu khác nhau.

| Phương pháp và phạm vi | Bước khôi phục → hoàn tất | PSNR / SSIM / LPIPS trên 20 ảnh PNG | Tổng loss cuối | Gaussian cuối | Thời gian vòng lặp và xuất |
| --- | --- | --- | ---: | ---: | ---: |
| B0, thử nghiệm, 12 ảnh huấn luyện, 320 × 180 | 550 → 1.200 | 12,7081 / 0,3185 / 0,4757 | 0,044246 | 79.570 | 13,24 s |
| B2 PGSR, thử nghiệm, 8 ảnh huấn luyện, 160 × 90 | 80 → 300 | 8,8862 / 0,0843 / 0,7375 | 1,503249 | 2.230 | 10,46 s |

Hai mức độ phân giải và tập huấn luyện khác nhau. Các giá trị trên chỉ xác nhận quy trình chạy thử và không dùng để xếp hạng mô hình. PGSR đã tải lại Gaussian PLY và xuất tám bộ ảnh độ sâu, pháp tuyến và mặt nạ. Bộ đánh giá đầu tiên đã ghi ảnh kiểm thử nhưng thất bại vì số lượng 33 ảnh được gán cố định theo Ignatius. Bản sửa đối chiếu trực tiếp danh sách ID của phân hoạch Museum. Lần đánh giá sau đã khớp đủ 20 ảnh với chỉ số hữu hạn.

Các bản ghi gốc được lưu tại `training.jsonl`. CSV và hình đường loss nằm trong thư mục `losses/` của từng lần chạy. Sai lệch tuyệt đối lớn nhất khi cộng các thành phần có trọng số là 1,19 × 10⁻⁸ với B0 và 2,83 × 10⁻⁶ với B2. Các kích thước checkpoint tương ứng là 37.261.887 và 1.712.615 byte. Dung lượng RAM đỉnh và hiệu năng suy luận theo một điều kiện phần cứng đối sánh chưa được đo.

Hình kiểm tra lấy từ dữ liệu thật của Museum. [Góc nhìn mô hình thưa](../../outputs/tanks_and_temples/Museum/evaluation/museum_sparse_views.png) · [Ảnh RGB thử nghiệm](../../outputs/tanks_and_temples/Museum/evaluation/museum_smoke_rgb.png) · [Độ sâu PGSR](../../outputs/tanks_and_temples/Museum/research/B2_PGSR/museum-video-pgsr-smoke-ad6f30b634/evaluation/geometry_preview/000002_depth.png) · [Pháp tuyến PGSR](../../outputs/tanks_and_temples/Museum/research/B2_PGSR/museum-video-pgsr-smoke-ad6f30b634/evaluation/geometry_preview/000002_normal.png).

## Phạm vi đánh giá hình học

Không có tham chiếu Museum cục bộ đã được xác minh về đơn vị, đăng ký, vùng đánh giá hoặc ngưỡng chuẩn. Chamfer Distance, Accuracy, Completeness, Precision, Recall, F-score, MAE, RMSE, AbsRel, Normal Consistency và sai số góc pháp tuyến đều là N/A. B0 xuất Gaussian PLY và ảnh RGB. PGSR có đường xuất độ sâu trục Z, pháp tuyến đơn vị theo camera, mặt nạ hợp lệ và trích xuất lưới CUDA TSDF trên Kaggle. Các số đo cấu trúc lưới chỉ được công bố sau khi trích xuất và đọc lại hiện vật thực tế.

[Hướng dẫn thực thi và đóng gói](../../docs/tech/museum_usage.md).

## Thực thi trên Kaggle

Gói nguồn và cảnh chuẩn bị đã phát hành riêng tư. Phiên B0 v2 đã xác thực gói cảnh và nhận đúng phân hoạch Museum trên Kaggle. PGSR dùng cùng gói cảnh. Notebook lưới đã được lưu và yêu cầu checkpoint của lần chạy PGSR Museum.

[Dữ liệu Museum](https://www.kaggle.com/datasets/vthkhnhhuyn/museum-prepared-video-v1) · [B0](https://www.kaggle.com/code/vthkhnhhuyn/museum-original-3dgs-v1) · [PGSR](https://www.kaggle.com/code/vthkhnhhuyn/museum-pgsr-v1) · [CUDA TSDF](https://www.kaggle.com/code/vthkhnhhuyn/museum-pgsr-mesh-v1).
