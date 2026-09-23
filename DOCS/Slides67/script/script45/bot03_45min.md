# Kịch bản 45 phút — Bot 03 (Slide 9-12/41)

---
## Slide 9: Compact Box — đòn bẩy tăng tốc #1 của FastGS
**Lời thuyết trình (rút gọn):**
Sau khi chiếu Gaussian 3D thành ellipse 2D, hệ thống cần một bounding-box bao quanh để biết vùng ảnh hưởng trên ảnh. Cài đặt gốc dùng bounding-box 3-sigma — an toàn về thống kê nhưng khá rộng, khiến Gaussian chạm nhiều tile hơn mức cần thiết, gây lãng phí tính toán. FastGS thay bằng compact box nhỏ gọn hơn, điều chỉnh qua tham số mult, mặc định 0.5, có thể tăng lên 0.7 cho cảnh lớn để tránh cắt mất phần đuôi Gaussian quan trọng. Kết quả là giảm đáng kể số tile mỗi Gaussian phải rasterize, tức giảm tải tính toán mà không đánh đổi nhiều chất lượng hình ảnh. Đây là đòn bẩy tăng tốc đầu tiên và quan trọng nhất của FastGS. Tiếp theo, ta xem công thức alpha blending — bước kết hợp các Gaussian này thành màu pixel cuối cùng.

---
## Slide 10: Alpha blending (front-to-back)
**Lời thuyết trình (rút gọn):**
Đây là công thức quyết định màu cuối cùng của mỗi pixel: C bằng tổng của ci nhân alpha_i nhân Ti, theo đúng thứ tự độ sâu đã sắp xếp. Trong đó ci là màu Gaussian giải mã từ hệ số SH, alpha_i là độ mờ tại điểm đó, còn Ti là transmittance — độ truyền sáng còn lại, bằng tích của (1 trừ alpha_j) cho mọi Gaussian j gần camera hơn i. Nói cách khác, Ti đo "còn bao nhiêu ánh sáng lọt qua" sau khi các lớp phía trước đã hấp thụ một phần. Gaussian càng gần camera thì đóng góp càng lớn vào màu cuối cùng, vì chưa bị lớp nào che khuất. Đây là mô hình front-to-back alpha compositing kinh điển. Vì Ti giảm theo cấp số nhân khi có nhiều lớp che, khi Ti đủ nhỏ, rasterizer có thể dừng sớm (early stopping) để tiết kiệm tính toán mà không ảnh hưởng chất lượng. Slide sau sẽ tổng kết toàn bộ phương trình render.

---
## Slide 11: Tổng kết phương trình render
**Lời thuyết trình (rút gọn):**
Slide này tổng kết toàn bộ phần chiếu phối cảnh và rasterizer bằng một phương trình duy nhất: C tại p bằng tổng ci nhân alpha_i nhân tích (1 trừ alpha_j) với j nhỏ hơn i — chính là công thức blend vừa nêu, viết gọn thành phương trình render hoàn chỉnh cho cả pipeline. Điều cốt lõi là toàn bộ phương trình này khả vi theo mọi tham số: vị trí tâm mu, hiệp phương sai Sigma, độ mờ alpha, và hệ số SH quyết định màu. Nhờ tính khả vi đầy đủ này, ta huấn luyện trực tiếp bằng gradient descent: so sánh ảnh render với ảnh RGB thực tế, rồi lan truyền ngược sai số qua toàn bộ chuỗi Gaussian 3D, chiếu 2D, trường alpha, blend, để cập nhật từng tham số. Đây là điểm khép lại phần render, và cũng là lúc chuyển sang phần tiếp theo: hàm mất mát và cách tối ưu các tham số này.

---
## Slide 12: Hàm mất mát: kết hợp L1 và D-SSIM
**Lời thuyết trình (rút gọn):**
Chuyển sang phần huấn luyện: hàm mất mát dùng để tối ưu tham số Gaussian. Loss kết hợp hai thành phần bổ trợ nhau. Thứ nhất là L1, đo sai khác trung bình theo từng pixel giữa ảnh render và ảnh gốc — tín hiệu cơ bản, đơn giản nhưng nhạy với nhiễu. Thứ hai là D-SSIM, bằng 1 trừ SSIM, phản ánh sai lệch về cấu trúc chứ không chỉ độ sáng từng điểm. Công thức tổng là trung bình có trọng số: L bằng (1 trừ lambda) nhân L1 cộng lambda nhân (1 trừ SSIM). Giá trị lambda_dssim mặc định của 3DGS gốc là 0.2, ưu tiên L1 nhiều hơn; FastGS-lite tăng lên 0.25 để nhấn mạnh cấu trúc, giữ chi tiết hình học tốt hơn khi giảm số Gaussian. Tiếp theo ta sẽ đi sâu vào cơ chế của SSIM.
