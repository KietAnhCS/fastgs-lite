# Kịch bản 45 phút — Bot 04 (Slide 13-16/41)

---
## Slide 13: Gradient tập trung tại biên Gaussian
**Lời thuyết trình (rút gọn):**
Một quan sát quan trọng: gradient trong không gian view không phân bố đều trên Gaussian mà tập trung mạnh nhất ở vùng biên — nơi ranh giới với nền hoặc Gaussian khác, cũng là nơi ảnh render dễ lệch nhiều nhất so với ảnh thật. Ý nghĩa thực tiễn rất lớn: gradient tại biên chính là tín hiệu để hệ thống quyết định có densify hay không, tức có thêm Gaussian mới vào vùng đó không, ở bước Adaptive Density Control. Ngược lại, vùng gradient thấp nghĩa là Gaussian đã khớp tốt với dữ liệu, không cần chia tách thêm. Đây là nền tảng kết nối phần loss, gradient với phần điều chỉnh mật độ Gaussian. Tiếp theo, ta tổng kết lại toàn bộ vòng lặp huấn luyện trước khi đi sâu vào phần densify/prune.

---
## Slide 14: Tổng kết vòng lặp huấn luyện
**Lời thuyết trình (rút gọn):**
Toàn bộ vòng lặp huấn luyện gồm năm bước lặp lại liên tục: Render ảnh từ tập Gaussian hiện tại theo góc camera; Tính loss kết hợp L1 và D-SSIM; Backward — lan truyền ngược gradient qua rasterizer khả vi về từng tham số; optimizer.step() dùng sparse Adam để cập nhật tham số hiệu quả; và cuối cùng, định kỳ chứ không phải mọi iteration, là Densify/Prune — điều chỉnh mật độ Gaussian. Sau bước năm, vòng lặp quay lại bước một, lặp hàng chục nghìn iteration tới khi hội tụ. Đây là bước chuyển rất quan trọng: từ đây, em sẽ đi sâu vào chính phần trọng tâm của đồ án — Adaptive Density Control, cơ chế quyết định trực tiếp chất lượng và hiệu năng của FastGS-lite.

---
## Slide 15: Tổng quan Adaptive Density Control (ADC)
**Lời thuyết trình (rút gọn):**
ADC giải quyết vấn đề: point cloud khởi tạo từ COLMAP thường không hoàn hảo — có vùng quá thưa, thiếu điểm ở khu vực chi tiết phức tạp, trong khi có Gaussian quá to che khuất chi tiết nhỏ. ADC chạy định kỳ sau mỗi khoảng densification_interval iteration, điều chỉnh động mật độ Gaussian theo độ phức tạp hình học thực tế của scene. ADC gồm hai nhóm thao tác: Densify, gồm clone (nhân bản) và split (tách), dùng để thêm Gaussian ở nơi thiếu hình học; và Prune, cắt tỉa Gaussian thừa, mờ hoặc không còn đóng góp giá trị. Mục tiêu là cân bằng chất lượng tái tạo với số lượng Gaussian, vì số lượng lớn ảnh hưởng trực tiếp tới bộ nhớ và tốc độ render. Tiếp theo, ta xem tín hiệu cụ thể để quyết định densify: gradient view-space tích lũy.

---
## Slide 16: Tích luỹ gradient view-space
**Lời thuyết trình (rút gọn):**
Để quyết định Gaussian nào cần densify, hệ thống cần tín hiệu định lượng: gradient view-space tích lũy qua thời gian. Với mỗi Gaussian, gradient vị trí 2D sinh ra từ mỗi lần render ở góc camera khác nhau được cộng dồn lại, đồng thời đếm denom — số lần Gaussian thực sự được nhìn thấy. Gradient trung bình g-bar bằng tổng gradient view-space chia cho denom. Nếu g-bar lớn, nghĩa là Gaussian liên tục gây sai số đáng kể qua nhiều góc nhìn, nó trở thành ứng viên clone hoặc split. Cách tích lũy qua nhiều view giúp tín hiệu ổn định hơn so với dùng một lần render đơn lẻ. Tuy nhiên, chính cách lấy trung bình vector này cũng tiềm ẩn một vấn đề quan trọng — gradient cancellation — mà slide tiếp theo sẽ phân tích.
