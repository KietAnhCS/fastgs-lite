# Kịch bản 45 phút — Bot 06 (Slide 21-24/41)

---
## Slide 21: Reset opacity định kỳ
**Lời thuyết trình (rút gọn):**
Cơ chế cuối trong Adaptive Density Control: cứ mỗi opacity_reset_interval, mặc định 3000 iteration, toàn bộ opacity của mọi Gaussian bị đặt lại về giá trị thấp qua sigmoid-inverse. Mục đích là buộc optimizer đánh giá lại vai trò từng Gaussian, thay vì để những Gaussian đã đạt opacity cao giữ nguyên trạng thái dù không còn cần thiết. Sau reset, Gaussian thực sự quan trọng sẽ nhanh chóng tăng opacity trở lại nhờ gradient; Gaussian dư thừa không kịp hồi phục sẽ bị prune ở vòng kế tiếp. Đây là bước dọn dẹp định kỳ giúp scene không tích tụ Gaussian "chết" theo thời gian.

Đến đây, em xin kết thúc phần tổng quan Loss, Gradient, Optimizer và ADC. Bây giờ ta đi sâu vào cách tính điểm Importance đa view — đóng góp cốt lõi của FastGS-lite.

---
## Slide 22: Tổng hợp counts qua nhiều view: đầu vào cho Importance
**Lời thuyết trình (rút gọn):**
Sau khi có counts cho từng Gaussian ở từng view riêng lẻ, bước tiếp theo là tổng hợp lại. Mỗi Gaussian có một vector counts ứng với các view nó xuất hiện. Lưu ý: tổng counts trên một view luôn ≥ số pixel lỗi thật của view đó, vì một pixel lỗi có thể nằm trong footprint của nhiều Gaussian chồng lấn nên bị đếm lặp — điều này có chủ đích, vì mục tiêu là quy trách nhiệm cho tất cả Gaussian liên quan, không chia đều lỗi. Về cài đặt, mỗi lần render một view, kernel CUDA cộng dồn atomic vào tensor metricCount, rồi lũy kế qua tất cả view thành full_metric_counts. Vector counts tổng hợp qua toàn bộ V view này chính là đầu vào trực tiếp cho bước tính Importance.

---
## Slide 23: Importance score: trung bình counts qua các view, làm tròn xuống
**Lời thuyết trình (rút gọn):**
Đây là công thức trung tâm, khớp với hàm compute_gaussian_score_fastgs trong code. Importance của một Gaussian bằng trung bình cộng counts qua tất cả view, rồi làm tròn xuống (torch.div chế độ floor). Một Gaussian được coi là "quan trọng" — giữ lại hoặc ưu tiên densify — khi Importance vượt quá 5. Dùng trung bình thay vì max hay tổng có lý do: nó đảm bảo Gaussian phải nhất quán gây lỗi qua nhiều góc nhìn, chứ không chỉ nổi bật ở một view đơn lẻ, mới được coi là thực sự quan trọng — tránh việc một góc nhìn bất thường quyết định số phận Gaussian.

Từ điểm Importance này, ta chuyển sang xem nó được dùng ở đâu trong quyết định densify.

---
## Slide 24: Sơ đồ quyết định densify đầy đủ
**Lời thuyết trình (rút gọn):**
Đây là sơ đồ tổng hợp toàn bộ logic densify của FastGS-lite, áp dụng mỗi 100-500 vòng lặp, trong khoảng iteration 500-15000. Bước một: dựa vào kích thước Gaussian so với ngưỡng để chọn nhánh. Nếu nhỏ — "under-reconstruction" — thì kiểm tra gradient có dấu vượt ngưỡng và Importance > 5; nếu đúng cả hai thì CLONE, tạo bản sao giữ nguyên tham số gốc. Nếu to — "over-reconstruction" — thì kiểm tra gradient trị tuyệt đối vượt ngưỡng cao hơn và Importance > 5; nếu đúng thì SPLIT thành hai Gaussian con, xóa Gaussian gốc. Điểm mấu chốt: nếu Importance không thỏa, 3DGS gốc vẫn densify chỉ cần gradient đạt, nhưng FastGS-lite kiên quyết chặn lại — đây chính là đóng góp cốt lõi giúp tránh sinh thừa Gaussian không cần thiết.
