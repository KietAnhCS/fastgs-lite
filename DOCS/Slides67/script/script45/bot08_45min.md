# Kịch bản 45 phút — Bot 08 (Slide 29-32/41)

---
## Slide 29: Tổng kết 5 cải tiến của FastGS-lite trong ADC
**Lời thuyết trình (rút gọn):**
Em tổng kết nhanh 5 cải tiến FastGS-lite đưa vào chu trình ADC so với 3DGS gốc. Một: quyết định densify không chỉ dựa gradient mà AND thêm với Importance đa view, lọc bớt ứng viên sai. Hai: split dùng tổng trị tuyệt đối gradient thay vì tổng có dấu, tránh bị triệt tiêu khi Gaussian đúng vị trí nhưng sai kích thước. Ba: bước prune không xoá cứng toàn bộ ứng viên mà rút mẫu multinomial theo trọng số, giữ lại phần còn hữu ích. Bốn: chặn trần opacity ở 0,8 để không "chôn" gradient của các lớp Gaussian phía sau. Và cuối cùng, cải tiến năm: thêm bước final prune sau khi densify kết thúc, tại t=1000 và t=1100, xoá theo alpha thấp hoặc xác suất Pruning cao — tạo ra hai bậc thang giảm N rõ rệt mà 3DGS gốc không có. Kết quả chung: loss L1 tương đương 3DGS gốc nhưng N nhỏ hơn hẳn — mô hình gọn nhất trong ba kịch bản so sánh. Tiếp theo, em chuyển sang phần ba đòn bẩy tăng tốc của FastGS.

---
## Slide 30: Vì sao 3DGS gốc chậm?
**Lời thuyết trình (rút gọn):**
Trước khi nói về tăng tốc, em phân tích 3DGS gốc chậm ở đâu. Mỗi iteration gồm bốn khâu: render, tính loss, backward, và định kỳ densify/prune. Với cảnh thực có hàng trăm nghìn đến hàng triệu Gaussian, ước tính render chiếm khoảng 45% thời gian, backward khoảng 35%, tính loss chỉ 10%, còn densify/prune tuy chỉ khoảng 10% tổng thời gian nhưng chạy định kỳ và gây đột biến chi phí mỗi lần kích hoạt. Nhìn vào tỉ trọng này, rõ ràng muốn tăng tốc hiệu quả thì phải nhắm đúng hai khâu chiếm phần lớn nhất là render và backward, thay vì tối ưu dàn trải. Đây chính là tiền đề để áp dụng định luật Amdahl: tăng tốc một phần hệ thống chỉ có lợi tương ứng với tỉ trọng thời gian phần đó chiếm trong toàn pipeline. Dựa trên phân tích này, FastGS chọn lọc đúng ba đòn bẩy tăng tốc, em sẽ giới thiệu tổng quan ngay sau đây.

---
## Slide 31: Ba đòn bẩy tăng tốc của FastGS (tổng quan)
**Lời thuyết trình (rút gọn):**
Đây là slide tổng quan quan trọng nhất của phần này: ba đòn bẩy tăng tốc chính của FastGS. Một, Multi-view score: đánh giá tầm quan trọng của Gaussian qua nhiều góc nhìn thay vì một view, giúp pruning chính xác hơn, giảm N hiệu quả hơn. Hai, Compact Box: giảm hệ số nhân bounding-box từ 3,0 mặc định xuống 0,5-0,7, khiến số tile mỗi Gaussian chạm tới giảm theo bình phương hệ số, kéo giảm trực tiếp chi phí rasterize. Ba, Sparse Adam: áp dụng lịch cập nhật phân tầng, giảm tần suất gọi optimizer.step() cho các tham số ít quan trọng như SH bậc cao. Cả ba đòn bẩy đều nhắm đúng những khâu chiếm tỉ trọng lớn theo phân tích Amdahl vừa nêu — rasterize, densify/prune, và bước cập nhật tham số — và gần như không chồng lấn nhau. Slide tiếp theo em sẽ cho thấy hiệu quả tích luỹ khi kết hợp cả ba.

---
## Slide 32: Hiệu năng tích luỹ — Biểu đồ thác nước
**Lời thuyết trình (rút gọn):**
Slide này tổng hợp hiệu quả ba đòn bẩy bằng biểu đồ thác nước, thể hiện mức tăng tốc tích luỹ khi bật lần lượt từng đòn bẩy lên nền baseline 3DGS gốc: bắt đầu từ baseline, cộng Compact Box, cộng tiếp Multi-view score, cộng tiếp Sparse Adam, và cuối cùng là FastGS đầy đủ với cả ba cùng hoạt động. Điểm mấu chốt là mỗi đòn bẩy đóng góp tăng tốc độc lập, và khi kết hợp cả ba thì mức tăng tốc tổng vượt trội hẳn so với dùng riêng lẻ. Điều này khớp hoàn toàn với dự đoán từ định luật Amdahl: vì ba đòn bẩy tác động vào ba khâu chi phí khác nhau, gần như không chồng lấn — rasterize, densify/prune, optimizer step — nên hiệu quả cộng dồn tự nhiên chứ không triệt tiêu lẫn nhau. Đây là bằng chứng thực nghiệm cho toàn bộ chiến lược thiết kế của FastGS.
