# Kịch bản 45 phút — Bot 05 (Slide 17-20/41)

---
## Slide 17: Vấn đề triệt tiêu gradient (gradient cancellation)
**Lời thuyết trình (rút gọn):**
Đây là một hạn chế đã biết của 3DGS gốc: khi lấy trung bình gradient từ nhiều góc nhìn, nếu các vector này ngược hướng nhau, chúng có thể triệt tiêu lẫn nhau thay vì cộng dồn. Hệ quả là dù mỗi góc nhìn riêng lẻ đều báo Gaussian đó có sai số lớn, gradient trung bình tổng hợp lại có thể rất nhỏ — Gaussian bị bỏ sót khỏi danh sách densify dù đáng lẽ cần xử lý. Đây là hạn chế nội tại của việc chỉ dùng gradient trung bình làm tín hiệu duy nhất. FastGS khắc phục bằng multi-view consistency score, tức điểm nhất quán đa góc nhìn, sẽ trình bày kỹ hơn ở phần sau. Tiếp theo, chúng ta xem cơ chế densify với hai thao tác cụ thể: clone và split.

---
## Slide 18: Clone (nhân bản)
**Lời thuyết trình (rút gọn):**
Clone áp dụng cho Gaussian nhỏ nhưng thiếu hình học — vùng đó cần thêm chi tiết, nhưng bản thân Gaussian đã đủ nhỏ nên không cần tách. Cơ chế: nhân đôi Gaussian, bản sao đặt lệch một khoảng nhỏ theo đúng hướng gradient vị trí, tức về phía sai số lớn nhất. Quan trọng là clone không đổi kích thước hay ma trận hiệp phương sai — cả bản gốc lẫn bản sao giữ nguyên hình dạng. Mục đích duy nhất là tăng mật độ điểm để lấp đầy vùng thiếu chi tiết, như thêm quân tiếp viện vào đúng khu vực thiếu người chứ không thay đổi cấu trúc đã có. Tiếp theo là thao tác còn lại trong nhóm densify: split.

---
## Slide 19: Split (tách)
**Lời thuyết trình (rút gọn):**
Ngược với clone, split áp dụng cho Gaussian quá to, thuộc trường hợp over-reconstruction — một Gaussian đang cố biểu diễn vùng quá nhiều chi tiết so với kích thước của nó. Khi split, Gaussian cha tách thành hai Gaussian con nhỏ hơn, scale giảm theo hệ số cố định. Điểm đáng chú ý: vị trí các Gaussian con không cố định mà được lấy mẫu ngẫu nhiên theo phân phối chuẩn, xây dựng dựa trên chính ma trận hiệp phương sai của Gaussian cha — nên các điểm con phân bố theo đúng hình dạng, hướng và độ trải rộng mà cha vốn biểu diễn, tránh lệch lạc ngẫu nhiên. Sau khi có clone và split để tăng mật độ, ta chuyển sang chiều ngược lại: prune, để cắt giảm Gaussian dư thừa.

---
## Slide 20: Prune theo độ quan trọng đa góc nhìn (FastGS)
**Lời thuyết trình (rút gọn):**
Đây là một đóng góp chính của FastGS. Trong 3DGS gốc, prune chỉ dựa trên ngưỡng opacity thấp hoặc kích thước quá lớn — cách này đơn giản nhưng không phản ánh đúng mức đóng góp thực tế của từng Gaussian vào chất lượng ảnh. FastGS cải tiến bằng cách tính điểm importance cho mỗi Gaussian, dựa trên đóng góp thực tế vào chất lượng render, và tổng hợp điểm này qua nhiều góc nhìn khác nhau chứ không chỉ một góc. Gaussian có importance thấp nhất bị cắt trước tiên, bất kể opacity hay kích thước — ngay cả Gaussian opacity cao nhưng đóng góp thực tế thấp vẫn có thể bị loại. Kết quả: mô hình gọn hơn đáng kể về số lượng Gaussian nhưng vẫn giữ chất lượng tái tạo — đây là một trong những đòn bẩy chính giúp FastGS tăng tốc độ render. Tiếp theo, ta xem số lượng Gaussian biến đổi ra sao theo thời gian huấn luyện.
