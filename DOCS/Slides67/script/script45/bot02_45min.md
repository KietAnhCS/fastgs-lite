# Kịch bản 45 phút — Bot 02 (Slide 5-8/41)

---
## Slide 5: Tham số hoá hiệp phương sai
**Lời thuyết trình (rút gọn):**
Nếu học trực tiếp ma trận hiệp phương sai $\Sigma$ với 6 giá trị tự do, gradient có thể khiến nó mất tính positive semi-definite, làm công thức Gaussian vô nghĩa. Giải pháp: không học $\Sigma$ trực tiếp, mà học một vector scale $s$ ba chiều và một quaternion $q$ bốn chiều biểu diễn phép xoay. Từ đó dựng lại $\Sigma = R S S^T R^T$, với $S$ là ma trận đường chéo từ $s$, $R$ là ma trận xoay từ $q$. Cách phân tích kiểu SVD này đảm bảo $\Sigma$ luôn hợp lệ dù $s$, $q$ nhận giá trị gì. Scale ban đầu được khởi tạo dựa trên khoảng cách tới điểm lân cận trong point cloud COLMAP. Về mặt hình học, $\Sigma$ chính là một ellipsoid: trục và độ dài trục lần lượt là eigenvector và eigenvalue của nó — Gaussian càng dẹt thì càng bất đẳng hướng. Tiếp theo, ta xem màu sắc được biểu diễn thế nào.

---
## Slide 6: Màu sắc qua Spherical Harmonics (SH)
**Lời thuyết trình (rút gọn):**
Mỗi Gaussian không lưu một màu RGB cố định mà lưu hệ số Spherical Harmonics, vì màu quan sát thực tế thay đổi theo góc nhìn — ví dụ hiệu ứng phản xạ, độ bóng. Biểu diễn màu như hàm phụ thuộc hướng nhìn qua hệ số SH cho phép mô hình tính lại màu phù hợp với từng góc camera. Hệ số SH bậc 0, gọi là thành phần DC, được khởi tạo trực tiếp từ màu RGB quan sát qua công thức $f_{dc} = \text{RGB2SH}(\text{color})$. Bậc càng cao thì càng biểu diễn được nhiều biến thiên màu phức tạp theo góc nhìn, nhưng tốn thêm bộ nhớ — nên khi huấn luyện, bậc SH được tăng dần chứ không dùng tối đa ngay. Vì đồ án đã trình bày đầy đủ phần suy diễn toán học của SH, trong bài báo cáo hôm nay nhóm em xin phép bỏ qua phần này để dành thời gian tập trung vào trọng tâm chính: Adaptive Density Control.

---
## Slide 7: Tổng quan pipeline render
**Lời thuyết trình (rút gọn):**
Sau khi đã xong phần nền tảng 3D Gaussian, ta chuyển sang phần thứ hai: chiếu phối cảnh và rasterizer khả vi — bước biến các Gaussian 3D cùng hệ số SH vừa học thành một bức ảnh 2D cụ thể. Pipeline gồm năm bước: chuyển từ tọa độ thế giới sang tọa độ camera bằng ma trận quay R và tịnh tiến t; chiếu Gaussian 3D xuống mặt phẳng ảnh 2D; tiling để xác định Gaussian nào ảnh hưởng tile nào; sorting theo độ sâu trong từng tile; và cuối cùng alpha blending để ra màu pixel. Toàn bộ chuỗi năm bước này đều khả vi, cho phép lan truyền gradient ngược từ ảnh render về tham số Gaussian. Tiếp theo, ta đi vào chi tiết bước đầu tiên: phép chiếu camera pinhole và chiếu hiệp phương sai.

---
## Slide 8: Chiếu hiệp phương sai (EWA splatting)
**Lời thuyết trình (rút gọn):**
Phép chiếu phối cảnh phi tuyến do có phép chia cho z, nên không thể áp dụng trực tiếp cho hình dạng elip 3D của Gaussian. Giải pháp là xấp xỉ tuyến tính cục bộ quanh tâm Gaussian bằng ma trận Jacobian J. Hiệp phương sai 2D sau chiếu được tính bằng $\Sigma' = JW\Sigma W^TJ^T$, với W là phần quay của ma trận view — đây là quy tắc biến đổi hiệp phương sai qua một ánh xạ tuyến tính gần đúng. Kết quả là mỗi Gaussian 3D trở thành một ellipse 2D cụ thể trên ảnh, có tâm và hình dạng xác định rõ ràng. Kỹ thuật này gọi là Elliptical Weighted Average splatting, viết tắt EWA splatting, nền tảng kinh điển của các phương pháp splatting. Tiếp theo, ta sẽ xem cách FastGS tối ưu vùng ảnh hưởng của ellipse này để tăng tốc rasterization.
