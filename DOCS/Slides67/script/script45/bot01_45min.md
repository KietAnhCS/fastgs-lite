# Kịch bản 45 phút — Bot 01 (Slide 1-4/41)

---
## Slide 1: Trang bìa
**Lời thuyết trình (rút gọn):**
Kính chào thầy cô và các bạn. Em xin trình bày đồ án "Quy trình Render 3D Gaussian Splatting và Cải tiến FastGS", tên gọi tắt là FastGS-lite, với phụ đề "Từ toán học nền tảng đến Adaptive Density Control chuyên sâu". Đồ án được thực hiện bởi nhóm Digital Twin GS, dựa trên việc tái hiện và rút gọn thuật toán FastGS gốc để chạy được trên phần cứng phổ thông. Trong 45 phút tới, chúng em sẽ đi nhanh qua nền tảng toán học, sau đó tập trung vào đóng góp chính là cải tiến Adaptive Density Control, và kết quả thực nghiệm đạt được. Sau đây em xin trình bày nội dung chi tiết.

---
## Slide 2: Agenda
**Lời thuyết trình (rút gọn):**
Bài trình bày gồm bảy phần chính. Một, nền tảng 3D Gaussian và Spherical Harmonics — sẽ đi rất nhanh vì chỉ là kiến thức nền. Hai, chiếu phối cảnh và rasterizer khả vi. Ba, hàm mất mát, gradient và optimizer. Bốn — trọng tâm của đồ án — Adaptive Density Control, đặc biệt là các cải tiến nhóm em đề xuất. Năm, ba đòn bẩy tăng tốc của FastGS. Sáu, kết quả thực nghiệm khi triển khai FastGS-lite trên Colab T4. Bảy, quan sát, giới hạn và kết luận. Vì thời lượng có hạn, phần toán nền tảng sẽ được lược gọn tối đa, dành phần lớn thời gian cho đóng góp chính và số liệu thực nghiệm. Bắt đầu với động lực ra đời của 3D Gaussian Splatting.

---
## Slide 3: Động lực: vì sao cần một biểu diễn cảnh mới?
**Lời thuyết trình (rút gọn):**
Trước 3DGS, NeRF là hướng tiếp cận thống trị. NeRF biểu diễn cảnh "implicit" — thông tin màu và mật độ mã hóa ẩn trong trọng số một mạng MLP. Chất lượng cao nhưng render phải ray-marching, bắn tia và truy vấn mạng liên tục dọc tia, nên rất chậm, có thể mất vài giây mỗi khung hình. 3D Gaussian Splatting chuyển sang biểu diễn "explicit": cảnh được mô tả tường minh bằng hàng trăm nghìn quả cầu Gaussian mờ có vị trí, hình dạng rõ ràng. Nhờ đó render chỉ cần rasterization — kỹ thuật đồ họa truyền thống nhanh hơn ray-marching rất nhiều — đạt hàng chục đến hàng trăm khung hình mỗi giây, gần thời gian thực. Đây chính là lý do 3DGS trở thành hướng đi mới. Tiếp theo, ta xem một Gaussian 3D thực chất là gì.

---
## Slide 4: 3D Gaussian là gì?
**Lời thuyết trình (rút gọn):**
Đây là công thức trung tâm của phương pháp: hàm mật độ $G(x) = \exp(-\frac12(x-\mu)^T\Sigma^{-1}(x-\mu))$. Trực quan, điểm x càng gần tâm $\mu$ thì giá trị càng lớn, càng xa thì giảm về 0, tạo hình "quả chuông" mờ dần. Ma trận $\Sigma^{-1}$ quyết định quả chuông giãn nở nhanh hay chậm theo từng hướng — tức hình dạng và độ định hướng. Mỗi Gaussian được đặc trưng bởi bốn nhóm tham số học được: vị trí tâm $\mu$, hiệp phương sai $\Sigma$ quy định hình dạng, độ mờ opacity $\alpha$, và màu sắc qua hệ số Spherical Harmonics. Bốn tham số này là những gì mô hình tối ưu trong suốt quá trình huấn luyện. Tiếp theo, chúng ta sẽ đi nhanh qua phần chiếu phối cảnh và rasterizer trước khi vào phần trọng tâm: Adaptive Density Control.
