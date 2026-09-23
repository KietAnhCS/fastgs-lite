# Kịch bản 45 phút — Bot 09 (Slide 33-37/41)

---
## Slide 33: Tổng kết: từ 3DGS gốc đến FastGS
**Lời thuyết trình (rút gọn):**
Slide này tổng kết hành trình từ 3DGS gốc sang FastGS qua bảng so sánh tham số then chốt. FastGS thêm loss_thresh 0,1; đổi gradient thường sang Abs-GS với ngưỡng 0,0012 để khắc phục gradient cancellation; tham số dense dao động 0,001-0,013 theo kích thước cảnh; learning rate bậc cao giảm còn 0,005-0,02; và hệ số bounding-box mult giảm từ 3,0 xuống 0,5-0,7. Điểm mấu chốt: FastGS không đổi mô hình biểu diễn Gaussian, chỉ tối ưu tham số hóa và lịch huấn luyện dựa trên mô hình chi phí và định luật Amdahl đã phân tích, giúp tăng tốc huấn luyện đáng kể mà PSNR/SSIM gần như không đổi. Sau phần lý thuyết này, em chuyển sang kết quả thực nghiệm thực tế trên bản fork fastgs-lite chạy trên Colab T4.

---
## Slide 34: FastGS-lite: từ 24GB RTX 4090 xuống Colab T4 miễn phí
**Lời thuyết trình (rút gọn):**
fastgs-lite là bản fork của nhóm, đóng gói lại FastGS gốc để chạy trên phần cứng phổ thông. Điểm quan trọng: nó không đổi bất kỳ công thức toán hay pipeline huấn luyện nào so với FastGS gốc — chỉ đổi cách đóng gói và vận hành, để bài toán vốn cần GPU 24GB như RTX 4090 chạy được trên Colab T4 miễn phí, chỉ khoảng 15GB VRAM. Cụ thể, CUDA extension được vendor sẵn trong repo thay vì submodule rời rạc, entry point bổ sung notebook có theo dõi tiến trình, và tín hiệu huấn luyện được đổi thành một Score tổng hợp log mỗi 1000 iteration. Đây là nền tảng để hiểu các bảng số liệu thực nghiệm tiếp theo, bắt đầu với chất lượng cuối cùng theo từng scene.

---
## Slide 35: Bảng 1: Chất lượng cuối cùng theo scene
**Lời thuyết trình (rút gọn):**
Đây là bảng kết quả quan trọng nhất. Score trung bình trên 4 scene đạt 0,7643, PSNR trung bình 24,46dB. Theo từng scene: playroom cao nhất với 0,8198, tiếp đến drjohnson 0,8027, rồi truck 0,7482, và thấp nhất là train chỉ 0,6867. Đáng chú ý, hai scene indoor — playroom và drjohnson — vượt trội hơn hai scene outdoor dù chỉ dùng khoảng 129-174 nghìn Gaussian, ít hơn hẳn so với 187-201 nghìn Gaussian của scene outdoor. Điều này cho thấy chất lượng không tỷ lệ thuận với số lượng Gaussian mà phụ thuộc nhiều vào đặc tính hình học và ánh sáng của cảnh. Tiếp theo em trình bày chi phí tính toán và lưu trữ để đo hiệu quả thực tế của pipeline này.

---
## Slide 36: Bảng 3: Chi phí & lưu trữ
**Lời thuyết trình (rút gọn):**
Về chi phí, kết quả khá ấn tượng cho phần cứng miễn phí: tổng thời gian train cả 4 scene chỉ 327,6 giây, khoảng 5,5 phút, tốc độ trung bình 85,5 iteration mỗi giây. VRAM đỉnh trung bình chỉ 0,88GB, cao nhất 1,16GB — rất thấp so với 14,56GB khả dụng trên T4, củng cố quan sát VRAM không phải nút thắt. Về lưu trữ, file point_cloud.ply trung bình nặng 171,6MB mỗi scene, khớp chính xác 248 byte trên mỗi Gaussian — kích thước một record 3DGS hoàn toàn chưa nén. Con số này là manh mối quan trọng cho phần giới hạn phép đo mà em trình bày ngay sau đây.

---
## Slide 37: Khả năng tái lập & giới hạn phép đo
**Lời thuyết trình (rút gọn):**
Trước khi sang phần Quan sát và Kết luận, nhóm nêu rõ giới hạn của phép đo này. Thứ nhất, đây chỉ là một lần chạy duy nhất, không cố định seed, không có error bar. Thứ hai, mẫu thử hẹp, chỉ 4 scene từ 2 dataset. Thứ ba và quan trọng nhất, ngân sách 7000 iteration chỉ bằng một phần ba so với lịch trình 30 nghìn iteration mà optimizer gốc thiết kế để chạy đủ. Thứ tư, quá trình này bỏ qua hoàn toàn bước prune cuối cùng final_prune_fastgs. Vì vậy, kết luận công bằng nhất: Bảng 1 nên được xem là baseline có thể tái lập của riêng repo này, không phải benchmark chính thức để so sánh trực tiếp với số liệu công bố của FastGS hay 3DGS gốc.
