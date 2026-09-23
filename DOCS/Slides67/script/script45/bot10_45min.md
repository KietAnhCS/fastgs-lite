# Kịch bản 45 phút — Bot 10 (Slide 38-41/41) — HẾT BÀI

---
## Slide 38: Quan sát 2: Nút thắt là RAM hệ thống, không phải VRAM
**Lời thuyết trình (rút gọn):**
Quan sát thứ hai: trên Colab T4, nút thắt thật sự là RAM hệ thống, không phải VRAM. VRAM chỉ dùng 8%, trong khi RAM hệ thống áp sát 91% giới hạn mềm. Hệ quả thực tiễn: khuyến nghị truyền thống "giảm resolution để tiết kiệm VRAM" thực ra không cần thiết, vì VRAM vốn dư thừa rất nhiều — muốn tối ưu nên tập trung quản lý RAM. Thú vị hơn, vì VRAM không phải nút thắt, train trực tiếp ở resolution gốc, tức resolution bằng 1, hoàn toàn khả thi — đồng thời loại bỏ luôn độ lệch resolution giữa train và eval mà nhóm gặp phải. Đây là hướng cải tiến khả thi cho các lần chạy sau. Tiếp theo, nhóm xin tổng kết lại toàn bộ pipeline render 3D Gaussian Splatting.

---
## Slide 39: Tổng kết pipeline render 3D Gaussian Splatting
**Lời thuyết trình (rút gọn):**
Nhóm xin tổng kết pipeline render bằng một sơ đồ khép kín, lặp lại qua từng iteration. Bắt đầu từ tập Gaussian 3D — vị trí, hiệp phương sai, độ mờ, hệ số spherical harmonics. Các Gaussian được chiếu lên ảnh qua phép chiếu phối cảnh xấp xỉ EWA, rasterize theo tile bằng rasterizer khả vi, rồi alpha-blend ra ảnh cuối. Ảnh này so với ảnh thật qua loss kết hợp L1 và D-SSIM, gradient lan truyền ngược để cập nhật tham số, đồng thời điều khiển Adaptive Density Control — thêm, tách, loại Gaussian — rồi vòng lặp quay lại điểm xuất phát. Toàn bộ cải tiến của FastGS chính là tăng tốc các bước trong chu trình này. Sau đây, nhóm tổng kết lại các cải tiến cụ thể và giới hạn còn lại.

---
## Slide 40: Tổng kết cải tiến FastGS & giới hạn còn lại
**Lời thuyết trình (rút gọn):**
FastGS cải tiến 3DGS gốc qua ba đòn bẩy: Multi-view score kết hợp Compact box giảm số Gaussian dư thừa mà không giảm chất lượng — lý do playroom và drjohnson đạt điểm cao dù dùng ít Gaussian hơn; Sparse Adam giảm chi phí mỗi iteration bằng cách chỉ cập nhật tham số thực sự liên quan. Bên cạnh đó, nhóm cũng nhận diện rõ giới hạn: dữ liệu Gaussian đầu ra chưa được nén; ngân sách 7000 iteration của fastgs-lite bỏ lỡ bước final_prune quan trọng; chưa có ablation study để cô lập đóng góp từng đòn bẩy; và bốn nhánh mở rộng của FastGS gốc — dynamic scenes, sparse view, surface reconstruction, SLAM — chưa được tích hợp, đây là hướng phát triển tiềm năng cho tương lai.

---
## Slide 41: Cảm ơn đã theo dõi! / Hỏi đáp (Q&A)
**Lời thuyết trình (rút gọn):**
Như vậy, nhóm em vừa trình bày xong toàn bộ nội dung đồ án: từ nền tảng toán học của 3D Gaussian Splatting, ba đòn bẩy cải tiến tốc độ của FastGS, quá trình đóng gói thành fastgs-lite để chạy được trên phần cứng miễn phí như Colab T4, cùng kết quả thực nghiệm và bài học rút ra — nổi bật nhất là RAM hệ thống mới là nút thắt thật sự chứ không phải VRAM, và việc thiếu bước prune cuối do giới hạn ngân sách iteration. Em xin chân thành cảm ơn quý thầy cô trong hội đồng cùng toàn thể các bạn đã dành thời gian lắng nghe phần trình bày của nhóm hôm nay. Rất mong nhận được câu hỏi, góp ý và nhận xét từ thầy cô để nhóm hoàn thiện đồ án tốt hơn. Nhóm em xin phép được lắng nghe và trả lời câu hỏi ạ. Xin cảm ơn.
