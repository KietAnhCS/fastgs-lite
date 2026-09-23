# Kịch bản 45 phút — Bot 07 (Slide 25-28/41)

---
## Slide 25: Top-k cứng vs. Multinomial: hai chiến lược pruning
**Lời thuyết trình (rút gọn):**
Em so sánh hai chiến lược pruning. Top-k cứng: sắp xếp ứng viên theo điểm số rồi xoá đúng số lượng thấp nhất — dễ đoán, luôn xoá đúng ngân sách, nhưng cắt theo ngưỡng cứng nên dễ xoá sạch cả một cụm điểm tương tự nhau cùng lúc. Multinomial: rút mẫu không hoàn lại theo trọng số tỉ lệ nghịch với (1 trừ P) — đa dạng hơn, tránh xoá đồng loạt một vùng, nhưng có nhiễu nên số lượng xoá thực tế có thể ít hơn ngân sách. Vì sao tránh "xoá sạch một cụm" lại quan trọng? Vì trên hệ Gaussian thật, một cụm điểm cao thường nằm ở cùng một vùng hình học — xoá sạch cùng lúc tạo lỗ hổng lớn khó phục hồi. Để thấy rõ cơ chế này hoạt động ra sao trên dữ liệu thật, mình chuyển sang một toy model 2D trực quan.

---
## Slide 26: Toy model 2D: mục tiêu và một Gaussian
**Lời thuyết trình (rút gọn):**
Từ đây em dùng một toy model 2D — mô hình thu nhỏ, đơn giản hoá để minh hoạ trực quan cơ chế ADC mà không cần dữ liệu 3D phức tạp. Ảnh mục tiêu chỉ 64x64 pixel: một hình tròn phẳng cộng một vùng chi tiết nhỏ (dải mảnh, chấm màu, ô vuông) để tạo cả vùng dễ và khó tái tạo. Mỗi Gaussian 2D là hàm mũ dạng toàn phương âm, tham số hoá bởi tâm mu, hai scale và một góc xoay; footprint giới hạn ở 3-sigma. Việc render dùng alpha-compositing theo thứ tự chỉ số — đúng nguyên lý blending đã học ở phần 3D, chỉ đơn giản hoá còn 2D cho dễ quan sát. Với nền tảng này, slide sau em sẽ so sánh trực tiếp kết quả có và không có ADC trên cùng toy model.

---
## Slide 27: So sánh trực tiếp: không ADC vs có ADC
**Lời thuyết trình (rút gọn):**
Đây là slide so sánh trực quan nhất của phần toy model, đặt ba kịch bản cạnh nhau: (1) không ADC, giữ N cố định bằng N0 ban đầu; (2) không ADC nhưng khởi tạo ngẫu nhiên ngay từ đầu với số lượng bằng đúng N cuối mà ADC đạt được; (3) có ADC đầy đủ, N tăng dần từ N0 lên N cuối theo đúng cơ chế densify. Điểm mấu chốt: giữa kịch bản 2 và 3, dù CÙNG số lượng Gaussian, kịch bản 2 với vị trí ngẫu nhiên vẫn không khớp tự nhiên vào vùng chi tiết cần thiết, nên chất lượng kém hơn. Kết luận cốt lõi: ADC không chỉ tăng số lượng N, mà quan trọng hơn là đặt đúng Gaussian vào đúng chỗ, có định hướng theo gradient của loss. Từ kết quả trên toy model này, slide tiếp theo sẽ cho thấy kết quả tương tự nhưng đầy đủ trên ba kịch bản thật, bao gồm cả FastGS-lite.

---
## Slide 28: Kết quả render cuối cùng: 3 kịch bản đối chiếu
**Lời thuyết trình (rút gọn):**
Đây là slide chốt hạ của toàn bộ phần so sánh, đối chiếu ba kịch bản song song. Hàng một: ADC 3DGS gốc, N tăng dần lên mức cuối cùng, kèm loss L1. Hàng hai: ADC FastGS-lite — đây là KẾT QUẢ QUAN TRỌNG NHẤT của toàn bộ đồ án: loss L1 cuối gần như TƯƠNG ĐƯƠNG với 3DGS gốc, nhưng số lượng Gaussian ÍT HƠN ĐÁNG KỂ. Nói cách khác, cùng chất lượng render nhưng chi phí bộ nhớ và tính toán thấp hơn hẳn, nhờ tổng hợp của Importance đa view, prune multinomial, và final prune. Hàng ba: không dùng ADC, giữ N cố định bằng N0, không clone/split/prune — chất lượng kém hơn rõ rệt, chứng minh ADC là thành phần không thể thiếu. Mỗi hàng gồm render cuối, bản đồ sai số, và ellipse 1.5-sigma. Sau đây chúng ta sẽ đi sâu vào từng cải tiến cụ thể để hiểu vì sao chúng hoạt động hiệu quả đến vậy.
