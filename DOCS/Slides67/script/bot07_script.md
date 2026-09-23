# Kịch bản thuyết trình — Bot 07 (Slide 117–135)
## Phạm vi: ADC chuyên sâu P5-B (Trọng số, Resample & Opacity Reset), P6 (Toy Model 2D: Có ADC vs Không ADC)

---

## Slide 117: Lịch chạy riêng của `final_prune_fastgs`
**Nội dung chính trên slide:** Timeline 30.000 vòng chia hai vùng theo mốc densify_until_iter = 15000; final prune chỉ chạy đúng 4 lần tại 18k/21k/24k/27k.

**Lời thuyết trình:**
Bây giờ em sẽ nói về lịch chạy riêng của hàm final_prune_fastgs. Toàn bộ 30.000 vòng huấn luyện được chia làm hai vùng theo mốc densify_until_iter bằng 15.000. Ở vùng đầu, tức là trước 15.000, cứ mỗi 100 vòng thì densify và prune multinomial chạy một lần, tổng cộng 145 lần, còn reset opacity chạy 5 lần tại các mốc chia hết cho 3.000. Còn final prune thì chỉ được phép chạy khi thoả đồng thời hai điều kiện: iteration lớn hơn 15.000, nhỏ hơn 30.000, và chia hết cho 3.000. Với điều kiện chặt như vậy, final prune chỉ thực sự kích hoạt đúng 4 lần, tại các mốc 18.000, 21.000, 24.000 và 27.000. Đặc biệt, mốc 30.000 không chạy, vì bất đẳng thức dùng dấu nhỏ hơn chặt chứ không phải nhỏ hơn hoặc bằng — đây là lần huấn luyện cuối nên số lượng Gaussian N được giữ nguyên. Slide sau em sẽ đi sâu vào điều kiện xoá của final prune.

---
## Slide 118: Điều kiện xoá cuối: phép OR, không phải AND
**Nội dung chính trên slide:** Final prune xoá Gaussian nếu opacity quá thấp HOẶC pruning score quá cao — khác với AND ở prune giữa chừng; ví dụ 5 Gaussian minh hoạ, vùng OR trên mặt phẳng (α, s_p).

**Lời thuyết trình:**
Điểm đặc biệt của final prune nằm ở loại phép toán logic mà nó dùng. Một Gaussian thứ i sẽ bị xoá nếu thoả ít nhất một trong hai điều kiện: opacity alpha-i nhỏ hơn 0,1, hoặc điểm số pruning s-p-i lớn hơn 0,9. Đây là phép OR — chỉ cần một trong hai tiêu chí là đủ để loại, khác hẳn với prune giữa chừng vốn dùng AND chặt kết hợp với ngưỡng bán kính và scale. Slide có ví dụ minh hoạ 5 Gaussian: G3 bị xoá vì điểm pruning bằng 1,000, vượt ngưỡng 0,9; G4 bị xoá vì opacity chỉ 0,07, dưới ngưỡng 0,1; còn G1, G2, G5 sống sót. Hình bên phải biểu diễn trực quan trên mặt phẳng hai trục alpha và s-p: có một dải màu đỏ ứng với alpha nhỏ hơn 0,1, một dải màu cam ứng với s-p lớn hơn 0,9, và Gaussian nào rơi vào một trong hai dải đó là bị loại. Ngoài ra, final prune không lấy mẫu ngẫu nhiên kiểu multinomial như prune giữa chừng, mà xoá thẳng toàn bộ tập thoả điều kiện. Tiếp theo, ta xem điều này ảnh hưởng thế nào đến đồ thị N theo thời gian.

---
## Slide 119: N(t) dạng bậc thang trong giai đoạn 15000–30000
**Nội dung chính trên slide:** Sau 15000, densify dừng, N chỉ giữ nguyên hoặc giảm theo bậc thang tại 4 mốc final prune; minh hoạ mức giảm 12%/8%/5%/3%.

**Lời thuyết trình:**
Hệ quả trực tiếp của cơ chế final prune là hình dạng đồ thị N theo thời gian thay đổi hẳn sau mốc 15.000. Vì densify không còn chạy nữa, N chỉ có thể giữ nguyên hoặc giảm, không bao giờ tăng thêm. Vì thế đồ thị N trong giai đoạn này có dạng bậc thang: những đoạn phẳng nằm ngang xen giữa các bước giảm đột ngột đúng tại bốn mốc final prune là 18.000, 21.000, 24.000 và 27.000. Trong ví dụ minh hoạ với N ban đầu là 420.000, mức giảm tại bốn mốc lần lượt khoảng 12%, 8%, 5%, và 3% — giảm dần vì càng về sau càng ít Gaussian dư thừa để loại. Đến mốc 30.000, không còn lần giảm nào nữa nên đồ thị đi ngang cho đến hết. Nhìn tổng thể, ta thấy rõ hai chế độ tách biệt: trước 15.000 là "tăng-giảm xen kẽ" do densify và prune multinomial hoạt động song song, còn sau 15.000 là "chỉ giảm theo bậc thang" do chỉ còn final prune. Tiếp theo em sẽ so sánh tổng thể giữa 3DGS gốc và FastGS-lite.

---
## Slide 120: So sánh đa chiều: 3DGS vs FastGS-lite
**Nội dung chính trên slide:** Radar 6 tiêu chí định tính; FastGS-lite đạt điểm tối đa cả 6 trục nhưng đánh đổi chi phí render phụ; số liệu tham chiếu N trung bình và R_gauss.

**Lời thuyết trình:**
Slide này tổng kết lại toàn bộ so sánh giữa 3DGS gốc và FastGS-lite bằng một biểu đồ radar định tính trên 6 tiêu chí: số điều kiện densify, số điều kiện prune, mức độ ngẫu nhiên trong prune, có tỉa cuối hay không, chi phí render phụ mỗi lần densify, và khả năng kiểm soát N. Thang điểm tự đặt từ 0 đến 3. 3DGS gốc có điểm thấp ở hầu hết các trục: densify chỉ 2 điều kiện, prune chỉ 1, không có ngẫu nhiên, không có tỉa cuối, không tốn chi phí render phụ, nhưng đổi lại khả năng kiểm soát N cũng thấp. Ngược lại, FastGS-lite đạt điểm tối đa ở cả 6 trục — nhiều tiêu chí hơn, có ngẫu nhiên hoá multinomial, có tỉa cuối — nhưng phải trả giá bằng chi phí render phụ để tính các điểm số importance và pruning. Số liệu tham chiếu minh hoạ cho thấy N trung bình của 3DGS khoảng 933 nghìn so với chỉ 293 nghìn của FastGS-lite, còn tỉ lệ Gaussian giữ lại R-gauss là 1,0 so với 0,314. Kết luận: FastGS-lite đánh đổi chi phí tính điểm số để đạt N nhỏ hơn đáng kể và kiểm soát chặt hơn. Slide tiếp theo sẽ minh hoạ cụ thể đường cong N(t) của hai phương pháp.

---
## Slide 121: N(t): 3DGS gốc so với FastGS-lite trên cùng trục thời gian
**Nội dung chính trên slide:** Mô hình toán N_t = N_{t-1}(1+r_spawn)(1-r_prune); so sánh hệ số 3DGS vs FastGS-lite, FastGS-lite hội tụ N nhỏ hơn nhờ spawn thấp và có final prune.

**Lời thuyết trình:**
Để minh hoạ định lượng cho sự khác biệt vừa nói, slide này xây dựng một mô hình toán đơn giản cho N theo thời gian. Công thức là N tại thời điểm t bằng N tại thời điểm trước đó nhân với (1 cộng tỉ lệ sinh mới r-spawn) rồi nhân với (1 trừ tỉ lệ tỉa r-prune), áp dụng mỗi 100 vòng, cộng thêm hệ số giảm tại mỗi lần reset opacity trước mốc 15.000. Với 3DGS minh hoạ, r-spawn là 0,0227, r-prune chỉ 0,003, giảm reset 4%, và quan trọng là không có final prune, nên trung bình N hội tụ khoảng 933 nghìn. Với FastGS-lite, r-spawn thấp hơn, chỉ 0,0155, r-prune cao hơn một chút là 0,004, cũng giảm reset 4%, nhưng có thêm final prune tại bốn mốc 18k/21k/24k/27k với tỉ lệ giảm 12%, 8%, 5%, 3% — nên trung bình N chỉ khoảng 293 nghìn. Như vậy FastGS-lite hội tụ về N nhỏ hơn rõ rệt nhờ hai yếu tố cộng hưởng: tốc độ sinh mới thấp hơn ngay từ đầu, và có thêm giai đoạn tỉa cuối mà 3DGS hoàn toàn không có. Tiếp theo là vấn đề kỹ thuật khi N thay đổi: trạng thái của bộ tối ưu Adam.

---
## Slide 122: Cắt/ghép trạng thái Adam khi N thay đổi
**Nội dung chính trên slide:** Adam lưu (m,v) mỗi hàng; thêm hàng mới nhận (0,0) nhưng step dùng chung; xoá theo mặt nạ; ép opacity chỉ reset nhóm opacity; step Adam không bao giờ reset.

**Lời thuyết trình:**
Một chi tiết kỹ thuật quan trọng là khi N thay đổi liên tục, trạng thái nội bộ của bộ tối ưu Adam cũng phải thay đổi đồng bộ. Adam lưu hai tensor cho mỗi hàng, tức mỗi Gaussian: m là exp_avg và v là exp_avg_sq. Khi thêm Gaussian mới qua clone hoặc split, các hàng mới được nối vào, và Gaussian con nhận (m, v) bằng (0, 0) — nhưng biến step của Adam thì dùng chung toàn cục, nên hệ số hiệu chỉnh bias gần như bằng 1 ngay lập tức dù step đã rất lớn, tạo ra một sự bất cân xứng nhỏ. Khi xoá Gaussian qua prune, hệ thống giữ lại các hàng theo mặt nạ boolean, loại đúng (m, v) của Gaussian bị xoá mà không làm xáo trộn thứ tự các hàng còn lại. Còn khi ép opacity, chỉ nhóm tham số opacity bị thay đổi có (m, v) reset về (0, 0), các nhóm khác như xyz, hệ số cầu điều hoà, scale, rotation vẫn giữ nguyên lịch sử — áp dụng ngay sau densify và tại mỗi lần reset opacity. Điểm cần nhớ: biến step của Adam không bao giờ bị reset trong bất kỳ trường hợp nào. Sau đây là sơ đồ khối tổng thể tóm tắt toàn bộ ADC.

---
## Slide 123: Sơ đồ khối tổng thể của Adaptive Density Control
**Nội dung chính trên slide:** Vòng lặp khép kín: tích luỹ thống kê → tính điểm số → densify (AND) → prune multinomial → ép opacity → reset opacity → final prune (OR) → cập nhật tensor.

**Lời thuyết trình:**
Slide này tổng hợp lại toàn bộ chu trình ADC thành một sơ đồ khối khép kín, giúp mình nhìn toàn cảnh sau khi đã đi qua từng phần riêng lẻ. Mỗi vòng huấn luyện trước 15.000, hàm add_densification_stats cập nhật các thống kê max_radii2D, accum, denom. Cứ mỗi 100 vòng, hệ thống tính điểm số trên 10 view để ra Importance và Pruning score, rồi thực hiện DENSIFY theo điều kiện AND: clone khi gradient lớn hơn hoặc bằng 2 nhân 10 mũ trừ 4, split khi gradient tuyệt đối lớn hơn hoặc bằng 1,2 nhân 10 mũ trừ 3, và cả hai đều cần Importance lớn hơn 5. Tiếp theo là PRUNE MULTINOMIAL: tập ứng viên C gồm những Gaussian có opacity nhỏ hơn 0,005, hoặc bán kính 2D lớn hơn 20, hoặc scale lớn nhất vượt 0,1 lần extent; trọng số w tính bằng nghịch đảo của (10 mũ trừ 6 cộng 1 trừ Pruning), rồi lấy mẫu multinomial theo trọng số đó để xoá. Mỗi 3.000 vòng có RESET opacity, và trong khoảng 15.000 đến 30.000 có FINAL PRUNE theo phép OR như đã nói. Sau mỗi lần gọi, các tensor thống kê được reset về 0, riêng step Adam thì không. Tiếp theo là một trường hợp thực tế: thứ tự thực thi trong code khác với lý thuyết.

---
## Slide 124: Thứ tự thực thi thật trong code (khác thứ tự lý thuyết)
**Nội dung chính trên slide:** Thứ tự thật: clone → split → prune multinomial (trọng số theo chỉ số cũ); ví dụ 4 Gaussian split ra 8, kết quả N=4 thay vì công thức lý thuyết dự đoán N=6.

**Lời thuyết trình:**
Đây là một phát hiện thú vị khi em đọc kỹ code thật của densify_and_prune_fastgs. Về mặt lý thuyết mình hay hình dung densify và prune là hai bước tách biệt, nhưng thứ tự thật trong code là clone, rồi split, rồi mới đến prune multinomial — và điều đặc biệt là prune chạy trên quần thể Gaussian mới sau khi đã clone/split, nhưng trọng số lại vẫn được gán theo chỉ số cũ. Em có ví dụ kiểm định: một cảnh có 4 Gaussian, ở thời điểm t lớn hơn 3.000, cả 4 đều split thành 8 Gaussian con. Do max_radii2D của Gaussian con bằng 0 nhưng scale lớn nhất của con lại vượt ngưỡng 0,169, nên cả 8 con đều rơi vào tập ứng viên C, với budget cần xoá là 4. Vấn đề là mảng padded_importance chỉ có 4 vị trí đầu mang trọng số cũ của 4 Gaussian gốc, còn 4 vị trí sau — ứng với Gaussian con thứ hai — đều bằng 0. Vì thế phép lấy mẫu multinomial buộc phải chọn đúng 4 chỉ số đầu tiên, dẫn đến kết quả N cuối cùng là 4, khác với công thức lý thuyết ở chương 7.8 dự đoán N bằng 6. Trong thực tế tỉ lệ Gaussian bị split mỗi lần rất nhỏ nên sai lệch này không đáng kể, nhưng đây là điểm cần lưu ý khi debug. Đến đây em đã trình bày xong phần lý thuyết chuyên sâu của ADC. Tiếp theo, em sẽ chuyển sang một toy model 2D trực quan để minh hoạ toàn bộ cơ chế này.

---
## Slide 125: Toy model 2D: mục tiêu và một Gaussian
**Nội dung chính trên slide:** Ảnh mục tiêu 64x64 px gồm hình tròn phẳng + vùng chi tiết; công thức Gaussian 2D, footprint 3-sigma, alpha-compositing theo độ sâu.

**Lời thuyết trình:**
Từ slide này em bắt đầu phần toy model 2D — một mô hình thu nhỏ, đơn giản hoá để minh hoạ trực quan cơ chế ADC mà không cần dữ liệu 3D phức tạp. Ảnh mục tiêu có kích thước chỉ 64 nhân 64 pixel, gồm một hình tròn phẳng màu đều, cộng thêm một vùng chi tiết nhỏ chứa một dải mảnh 2 pixel, 5 chấm màu xen kẽ, và một ô vuông nhỏ — mục đích là tạo ra cả vùng "dễ" và vùng "khó" tái tạo. Mỗi Gaussian 2D được biểu diễn bằng công thức hàm mũ của dạng toàn phương âm, với d là khoảng cách từ điểm x đến tâm mu, và ma trận hiệp phương sai Sigma được tham số hoá bởi hai giá trị scale s1, s2 và một góc xoay theta. Footprint, tức vùng mà Gaussian được phép ảnh hưởng, giới hạn ở 3-sigma — ngoài phạm vi đó chỉ còn màu nền. Việc render dùng alpha-compositing theo thứ tự chỉ số, giống hệt nguyên lý blending mà mình đã học ở phần 3D, chỉ khác là làm trên mặt phẳng 2D cho dễ hình dung. Slide minh hoạ ba hình: ảnh mục tiêu với lưới pixel, một Gaussian mẫu với các đường mức 1, 2, 3 sigma, và kết quả render của nó lên nền. Tiếp theo em sẽ chỉ ra trường hợp thiếu Gaussian.

---
## Slide 126: Trường hợp thiếu Gaussian (under-reconstruction)
**Nội dung chính trên slide:** Khởi tạo 6 Gaussian nhỏ không đủ phủ vùng chi tiết; sai số lớn tại vùng trống; gradient vị trí vượt ngưỡng clone τ_grad=4e-5 → tín hiệu CLONE.

**Lời thuyết trình:**
Ở slide này, em khởi tạo toy model chỉ với 6 Gaussian kích thước nhỏ, scale bằng 3 pixel, rải rác trên ảnh — không đủ số lượng để phủ hết vùng chi tiết gồm dải, chấm và ô vuông. Nhìn vào bản đồ sai số, tức trị tuyệt đối của hiệu giữa ảnh render và ảnh mục tiêu trung bình trên 3 kênh màu, ta thấy sai số lớn tập trung đúng tại những vùng không có Gaussian nào chạm tới — điều này rất trực quan vì đơn giản là "không có gì ở đó để vẽ". Khi tính gradient của loss theo vị trí tâm mu của từng Gaussian, độ lớn của gradient này vượt ngưỡng clone toy đặt ra là 4 nhân 10 mũ trừ 5. Đây chính là tín hiệu để cơ chế ADC quyết định thực hiện CLONE: ý nghĩa là Gaussian đang cố di chuyển về phía vùng thiếu nhưng không đủ nhanh để một mình phủ hết, nên cần thêm bản sao để chia sẻ nhiệm vụ. Hình minh hoạ gồm bốn phần: ảnh render với 6 Gaussian nhỏ, ảnh mục tiêu, bản đồ sai số, và các mũi tên biểu diễn hướng cùng độ lớn gradient tại từng Gaussian. Tiếp theo là trường hợp ngược lại: Gaussian quá to.

---
## Slide 127: Trường hợp Gaussian dư / quá to (over-reconstruction)
**Nội dung chính trên slide:** Một Gaussian lớn s=(13, 6.5) cố phủ cả vùng chi tiết; sai số mờ đều; chỉ số split dùng tổng trị tuyệt đối gradient theo pixel, ngưỡng τ_abs=4e-4.

**Lời thuyết trình:**
Ngược lại với slide trước, đây là trường hợp một Gaussian duy nhất nhưng kích thước rất lớn, scale bằng (13, 6.5) pixel, cố gắng một mình phủ toàn bộ vùng chi tiết gồm dải và 5 chấm màu cùng lúc. Kết quả là sai số bị mờ đều trên toàn vùng — sai cả ở những chỗ có chấm màu lẫn những khoảng trống giữa các chấm, vì một Gaussian to và trơn không thể tái tạo được các chi tiết cục bộ có tần số cao. Ngưỡng phân loại một Gaussian là "to" được đặt là scale lớn nhất vượt quá S_big bằng 3,5 pixel. Điểm mấu chốt kỹ thuật ở đây là: chỉ số dùng để quyết định split không phải là tổng có dấu của gradient trong footprint, mà là tổng trị tuyệt đối của gradient theo từng pixel, với ngưỡng tau_abs bằng 4 nhân 10 mũ trừ 4. Lý do là vì bên trong footprint của một Gaussian quá to, các mũi tên gradient tại các pixel khác nhau kéo về nhiều hướng trái ngược nhau — nếu cộng có dấu chúng sẽ triệt tiêu lẫn nhau và cho kết quả gần 0, che mất tín hiệu thật sự; còn cộng trị tuyệt đối thì vẫn lớn, phản ánh đúng mức độ "bối rối" của Gaussian. Bốn hình minh hoạ cho thấy rõ điều này, và kết luận là cần SPLIT. Sau đây em sẽ trình bày cơ chế CLONE trước.

---
## Slide 128: CLONE: sao chép nguyên trạng Gaussian thiếu
**Nội dung chính trên slide:** Clone tạo bản sao y hệt, Adam của bản mới khởi tạo lại (0,0); sau ~120 bước hai bản tách rời để chia vùng phủ; loss L1 giảm.

**Lời thuyết trình:**
Slide này minh hoạ chi tiết cơ chế CLONE bằng toy model. Trước khi clone, một Gaussian nhỏ có độ lớn gradient vị trí vượt ngưỡng, nghĩa là nó cần phủ một vùng rộng hơn kích thước hiện tại của nó cho phép. CLONE tạo ra một bản sao hoàn toàn y hệt — cùng vị trí mu, cùng hiệp phương sai Sigma, cùng màu c, cùng opacity alpha. Điểm quan trọng là trạng thái Adam của bản mới được khởi tạo lại về m bằng v bằng 0, trong khi bản gốc vẫn giữ nguyên toàn bộ lịch sử tối ưu của nó — chính sự bất đối xứng này khiến hai bản di chuyển với tốc độ khác nhau sau đó. Ngay sau khi clone, hai bản còn trùng khít hoàn toàn lên nhau, nhìn ảnh render gần như không đổi. Nhưng sau khoảng 120 bước Adam, quỹ đạo của tâm mu ở bản gốc và bản sao bắt đầu tách rời nhau, mỗi bản dần dịch chuyển để phủ một phần riêng của vùng cần thiết. Kết quả cuối cùng là loss L1 giảm rõ rệt, đúng vào lúc hai bản đã tách ra và chia nhau công việc. Hình minh hoạ bốn giai đoạn: trước clone, ngay sau clone, sau 120 bước, và biểu đồ khoảng cách giữa hai tâm theo thời gian. Tiếp theo là cơ chế SPLIT.

---
## Slide 129: SPLIT: tách Gaussian quá to thành các con nhỏ hơn
**Nội dung chính trên slide:** 2 con lấy mẫu quanh gốc, scale giảm còn s/1.6, gốc bị xoá; loss xấu tạm thời rồi cải thiện sau 150 bước.

**Lời thuyết trình:**
Còn đây là cơ chế SPLIT, áp dụng cho trường hợp Gaussian quá to ở hai slide trước. Với toy model này, mỗi lần split tạo ra n_child bằng 2 Gaussian con. Vị trí của mỗi con được lấy mẫu quanh vị trí gốc theo công thức mu-con bằng mu cộng R nhân epsilon, trong đó epsilon được lấy mẫu từ phân phối chuẩn với ma trận hiệp phương sai là diag của s bình phương — nghĩa là các con được rải ra theo đúng hình dạng elip của Gaussian gốc, chứ không phải ngẫu nhiên hoàn toàn. Scale của mỗi con giảm còn s chia 1,6 so với gốc, và quan trọng là Gaussian gốc bị xoá hoàn toàn sau khi split, không giữ lại. Về mặt loss, ngay sau khi split, chất lượng ảnh tạm thời xấu đi, vì hai Gaussian con nhỏ chưa kịp định vị đúng chỗ. Nhưng sau khoảng 150 bước Adam, mỗi con tự điều chỉnh lại vị trí và hình dạng, kết quả cuối cùng là L1 loss giảm, và các chi tiết cục bộ được khôi phục tốt hơn hẳn so với khi chỉ có một Gaussian to duy nhất. Hình minh hoạ cho thấy rõ ba giai đoạn: trước, ngay sau, và 150 bước sau split, kèm theo render và giá trị L1 tương ứng. Sau khi đã nắm rõ CLONE và SPLIT riêng lẻ, slide tiếp theo sẽ cho thấy toàn bộ vòng đời densify theo thời gian.

---
## Slide 130: Chuỗi thời gian: toàn bộ vòng đời densify trên toy model
**Nội dung chính trên slide:** N0=12 Gaussian khởi tạo kiểu SfM thu nhỏ, huấn luyện 1200 vòng, ADC chạy mỗi 100 vòng trong cửa sổ [100,900]; các mốc quan sát N và L1.

**Lời thuyết trình:**
Slide này tổng hợp toàn bộ vòng đời densify trên toy model thành một chuỗi thời gian liên tục. Khởi tạo mô phỏng kiểu SfM thu nhỏ: chỉ 12 Gaussian ban đầu, nhưng vị trí không phải ngẫu nhiên hoàn toàn mà được lấy mẫu ưu tiên tại những vùng có gradient ảnh mục tiêu lớn, tức là vùng chi tiết — mô phỏng cách point cloud SfM thật thường có mật độ cao hơn ở những vùng có kết cấu rõ. Toàn bộ quá trình huấn luyện chạy 1.200 vòng bằng Adam, còn ADC được kích hoạt mỗi K bằng 100 vòng, nhưng chỉ trong cửa sổ densify từ vòng 100 đến vòng 900 — giống hệt logic densify_until_iter ở mô hình thật. Slide ghi lại các mốc quan sát tại t bằng 0, 100, 300, 600 và 1000, mỗi mốc đều lưu lại giá trị N và L1 loss tương ứng. Qua các mốc này, có thể thấy các Gaussian tự dịch chuyển, co giãn hình dạng, và nhân đôi qua clone hoặc split, dần dần phủ kín ảnh mục tiêu, trong khi loss giảm liên tục và ổn định. Hình minh hoạ có hai hàng: hàng trên là ảnh render tại từng mốc kèm N và L1, hàng dưới là các ellipse 1,5-sigma của từng Gaussian chồng lên ảnh mục tiêu mờ, giúp thấy rõ hình dạng thay đổi ra sao. Tiếp theo, em sẽ minh hoạ khái niệm "lãnh thổ" của từng Gaussian.

---
## Slide 131: Chia vùng: mỗi Gaussian "phụ trách" một lãnh thổ
**Nội dung chính trên slide:** Mỗi pixel gán cho Gaussian đóng góp lớn nhất T_i*α_i*G_i; theo thời gian số lãnh thổ tăng và khớp sát hơn với biên vật thể.

**Lời thuyết trình:**
Đây là một cách trực quan khác để hiểu ADC: khái niệm "lãnh thổ" của từng Gaussian. Với mỗi pixel trên ảnh, em xác định Gaussian nào đóng góp lớn nhất vào giá trị màu tại pixel đó, dựa trên trọng số T-i nhân alpha-i nhân G-i — chính là đóng góp thực sự sau khi đã tính alpha-compositing, chứ không phải chỉ dựa vào khoảng cách hình học đơn thuần. Mỗi pixel sau đó được tô theo màu của Gaussian "thắng" tại vị trí đó; còn màu trắng biểu thị nền, tức là khi giá trị T-end, phần ánh sáng còn lại truyền qua tất cả Gaussian, lớn hơn mọi đóng góp riêng lẻ. Đường viền màu đen trên hình là biên vật thể mục tiêu, được xác định qua đường mức gradient, dùng để đối chiếu xem các lãnh thổ có khớp với hình dạng thật hay không. Theo thời gian huấn luyện, do có clone và split liên tục, số lượng lãnh thổ tăng lên, đồng thời từng lãnh thổ co lại và trở nên khớp sát hơn với đường biên vật thể cũng như các vùng chi tiết nhỏ. Đây chính là hình ảnh trực quan cho thấy ADC đang "chia để trị" không gian ảnh như thế nào. Slide kế tiếp sẽ đưa ra định lượng qua đường cong loss và N(t).

---
## Slide 132: Đường cong loss và số Gaussian N(t)
**Nội dung chính trên slide:** 3 kịch bản so sánh: có ADC, không ADC N=N0, không ADC N=N_cuối ngẫu nhiên; N(t) tăng bậc thang tại mốc ADC trong [100,900]; có ADC luôn đạt L1 thấp hơn.

**Lời thuyết trình:**
Slide này bắt đầu phần so sánh định lượng then chốt của toy model, đối chiếu ba kịch bản: thứ nhất là có ADC đầy đủ; thứ hai là không có ADC, giữ N cố định bằng N0 ban đầu suốt quá trình; thứ ba là không có ADC nhưng N cố định bằng đúng N cuối cùng mà ADC đạt được, chỉ khác là khởi tạo vị trí ngẫu nhiên ngay từ đầu thay vì tăng dần. Trên biểu đồ loss theo thang log, các vạch dọc đánh dấu đúng những mốc mà ADC được kích hoạt, cách nhau K_ADC bằng 100 vòng. Đồ thị N theo thời gian tăng theo dạng bậc thang, đúng tại các mốc ADC đó, và chỉ tăng trong cửa sổ densify từ vòng 100 đến vòng 900, sau đó giữ nguyên. Biểu đồ cột bên cạnh cho biết số lượng Gaussian bị clone, split, hoặc prune tại từng mốc ADC, cho thấy tỉ lệ giữa các cơ chế thay đổi ra sao qua thời gian — thường clone và split nhiều ở giai đoạn đầu, prune tăng dần về sau. Kết quả quan trọng nhất: kịch bản có ADC luôn đạt L1 loss thấp hơn rõ rệt so với việc giữ nguyên N0 ban đầu, trong suốt toàn bộ quá trình huấn luyện. Tiếp theo em sẽ phân tích mối quan hệ giữa kích thước Gaussian và mức độ chi tiết mà nó tái tạo.

---
## Slide 133: Kích thước Gaussian so với mức chi tiết tái tạo
**Nội dung chính trên slide:** Phân bố kích thước Gaussian đầu vs cuối; quan hệ ngược chiều giữa gradient chi tiết và kích thước; màu điểm = opacity.

**Lời thuyết trình:**
Slide này đào sâu vào mối quan hệ giữa kích thước Gaussian và mức độ chi tiết mà nó đảm nhiệm tái tạo. Đầu tiên là histogram phân bố kích thước, cụ thể là scale lớn nhất của mỗi Gaussian, so sánh giữa thời điểm t bằng 0 và lúc kết thúc huấn luyện — có thể thấy nhóm Gaussian cuối cùng tập trung nhiều hơn quanh và dưới ngưỡng toy S_big bằng 3,5 pixel, tức là ADC có xu hướng "thu nhỏ" quần thể Gaussian về kích thước phù hợp. Để đo độ chi tiết mà một Gaussian đang phủ, em tính gradient ảnh mục tiêu trung bình trong footprint của nó, tức là vùng mà q nhỏ hơn 9. Biểu đồ scatter cho thấy một quan hệ ngược chiều khá rõ rệt: những Gaussian nằm ở vùng có mức chi tiết cao thường có kích thước nhỏ hơn, trong khi những Gaussian nằm ở vùng phẳng, ví dụ bên trong hình tròn màu đều, lại có kích thước lớn hơn — điều này hoàn toàn hợp lý vì vùng phẳng không cần nhiều Gaussian nhỏ để biểu diễn. Màu của từng điểm trên scatter biểu diễn giá trị opacity alpha, và có thể thấy hầu hết Gaussian còn sống sót sau ADC đều có alpha cao, phản ánh chúng thực sự đóng góp có ý nghĩa vào ảnh cuối cùng, không phải Gaussian "chết". Tiếp theo là slide so sánh trực tiếp ba kịch bản cạnh nhau.

---
## Slide 134: So sánh trực tiếp: không ADC vs có ADC
**Nội dung chính trên slide:** 3 kịch bản (không ADC N0; không ADC ngẫu nhiên N_cuối; có ADC) đặt cạnh nhau 3x3; ADC không chỉ tăng N mà đặt đúng chỗ cần thiết.

**Lời thuyết trình:**
Đây là slide so sánh trực quan nhất trong toàn bộ phần toy model, đặt ba kịch bản cạnh nhau để mắt thường cũng nhận ra sự khác biệt. Kịch bản một: không có ADC, giữ nguyên N bằng N0 từ đầu đến cuối. Kịch bản hai: cũng không có ADC, nhưng khởi tạo ngẫu nhiên ngay từ đầu với số lượng Gaussian bằng đúng N cuối mà ADC đạt được — nghĩa là "cho trước" số lượng Gaussian tương đương nhưng không qua quá trình tăng trưởng có định hướng. Kịch bản ba: có ADC đầy đủ, N tăng dần từ N0 lên N cuối theo đúng cơ chế densify. Điểm mấu chốt cần nhấn mạnh là giữa kịch bản hai và ba: dù có cùng số lượng Gaussian, nhưng kịch bản hai với vị trí khởi tạo ngẫu nhiên lại không khớp tự nhiên vào những vùng chi tiết cần thiết, nên chất lượng tái tạo vẫn kém hơn. Từ đó rút ra kết luận cốt lõi của toàn bộ chương: ADC không chỉ đơn thuần là tăng số lượng Gaussian N, mà quan trọng hơn là đặt đúng Gaussian vào đúng chỗ cần thiết, một cách có định hướng theo gradient của loss. Hình minh hoạ có 3 hàng ứng với 3 kịch bản, nhân với 3 cột thể hiện các góc nhìn khác nhau. Slide cuối cùng của phần này sẽ nói về cơ chế PRUNE dựa trên opacity.

---
## Slide 135: PRUNE: loại bỏ Gaussian dư theo opacity
**Nội dung chính trên slide:** Tại t=550 reset opacity về min(α, 0.02); Gaussian không phục hồi bị prune theo ngưỡng ε_opa=0.03; ảnh gần như không đổi sau prune.

**Lời thuyết trình:**
Slide cuối cùng của phần toy model 2D nói về cơ chế PRUNE dựa trên opacity, khép lại toàn bộ chu trình ADC. Tại mốc t bằng 550, opacity của mọi Gaussian được reset theo công thức alpha nhận giá trị nhỏ nhất giữa alpha hiện tại và alpha_reset bằng 0,02 — nói cách khác, mọi Gaussian đều bị ép về gần như trong suốt, buộc chúng phải "chứng minh lại" sự cần thiết của mình thông qua quá trình tối ưu tiếp theo. Những Gaussian nào thực sự đóng góp vào việc giảm loss sẽ tự động được Adam đẩy opacity tăng trở lại; còn Gaussian nào không phục hồi được, tức là không giúp ích gì cho việc tái tạo ảnh, sẽ bị đánh dấu để prune tại mốc ADC kế tiếp, theo ngưỡng epsilon_opa bằng 0,03. Khi so sánh ảnh render ngay trước và ngay sau khi prune, sai lệch tối đa giữa hai ảnh rất nhỏ — gần như không nhận ra sự khác biệt bằng mắt thường, dù đã loại bỏ hẳn các Gaussian dư thừa. Điều này chứng minh đúng vai trò của bước prune trong ADC: mô hình trở nên gọn hơn, ít Gaussian hơn, nhưng chất lượng tái tạo được giữ nguyên gần như hoàn toàn — một bước dọn dẹp hiệu quả và an toàn. Đến đây em đã hoàn tất phần trình bày về ADC chuyên sâu và toy model minh hoạ 2D.

---
