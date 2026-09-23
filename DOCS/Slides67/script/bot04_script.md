# Kịch bản thuyết trình — Bot 04 (Slide 60–77)
## Phạm vi: ADC chuyên sâu P1 (Sai số ảnh & tín hiệu pixel), P2-A (Footprint & Importance đa view)

---

## Slide 60: Lịch chạy ADC trong vòng lặp huấn luyện (0 → 30000)
**Nội dung chính trên slide:** Timeline 4 dải mốc của ADC trong 30000 iteration: tích luỹ gradient, densify+prune, reset opacity, final prune (đặc thù FastGS).

**Lời thuyết trình:**
Sau khi đã tìm hiểu tổng quan về Adaptive Density Control, bây giờ em sẽ đi sâu vào phần cơ chế thực sự bên trong nó, bắt đầu bằng lịch trình chạy. Điều đầu tiên cần nhấn mạnh là ADC không chạy liên tục ở mọi iteration, mà được lồng ghép vào vòng lặp huấn luyện 30000 bước theo các mốc cấu hình sẵn. Cụ thể có bốn dải hoạt động: thứ nhất, tích luỹ thống kê gradient chạy suốt từ đầu đến iteration 15000; thứ hai, densify và prune lặp lại mỗi 500 vòng, tổng cộng 28 lần trong giai đoạn từ 500 đến 15000; thứ ba, reset opacity mỗi 3000 vòng, diễn ra 4 lần; và cuối cùng, điểm đặc trưng của FastGS là final prune, chạy 4 lần trong nửa sau quá trình huấn luyện, từ 15000 đến 30000, để dọn dẹp những Gaussian kém chất lượng còn sót lại. Sau 30000 vòng thì không còn thao tác cắt tỉa nào nữa. Tiếp theo, em sẽ cho thấy lịch trình này ảnh hưởng thế nào đến số lượng Gaussian theo thời gian.

---

## Slide 61: Số lượng Gaussian N(t) thay đổi ra sao qua các giai đoạn
**Nội dung chính trên slide:** N tăng theo công thức nhân N←N·(1+r_spawn)(1−r_prune) mỗi lần densify; so sánh 3DGS gốc và FastGS-lite.

**Lời thuyết trình:**
Một điểm rất quan trọng mà nhiều người dễ bỏ qua là: số lượng Gaussian không tăng theo kiểu cộng dồn tuyến tính, mà tăng theo phép nhân. Sau mỗi lần densify, N được cập nhật bằng N nhân với một hệ số sinh thêm trừ đi phần bị cắt tỉa. Vì bước này lặp lại tới 28 lần trong giai đoạn densify, chỉ một chênh lệch nhỏ ở tỉ lệ sinh Gaussian cũng sẽ bị khuếch đại theo cấp số nhân, ảnh hưởng rất lớn đến số Gaussian cuối cùng. So sánh hai phương pháp: 3DGS gốc có tỉ lệ sinh khoảng 8.5%, không có bước final prune, nên N tăng nhanh rồi giữ nguyên gần như mãi mãi. Trong khi đó FastGS-lite chọn lọc hơn nhiều, chỉ sinh thêm Gaussian khi điểm Importance vượt ngưỡng 5, tỉ lệ sinh thấp hơn, khoảng 4.5%, nhưng bù lại có thêm final prune loại bỏ khoảng 12% mỗi lần. Kết quả là đường cong N của FastGS-lite tăng nhanh đầu train, chững lại, rồi giảm theo bậc thang ở nửa sau. Tiếp theo em sẽ giải thích tín hiệu đầu vào cho quyết định densify hay prune đó — chính là sai số ảnh theo từng pixel.

---

## Slide 62: Bước (1): sai số ảnh theo từng pixel e_v(x)
**Nội dung chính trên slide:** Định nghĩa sai số pixel bằng L1 trung bình 3 kênh màu giữa ảnh render và ground truth; ví dụ minh hoạ trên lưới 4x4.

**Lời thuyết trình:**
Bây giờ mình bắt đầu quy trình tính điểm Importance từ bước đầu tiên: đo sai số ảnh ở mức từng pixel. Với mỗi view v, ta có ảnh render và ảnh ground truth, mỗi ảnh có 3 kênh màu RGB. Sai số tại một pixel được định nghĩa là trung bình cộng của trị tuyệt đối hiệu hai ảnh trên cả ba kênh — nói cách khác là một dạng L1 loss cục bộ theo từng điểm ảnh, chứ không phải một con số tổng cho cả bức ảnh. Trên ví dụ minh hoạ lưới 4 nhân 4 pixel, có thể thấy rõ sự tương phản: những pixel thuộc vùng nền phẳng, ít chi tiết chỉ có sai số khoảng 0.01 đến 0.02, trong khi những pixel ở vùng vật thể phức tạp có thể lên tới 0.25, thậm chí 0.40. Bản đồ sai số thô này chính là nguyên liệu đầu vào, nhưng nó chưa dùng được trực tiếp vì thang giá trị chưa thống nhất giữa các view — đó là lý do bước tiếp theo cần chuẩn hoá. Trước khi đến bước chuẩn hoá, em sẽ giải thích tại sao chọn chuẩn L1 chứ không phải L2.

---

## Slide 63: Vì sao dùng L1 thay vì L2 ở bước (1)
**Nội dung chính trên slide:** So sánh d và d² trên sai số nhỏ và lớn; L2 nhạy outlier hơn, L1 giữ tuyến tính.

**Lời thuyết trình:**
Một câu hỏi hợp lý là tại sao không dùng L2, tức bình phương sai số, như nhiều bài toán regression khác vẫn hay dùng. Ở đây có một sự đánh đổi rõ ràng. Với sai số nhỏ, ví dụ d bằng 0.05, thì L1 vẫn giữ nguyên giá trị 0.05, nhưng L2 lại cho ra 0.0025 — gần như bị bóp về 0, khiến những lệch màu nhỏ nhưng thực sự tồn tại bị bỏ qua. Ngược lại, với sai số lớn ở những vùng vật thể chuyển động hay chi tiết phức tạp, bình phương sẽ khuếch đại rất mạnh, khiến L2 cực kỳ nhạy với outlier — chỉ một vài pixel lỗi nặng có thể chi phối toàn bộ bản đồ sai số. Khi so sánh tỉ lệ giữa giá trị lớn nhất và giá trị trung vị trên cùng bộ pixel, L2 cho tỉ lệ này cao hơn hẳn L1, tức đuôi phân phối dài hơn. Vì bước tiếp theo còn phải chuẩn hoá theo min và max của cả ảnh — vốn đã rất nhạy với outlier — nên dùng L1 ngay từ đầu giúp bản đồ lỗi ổn định hơn, tránh bị khuếch đại sai số hai lần liên tiếp.

---

## Slide 64: Bước (2): chuẩn hoá min-max trên từng ảnh
**Nội dung chính trên slide:** Công thức min-max chuẩn hoá theo min/max của chính view đó; ba view khác nhau nhưng dùng chung ngưỡng τ_loss=0.1.

**Lời thuyết trình:**
Sau khi có bản đồ sai số thô, bước thứ hai là chuẩn hoá min-max, nhưng điều đặc biệt là chuẩn hoá này được thực hiện độc lập cho từng view, dựa trên chính giá trị nhỏ nhất và lớn nhất của sai số trong ảnh đó. Công thức đơn giản: lấy sai số trừ đi giá trị nhỏ nhất, chia cho khoảng cách giữa lớn nhất và nhỏ nhất, kết quả nằm trong đoạn 0 đến 1. Điều này rất quan trọng vì các view có thể khác nhau rất nhiều về chất lượng render: có view "tệ" với sai số dao động từ 0.01 đến 0.40, có view "tốt" chỉ từ 0.01 đến 0.16. Nếu không chuẩn hoá riêng, một ngưỡng cố định sẽ đánh giá bất công giữa các view. Sau khi chuẩn hoá, cả ba view có thể dùng chung một ngưỡng tương đối là 0.1 ở bước tiếp theo. Nhưng cũng chính vì ngưỡng tuyệt đối tương ứng lại khác nhau theo từng view — ví dụ view tệ là 0.049 còn view tốt chỉ 0.025 — nên phép chuẩn hoá này đảm bảo công bằng về mặt tương đối. Tuy nhiên, cách làm này cũng có một điểm yếu mà slide sau sẽ chỉ ra.

---

## Slide 65: Vấn đề của min-max: một outlier bóp méo toàn bộ thang đo
**Nội dung chính trên slide:** Một pixel outlier (e=0.90) kéo giãn mẫu số, làm các pixel lỗi thật bị nén giá trị chuẩn hoá xuống gần một nửa.

**Lời thuyết trình:**
Điểm yếu lớn nhất của chuẩn hoá min-max là nó chỉ phụ thuộc vào đúng hai con số: giá trị nhỏ nhất và lớn nhất của toàn ảnh, nên cực kỳ nhạy cảm với một điểm bất thường duy nhất. Em minh hoạ bằng cách giữ nguyên lưới 4 nhân 4 của view 1, nhưng đặt sai số của một pixel lên tới 0.90, giả lập một pixel lỗi hoặc một vật thể chuyển động đột ngột. Kết quả là mẫu số, tức khoảng cách max trừ min, tăng từ 0.39 lên gần gấp đôi là 0.89. Hệ quả trực tiếp là ngưỡng tuyệt đối tương đương cũng tăng gấp đôi, từ 0.049 lên 0.099. Điều này khiến các pixel lỗi thật sự khác trong ảnh, vốn không hề thay đổi bản chất sai số, lại bị "nén" giá trị chuẩn hoá xuống gần một nửa so với trước. Một số pixel như P15 vẫn còn vượt ngưỡng nhưng biên an toàn bị thu hẹp đáng kể, chỉ cần thêm vài outlier nữa là toàn bộ điểm Importance của các Gaussian khác trong ảnh sẽ bị méo theo. Đây chính là lý do cần xem xét kỹ hơn các lựa chọn chuẩn hoá thay thế, mà phần tiếp theo sẽ phân tích.

---

## Slide 66: Bước (2): vì sao chọn min-max thay vì z-score/percentile?
**Nội dung chính trên slide:** So sánh min-max, z-score (cùng thứ hạng affine) và percentile 5-95 (bền outlier hơn nhưng tốn chi phí); FastGS chọn min-max vì rẻ.

**Lời thuyết trình:**
Tiếp nối vấn đề outlier vừa nêu, câu hỏi tự nhiên là: tại sao FastGS không dùng một phương pháp chuẩn hoá bền vững hơn? Trên một bản đồ sai số mô phỏng kích thước 48 nhân 32, có nền thấp, hai vùng lỗi cục bộ, và một outlier đơn lẻ với giá trị 0.95, em so sánh ba cách. Z-score, tức trừ trung bình rồi chia độ lệch chuẩn, về bản chất là một phép biến đổi affine, nên cho ra đúng thứ hạng pixel giống hệt min-max — không giải quyết được vấn đề outlier. Percentile 5 đến 95, tức cắt bớt theo phân vị, thì bền với outlier hơn hẳn vì giá trị cực đoan bị giới hạn lại, nhưng đổi lại phải tính thêm một lần sắp xếp hoặc tính phân vị, tốn chi phí tính toán hơn. Trong bối cảnh mã nguồn FastGS, nơi phép chuẩn hoá này phải chạy hàng chục nghìn lần trong quá trình huấn luyện, nhóm tác giả chọn min-max vì chi phí cực rẻ — chỉ cần tìm min và max — và đủ tốt cho mục đích so sánh tương đối giữa các pixel trong cùng một view, chấp nhận đánh đổi độ nhạy với outlier để lấy tốc độ.

---

## Slide 67: Bước (3): mặt nạ nhị phân với τ_loss = 0.1
**Nội dung chính trên slide:** Mặt nạ lỗi cao m_v = 1 khi ẽ_v > τ_loss; quét τ trên nhiều mức để chọn 0.1 là điểm cân bằng.

**Lời thuyết trình:**
Sau khi có bản đồ sai số đã chuẩn hoá, bước thứ ba là biến nó thành một mặt nạ nhị phân: mỗi pixel chỉ nhận giá trị 0 hoặc 1, tuỳ vào việc sai số chuẩn hoá có vượt ngưỡng τ_loss bằng 0.1 hay không. Đây là bước quan trọng vì nó chuyển từ một bản đồ liên tục sang một quyết định rõ ràng: pixel này có phải "vùng lỗi cao" hay không. Để chọn ra ngưỡng 0.1, nhóm tác giả đã quét qua nhiều mức khác nhau — 0.05, 0.1, 0.2, 0.5 — trên cả ba view của ví dụ lưới nhỏ. Kết quả cho thấy số pixel được đánh dấu giảm dần khi ngưỡng tăng lên, điều này khá trực quan. Với τ_loss bằng 0.1, mặt nạ tách khá sạch giữa vùng nền và vùng lỗi thực sự — không quá thưa thớt như ngưỡng 0.5, cũng không quá dày đặc như ngưỡng 0.05 khiến gần như cả ảnh đều bị đánh dấu. Về mặt công thức, ngưỡng tương đối này tương đương với một ngưỡng tuyệt đối trên sai số gốc, được tính lại riêng cho từng view. Tiếp theo, em sẽ phân tích độ ổn định của lựa chọn ngưỡng này qua một đường cong định lượng.

---

## Slide 68: Đường cong % pixel bị đánh dấu theo τ
**Nội dung chính trên slide:** Trên lưới nhỏ, đường cong dạng bậc thang; τ=0.1 nằm trên bậc phẳng rộng, ổn định; trên ảnh lớn, các đường gần trùng sau chuẩn hoá.

**Lời thuyết trình:**
Để kiểm chứng lựa chọn ngưỡng 0.1 có thực sự ổn định hay không, em vẽ đường cong phần trăm pixel bị đánh dấu theo từng giá trị τ. Với ảnh nhỏ 4 nhân 4, vì chỉ có 16 pixel rời rạc nên đường cong có dạng bậc thang. Điều đáng chú ý là τ_loss bằng 0.1 rơi đúng vào một bậc phẳng khá rộng, nghĩa là nếu chọn 0.08 hay 0.12 thay vì đúng 0.1 thì kết quả mặt nạ gần như không đổi — đây là một tín hiệu tốt cho thấy lựa chọn ngưỡng này không quá nhạy cảm, không cần tinh chỉnh chính xác tuyệt đối. Ở một thử nghiệm khác trên ảnh mô phỏng lớn hơn, kích thước 48 nhân 32, với ba mức chất lượng ảnh khác nhau — sai số tối đa lần lượt khoảng 0.40, 0.16 và 0.05 — sau khi áp dụng chuẩn hoá min-max theo từng ảnh, các đường cong phần trăm pixel gần như trùng khít lên nhau. Điều này chứng minh rất rõ rằng: nhờ chuẩn hoá theo ảnh, ngưỡng τ_loss mang ý nghĩa tương đối, hoàn toàn không phụ thuộc vào chất lượng tuyệt đối của ảnh render — dù ảnh tốt hay tệ, cơ chế vẫn hoạt động nhất quán.

---

## Slide 69: Ngưỡng chuẩn hoá ⇔ ngưỡng tuyệt đối trên e gốc
**Nội dung chính trên slide:** Cùng τ_loss=0.1 nhưng ngưỡng tuyệt đối khác nhau theo từng view, tỉ lệ thuận với khoảng [min,max] của view đó.

**Lời thuyết trình:**
Slide này làm rõ thêm mối liên hệ giữa ngưỡng tương đối và ngưỡng tuyệt đối mà mình đã nhắc qua trước đó. Về mặt toán học, τ_loss bằng 0.1 trên sai số đã chuẩn hoá tương đương với một ngưỡng tuyệt đối trên sai số gốc, bằng giá trị nhỏ nhất cộng thêm 0.1 lần khoảng cách max trừ min của view đó. Với ba view ví dụ có khoảng giá trị lần lượt là 0.01 đến 0.40, 0.01 đến 0.16, và 0.01 đến 0.24, ngưỡng tuyệt đối tương đương tính ra sẽ là khoảng 0.049, 0.025, và 0.033. Điều thú vị và cũng là bản chất của cơ chế này: dù dùng chung một τ_loss cố định, nhưng ngưỡng tuyệt đối lại khác nhau tuỳ theo từng view — view nào có khoảng sai số rộng hơn, tức là "tệ" hơn, thì ngưỡng tuyệt đối cũng cao hơn theo. Đây chính là hệ quả trực tiếp của việc chuẩn hoá theo từng ảnh riêng biệt trước khi so sánh với một ngưỡng cố định chung. Tiếp theo, em sẽ xem xét những trường hợp biên đặc biệt mà công thức chuẩn hoá này có thể gặp phải trong thực tế.

---

## Slide 70: Trường hợp biên: ảnh hoàn hảo, ảnh rất tệ, và chia cho 0
**Nội dung chính trên slide:** Ảnh hằng số gây NaN nhưng mặt nạ vẫn rỗng an toàn; ảnh gần hội tụ bị nhiễu nhỏ phóng đại; epsilon không đáng kể.

**Lời thuyết trình:**
Một công thức có mẫu số là hiệu giữa max và min chắc chắn phải xét đến trường hợp mẫu số bằng 0. Trường hợp đầu tiên: nếu toàn bộ ảnh có sai số bằng nhau tuyệt đối, ví dụ mọi pixel đều 0.03, thì max trừ min bằng 0, dẫn tới phép chia 0 trên 0 cho ra NaN. Điều thú vị là trong PyTorch và NumPy, phép so sánh NaN lớn hơn 0.1 luôn trả về False, nên mặt nạ kết quả tự động rỗng hoàn toàn một cách an toàn — không có pixel nào được đánh dấu, không xảy ra crash chương trình, và không cần thêm epsilon để xử lý riêng. Trường hợp thứ hai đáng chú ý hơn: khi ảnh gần hội tụ, sai số rất nhỏ, ví dụ chỉ trong khoảng 0.004 đến 0.006, thì phép chuẩn hoá min-max vẫn kéo giãn toàn bộ khoảng giá trị nhỏ này về đủ 0 đến 1 — nghĩa là nhiễu rất nhỏ cũng bị phóng đại thành "lỗi cao". Đây là cái giá phải trả của chuẩn hoá theo từng ảnh. Nếu có cộng thêm epsilon nhỏ vào mẫu số, dù code hiện tại không làm vậy, ảnh hưởng lên kết quả cũng dưới 1%, không đáng kể. Sau khi hiểu rõ ba bước lý thuyết, slide tiếp theo sẽ cho thấy luồng cài đặt thực tế trong PyTorch.

---

## Slide 71: Luồng PyTorch thực tế: từ render đến mặt nạ (get_loss)
**Nội dung chính trên slide:** Ba bước tensor: L1 loss (torch.abs, mean) → min-max normalize → ngưỡng hoá thành metric_map, hàm get_loss() trong fast_utils.py.

**Lời thuyết trình:**
Để kết thúc phần này, em tổng hợp lại toàn bộ ba bước lý thuyết vừa trình bày dưới dạng luồng cài đặt thực tế bằng PyTorch, đúng như trong hàm get_loss của mã nguồn FastGS. Bước một, tính L1 loss: lấy trị tuyệt đối của hiệu ảnh render và ground truth, có kích thước 3 kênh nhân chiều cao nhân chiều rộng, rồi lấy trung bình theo chiều kênh màu để ra một bản đồ sai số kích thước chiều cao nhân chiều rộng. Bước hai, chuẩn hoá min-max: lấy sai số trừ min chia cho max trừ min, với min và max được tính trên toàn ảnh, ra một tensor l1_loss_norm nằm trong đoạn 0 đến 1. Bước ba, ngưỡng hoá: so sánh tensor này với 0.1 rồi ép kiểu về số nguyên, cho ra metric_map chỉ gồm giá trị 0 và 1. Toàn bộ ba bước này nằm gọn trong khoảng 5 dòng code tại file utils/fast_utils.py. Tensor metric_map này chính là cầu nối quan trọng, được truyền tiếp vào hàm render_fastgs để bước 4 sử dụng — đó là tính footprint và tổng hợp importance qua nhiều view, nội dung mà phần tiếp theo sẽ đi sâu vào.

---

## Slide 72: Footprint thật của một Gaussian không phải là ellipse hình học
**Nội dung chính trên slide:** 4 định nghĩa footprint khác nhau (ellipse 3σ, sau cull compact-box, ngưỡng hiển thị, occlusion) — chỉ Ω_i^v hữu hình mới dùng để đếm lỗi.

**Lời thuyết trình:**
Bây giờ chuyển sang phần thứ hai: từ mặt nạ lỗi, làm sao biết Gaussian nào chịu trách nhiệm cho lỗi ở đâu? Câu trả lời nằm ở khái niệm footprint — vùng pixel mà một Gaussian thực sự ảnh hưởng tới trên màn hình. Nhưng footprint thật không đơn giản là hình ellipse lý thuyết như ta thường hình dung. Với mỗi pixel, ta tính khoảng cách Mahalanobis đến tâm Gaussian, từ đó suy ra độ mờ hiệu dụng bằng độ mờ gốc nhân với hàm mũ âm của khoảng cách đó, giới hạn tối đa 0.99. Slide này trình bày bốn lớp định nghĩa khác nhau: thứ nhất là ellipse hình học ba sigma, chỉ mang tính lý thuyết; thứ hai là vùng còn lại sau khi cắt bằng compact-box; thứ ba là chỉ giữ những pixel có độ mờ đủ lớn để mắt còn nhìn thấy được, tức trên ngưỡng hiển thị 1 trên 255; và cuối cùng, quan trọng nhất, là footprint hữu hình thực sự — lấy giao của cả ba điều kiện trên cộng thêm điều kiện chưa bị Gaussian khác che khuất hoàn toàn. Chỉ có định nghĩa thứ tư này mới được dùng để đếm lỗi trong bước tiếp theo. Vậy khi nhiều Gaussian chồng lấn nhau, việc quy trách nhiệm lỗi sẽ phức tạp ra sao?

---

## Slide 73: Nhiều Gaussian chồng lấn: một pixel lỗi "tố cáo" tất cả
**Nội dung chính trên slide:** Alpha-blending cập nhật T theo thứ tự gần-xa; điều kiện early-out T≥1e-4; một pixel có thể thuộc nhiều footprint.

**Lời thuyết trình:**
Trong thực tế, một cảnh 3D Gaussian Splatting có hàng trăm nghìn Gaussian chồng lấn lên nhau khi nhìn từ một góc camera, và pixel cuối cùng ta thấy là kết quả pha trộn của tất cả chúng theo thứ tự alpha-blending, từ gần đến xa. Ví dụ minh hoạ ba Gaussian A, B, C với độ mờ khác nhau: mỗi khi một Gaussian đóng góp vào một pixel, độ trong suốt còn lại T được cập nhật bằng T nhân với 1 trừ độ mờ tại điểm đó. Quá trình này có một điều kiện dừng sớm gọi là early-out: nếu T giảm xuống dưới 0.0001, các Gaussian phía sau coi như bị che hoàn toàn và không còn đóng góp gì nữa, nên không được tính vào footprint. Điểm mấu chốt ở đây là: không thể chỉ nhìn màu cuối cùng của một pixel để suy luận Gaussian nào gây ra lỗi, bởi vì nhiều Gaussian có thể cùng lúc đóng góp vào đúng pixel đó — một pixel có thể thuộc 0, 1, 2 hay cả 3 footprint cùng lúc. Vì vậy cần phải đếm và tổng hợp riêng biệt đóng góp của từng Gaussian. Slide tiếp theo sẽ cho thấy cách lấy giao giữa mặt nạ lỗi và các footprint này.

---

## Slide 74: Giao mặt nạ lỗi m_v với footprint Ω_i^v: chỉ phần giao mới được tính
**Nội dung chính trên slide:** counts_i^v = |Ω_i^v ∩ m_v|; ví dụ G4 footprint co lại qua các view do che khuất, G5 chỉ hữu hình ở view 1.

**Lời thuyết trình:**
Bây giờ mình kết hợp hai khái niệm đã học: mặt nạ lỗi cao và footprint của từng Gaussian. Với mỗi Gaussian i và mỗi view v, ta tính counts, chính là số pixel nằm trong cả footprint của Gaussian đó lẫn mặt nạ lỗi cao của view đó — nói đơn giản là đếm xem Gaussian này "chạm" vào bao nhiêu pixel lỗi. Trên ví dụ minh hoạ với 5 Gaussian và 3 view, có hai điểm rất đáng chú ý. Thứ nhất, với Gaussian G4, kích thước phần giao co lại dần qua các view — 6, rồi 4, rồi 3 pixel — bởi vì ở các góc nhìn khác nhau, Gaussian này bị các vật thể khác che khuất một phần, nên footprint hữu hình bị thu hẹp. Thứ hai, đáng chú ý hơn, là Gaussian G5 chỉ hữu hình ở view 1, còn ở view 2 và view 3 thì hoàn toàn bị che khuất, footprint rỗng, nên không đóng góp gì cả ở hai view đó. Điều này cho thấy tầm quan trọng của việc quan sát một Gaussian từ nhiều góc nhìn khác nhau trước khi kết luận nó có "gây lỗi" hay không. Tiếp theo, ta sẽ tổng hợp các counts này qua tất cả các view.

---

## Slide 75: Tổng hợp counts qua nhiều view: đầu vào cho Importance
**Nội dung chính trên slide:** Σ_i counts_i^v ≥ |m_v|; cài đặt CUDA cộng dồn atomic vào metricCount rồi lũy kế qua các view.

**Lời thuyết trình:**
Sau khi có counts cho từng Gaussian ở từng view riêng lẻ, bước tiếp theo là tổng hợp lại. Với ví dụ lưới nhỏ đã dùng, mỗi Gaussian i giờ có một vector counts gồm ba giá trị tương ứng ba view, đặt trên nền là kích thước footprint của nó ở view đó. Có một bất đẳng thức khá thú vị cần lưu ý: tổng counts của tất cả Gaussian trên một view luôn lớn hơn hoặc bằng số pixel lỗi thật của view đó, vì một pixel lỗi có thể đồng thời nằm trong footprint của nhiều Gaussian chồng lấn, nên bị đếm lặp lại nhiều lần — đây là điều hoàn toàn hợp lý và có chủ đích, vì mục tiêu là quy trách nhiệm cho tất cả các Gaussian liên quan, không phải chia đều lỗi. Về mặt cài đặt, mỗi lần render một view, kernel CUDA sẽ cộng dồn kiểu atomic vào một tensor tên metricCount có kích thước bằng số Gaussian, sau đó cộng lũy kế qua tất cả các view để có full_metric_counts. Vector counts tổng hợp này, trải qua toàn bộ V view, chính là đầu vào trực tiếp cho bước tính điểm Importance mà slide sau sẽ trình bày.

---

## Slide 76: Importance score: trung bình counts qua các view, làm tròn xuống
**Nội dung chính trên slide:** Importance_i = floor(mean_v counts_i^v); ngưỡng >5 để giữ lại/densify; so sánh mean vs max vs sum.

**Lời thuyết trình:**
Đây là công thức trung tâm của toàn bộ phần Importance, khớp chính xác với hàm compute_gaussian_score_fastgs trong mã nguồn. Điểm Importance của một Gaussian được tính bằng trung bình cộng counts của nó qua tất cả các view, sau đó làm tròn xuống — trong PyTorch chính là torch.div với chế độ làm tròn floor. Một Gaussian được xem là "quan trọng", tức được giữ lại hoặc ưu tiên densify, khi điểm Importance vượt quá 5. Trên ví dụ với 5 Gaussian và 3 view, chỉ có G3 đạt trung bình lớn hơn 5 và vượt ngưỡng; các Gaussian còn lại đều bị chặn, kể cả G5 dù nó có mean bằng 1.67. Một câu hỏi thú vị: nếu thay trung bình bằng giá trị lớn nhất hoặc tổng qua các view thì sao? Với G5, max bằng 5 và tổng cũng bằng 5, vẫn không vượt ngưỡng 5, nhưng thứ hạng tương đối giữa các Gaussian có thể thay đổi khác đi so với dùng mean. Việc dùng trung bình giúp đảm bảo một Gaussian phải nhất quán gây lỗi qua nhiều góc nhìn, chứ không chỉ nổi bật ở một view đơn lẻ, mới được coi là thực sự quan trọng. Slide cuối cùng của phần này sẽ kiểm chứng công thức này ở quy mô lớn hơn.

---

## Slide 77: Kiểm chứng ở quy mô lớn: mô phỏng 200 Gaussian, V=10 view
**Nội dung chính trên slide:** Mô hình mô phỏng footprint, tỉ lệ lỗi, xác suất hữu hình theo phân phối ngẫu nhiên; Importance ≈ |Ω_i|·ρ_i·p_vis.

**Lời thuyết trình:**
Để kết thúc phần footprint và Importance, em mở rộng kiểm chứng công thức trên một quy mô lớn hơn nhiều so với ví dụ lưới nhỏ ban đầu: mô phỏng 200 Gaussian quan sát qua 10 view. Mô hình mô phỏng gán cho mỗi Gaussian một kích thước footprint ngẫu nhiên trải rộng theo phân phối log-uniform, một tỉ lệ lỗi trong footprint theo phân phối Beta lệch về giá trị nhỏ — mô phỏng thực tế là đa số Gaussian ít gây lỗi — và một xác suất hữu hình ở mỗi view nằm trong khoảng 0.3 đến 1.0. Với mỗi view, nếu Gaussian hữu hình thì counts được sinh ra theo phân phối nhị thức dựa trên kích thước footprint và tỉ lệ lỗi, còn nếu không hữu hình thì counts bằng 0. Áp dụng đúng công thức Importance bằng trung bình làm tròn xuống như slide trước, kết quả phân bố Importance trên 200 Gaussian không bị lệch bất thường: một số vượt ngưỡng 5, một số bằng 0, phù hợp với trực giác rằng Importance tỉ lệ thuận với tích của kích thước footprint, tỉ lệ lỗi, và xác suất hữu hình. Đáng chú ý, một Gaussian nhỏ dù có tỉ lệ lỗi cao vẫn dễ bị chặn vì số pixel tuyệt đối quá thấp — đây chính là cách cơ chế ưu tiên những Gaussian có ảnh hưởng thực sự lớn trên toàn cảnh.

