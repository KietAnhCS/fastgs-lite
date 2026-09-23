# Kịch bản thuyết trình — Bot 05 (Slide 78–95)
## Phạm vi: ADC chuyên sâu P2-B (Footprint & Importance đa view), P3 (Công thức Score & Pruning)

---

## Slide 78: Ước lượng Monte Carlo cho Importance khi $V$ nhỏ
**Nội dung chính trên slide:** Khi huấn luyện thật chỉ lấy mẫu ngẫu nhiên V view (thay vì duyệt hết 200 view), Importance ước lượng hội tụ về giá trị thật khi V tăng, nhưng phương sai lớn với artefact hiếm gặp.

**Lời thuyết trình:**
Ở phần trước em đã trình bày công thức Importance tính trên toàn bộ view. Nhưng trong thực tế huấn luyện, ta không thể render hết hai trăm view mỗi lần densify vì quá tốn kém — code chỉ lấy mẫu ngẫu nhiên một số V view, ví dụ 1, 3, 10 hoặc 30. Câu hỏi đặt ra là: ước lượng từ mẫu nhỏ này có đáng tin không? Em đã mô phỏng ba loại Gaussian: loại lỗi nhất quán, loại artefact chỉ xuất hiện ở một góc hiếm, và loại biên giới. Kết quả trên đồ thị cho thấy khi V tăng, giá trị ước lượng trung bình hội tụ về đường chấm — tức giá trị tính trên toàn bộ hai trăm view — và độ lệch chuẩn giảm xấp xỉ theo một trên V. Điều đáng chú ý nhất là loại B, artefact hiếm gặp, có phương sai ước lượng lớn nhất, nghĩa là với V nhỏ như mười view trong code thật, ta dễ bỏ sót hoặc đánh giá sai loại lỗi hiếm này. Đây là một đánh đổi giữa chi phí tính toán và độ chính xác mà nhóm cần lưu ý.

---
## Slide 79: 3DGS gốc và FastGS-lite: hai tín hiệu densify khác nhau
**Nội dung chính trên slide:** 3DGS gốc chỉ dùng gradient để densify; FastGS-lite thêm điều kiện AND với Importance đa view, loại bỏ Gaussian có gradient cao nhưng không liên quan lỗi ảnh hiện tại.

**Lời thuyết trình:**
Slide này là điểm mấu chốt để hiểu vì sao FastGS-lite nhanh hơn 3DGS gốc. Trong 3DGS nguyên bản, quyết định densify một Gaussian chỉ dựa vào một tiêu chí duy nhất: độ lớn gradient tích luỹ trung bình vượt ngưỡng tau bằng hai nhân mười mũ âm bốn. Nghĩa là bất kỳ Gaussian nào có gradient cao đều bị densify, kể cả khi nó không thực sự gây lỗi hiển thị ở thời điểm hiện tại. FastGS-lite thêm một điều kiện AND: ngoài gradient vượt ngưỡng, Gaussian đó còn phải có Importance đa view lớn hơn năm, tức thực sự nằm trong vùng lỗi ảnh được nhiều góc nhìn xác nhận. Trên quần thể mô phỏng bốn trăm Gaussian, mười view, có thể thấy rõ: các điểm đỏ dấu x là những Gaussian 3DGS gốc vẫn densify dù Importance thấp — đó là densify lãng phí. FastGS-lite lọc bỏ đúng nhóm này, làm giảm mạnh số Gaussian được densify mỗi vòng, và vì số lượng Gaussian ảnh hưởng luỹ thừa đến tốc độ tăng N qua nhiều vòng lặp, hiệu ứng tiết kiệm này nhân lên rất nhanh.

---
## Slide 80: Sơ đồ tensor: luồng dữ liệu bước 4–5
**Nội dung chính trên slide:** Sơ đồ chi tiết luồng tensor từ render, tính sai số, tạo mặt nạ lỗi, đến kernel CUDA đếm counts và tính Importance.

**Lời thuyết trình:**
Để không chỉ nói bằng công thức mà còn cho thấy code thật hoạt động thế nào, em trình bày sơ đồ tensor lấy từ file fast_utils.py và forward.cu. Bắt đầu từ render_fastgs cho ra ảnh ba kênh, kích thước H nhân W. Hàm get_loss tính sai số tuyệt đối trung bình theo kênh màu, rồi chuẩn hoá minmax để được bản đồ sai số chuẩn hoá. Từ đó, metric_map là mặt nạ nhị phân đánh dấu pixel nào có sai số vượt ngưỡng không phẩy một. Phần quan trọng nhất nằm ở kernel CUDA: với mỗi cặp pixel và Gaussian thoả các điều kiện render hợp lệ — power nhỏ hơn không, alpha đủ lớn, T còn đủ độ trong suốt — nếu pixel đó nằm trong vùng lỗi thì thực hiện atomicAdd để cộng vào bộ đếm metricCount của Gaussian đó. Cộng dồn qua mười view ta có full_metric_counts, rồi chia cho V và lấy floor để ra importance_score. Đây chính là cầu nối giữa lý thuyết toán học và cách CUDA thực thi song song trên hàng triệu cặp pixel-Gaussian.

---
## Slide 81: Hiệu ứng floor() và ngưỡng chặt ">5"
**Nội dung chính trên slide:** Do làm tròn xuống, cần tổng counts ít nhất 60 (bằng 6V) mới vượt ngưỡng Importance; cơ chế này ép tính nhất quán đa góc nhìn.

**Lời thuyết trình:**
Bây giờ em phân tích một chi tiết nhỏ nhưng ảnh hưởng lớn: hàm floor trong công thức Importance. Vì Importance là floor của tổng counts chia cho V, và ta so sánh chặt lớn hơn năm, nên với V bằng mười, tổng counts phải đạt ít nhất sáu mươi — tức sáu nhân V — mới thực sự vượt ngưỡng. Điều này tạo ra một vùng nguy hiểm: nếu tổng nằm trong khoảng từ năm mươi đến dưới sáu mươi, trung bình rơi vào khoảng năm đến dưới sáu, floor sẽ cho ra đúng năm, và Gaussian đó không được densify dù rất gần ngưỡng. Đáng chú ý hơn, nếu một Gaussian chỉ hữu hình ở k trong mười view, nó cần trung bình c lớn hơn hoặc bằng sáu mươi chia k pixel lỗi mỗi lần xuất hiện mới đạt ngưỡng — nghĩa là Gaussian hữu hình ở ít view bị dìm xuống dù lỗi cục bộ của nó rất nặng. Đây thực chất là một cơ chế ngầm ép tính nhất quán đa góc nhìn: lỗi thoáng qua chỉ ở một hai view không đủ để kích hoạt densify, tránh việc mô hình phản ứng thái quá với nhiễu cục bộ.

---
## Slide 82: Ví dụ kiểm định số trên cảnh đồ chơi 48×32
**Nội dung chính trên slide:** Ví dụ số cụ thể với ảnh nhỏ, 3 view, 4 Gaussian, minh hoạ counts và Importance tính ra bằng tay để kiểm chứng công thức.

**Lời thuyết trình:**
Để kiểm chứng công thức không chỉ đúng về lý thuyết mà còn đúng khi cài đặt, em dựng một cảnh đồ chơi rất nhỏ: ảnh bốn mươi tám nhân ba mươi hai pixel, ba view, bốn Gaussian đặt tên G1 đến G4. Cảnh này được cố tình thiết kế sao cho gần như toàn bộ ảnh đều sai, để dễ kiểm tra bằng tay. Số pixel lỗi mỗi view lần lượt là một nghìn bốn trăm lẻ chín, một nghìn không trăm bảy mươi bảy, và một nghìn không trăm bốn mươi hai. Vì hầu hết footprint của mỗi Gaussian đều nằm trong vùng lỗi, tỉ lệ counts trên tổng số pixel lỗi luôn lớn hơn không phẩy chín cho tất cả các Gaussian. Tổng counts qua ba view của mỗi Gaussian còn lớn hơn nhiều so với số pixel lỗi của một view đơn lẻ, vì một pixel lỗi có thể "tố cáo" nhiều Gaussian chồng lấp cùng lúc. Kết quả cuối cùng, Importance của G1 đến G4 xấp xỉ một nghìn không trăm bốn mươi bốn, một nghìn không mười bảy, một nghìn không sáu mươi sáu, chín trăm bảy mươi bốn — vượt xa ngưỡng năm, đúng như kỳ vọng của một cảnh sai toàn khung hình.

---
## Slide 83: Gaussian TO và NHỎ: cùng mặt nạ lỗi, Importance khác nhau
**Nội dung chính trên slide:** Gaussian kích thước lớn có footprint lớn hơn nên counts và Importance cao hơn một cách tự nhiên dù không được chuẩn hoá theo diện tích; gợi ý chính sách split/clone.

**Lời thuyết trình:**
Slide này chỉ ra một điểm quan trọng cho phần chính sách densify sau này. Em đặt ba Gaussian có độ lệch chuẩn khác nhau — nhỏ với s bằng không phẩy bảy, vừa với s bằng ba, to với s bằng sáu — lên cùng một mặt nạ lỗi giống hệt nhau. Vì counts đếm số pixel trong phần giao giữa footprint và mặt nạ lỗi, và diện tích footprint tỉ lệ xấp xỉ với s bình phương, nên Gaussian to có counts lớn hơn hẳn, kéo theo Importance cao hơn một cách tự nhiên — dù tỉ lệ lỗi cục bộ, tức counts chia cho diện tích mặt nạ, không hề cao hơn Gaussian nhỏ. Điểm mấu chốt là code hiện tại không hề chuẩn hoá theo diện tích, nên kích thước hình học của Gaussian ảnh hưởng trực tiếp đến quyết định densify. Điều này gợi ý một chính sách hợp lý cho phần 4 sắp tới: khi Gaussian to vượt ngưỡng, nên split — chia nhỏ ra; còn Gaussian nhỏ vượt ngưỡng thì nên clone — nhân bản, vì bản chất "quan trọng" của chúng khác nhau về quy mô không gian ảnh hưởng.

---
## Slide 84: Bước (6): tổng hợp sai số ảnh $E_{photo}$ qua nhiều view
**Nội dung chính trên slide:** Công thức E_photo kết hợp L1 và 1-SSIM với trọng số lambda=0.2, minh hoạ bằng ví dụ số 3 view.

**Lời thuyết trình:**
Sau khi đã hiểu Importance dùng cho densify, em chuyển sang bước sáu — tính điểm sai số ảnh tổng hợp E_photo, thành phần sẽ dẫn đến điểm Pruning ở phần sau. Với mỗi view v, code tính ra một cặp giá trị L1 và SSIM giữa ảnh render và ảnh ground truth. Công thức tổng hợp trong code là trung bình có trọng số: một trừ lambda nhân L1 cộng lambda nhân một trừ SSIM, với lambda mặc định là không phẩy hai. Đây không phải là lấy max qua các view, mà là một phép trộn tuyến tính giữa cường độ sai khác pixel và cấu trúc ảnh. Với ví dụ ba view trong tài liệu 3.md, L1 lần lượt là không phẩy một một một hai lăm, không phẩy không sáu, không phẩy không chín không sáu hai lăm; SSIM là không phẩy bảy hai, không phẩy tám tám, không phẩy tám. Kết quả E_photo cho ba view là không phẩy một bốn năm, không phẩy không bảy hai, không phẩy một một hai năm — view một, có SSIM thấp nhất, đóng góp lỗi lớn nhất. Giá trị này sẽ đóng vai trò trọng số ở bước bảy sắp tới.

---
## Slide 85: Vì sao cần SSIM: L1 không phân biệt được "mờ" và "nhiễu"
**Nội dung chính trên slide:** Thí nghiệm cho thấy hai loại lỗi khác nhau (blur và noise) có cùng L1 nhưng SSIM khác nhau, chứng minh sự cần thiết của việc kết hợp cả hai chỉ số.

**Lời thuyết trình:**
Câu hỏi tự nhiên là: tại sao không chỉ dùng L1 cho đơn giản mà phải cộng thêm SSIM? Em làm một thí nghiệm nhỏ để trả lời. Từ một ảnh ground truth tổng hợp, em tạo hai ảnh render: Render A bị làm mờ bằng Gaussian blur với sigma bằng một phẩy sáu, Render B bị thêm nhiễu được hiệu chỉnh sao cho có cùng giá trị L1 với A. Nếu chỉ nhìn L1, hai render này "lỗi như nhau". Nhưng khi tính SSIM — vốn dùng cửa sổ Gauss mười một nhân mười một để đánh giá tương quan cấu trúc cục bộ — hai giá trị lại khác hẳn nhau, vì làm mờ phá cạnh nhưng vẫn giữ được tương quan patch, còn nhiễu phá cả cường độ lẫn cấu trúc. Hệ quả là E_photo của A và B cũng khác nhau đáng kể dù L1 gần bằng nhau. Điều này chứng minh rằng nếu chỉ dùng L1, mô hình sẽ đánh giá sai mức độ nghiêm trọng của hai loại lỗi hoàn toàn khác bản chất. Đó là lý do E_photo cần kết hợp cả L1 để đo cường độ và một trừ SSIM để đo cấu trúc.

---
## Slide 86: $E_{photo}$ theo trọng số trộn $\lambda$
**Nội dung chính trên slide:** E_photo là hàm tuyến tính theo lambda; các đường của từng view có thể cắt nhau, cho thấy lambda ảnh hưởng đến việc view nào bị coi là tệ nhất.

**Lời thuyết trình:**
Tiếp theo, em phân tích độ nhạy của E_photo theo tham số trộn lambda. Vì công thức là tuyến tính theo lambda trong khoảng từ không đến một, khi lambda bằng không ta chỉ dùng L1 thuần, còn khi lambda bằng một ta chỉ dùng một trừ SSIM — tương tự vai trò của lambda_dssim trong hàm loss huấn luyện gốc. Code hiện tại chọn lambda bằng không phẩy hai, tức ưu tiên L1 nhưng vẫn giữ lại một phần tín hiệu cấu trúc. Điều thú vị trên đồ thị là các đường E_photo của từng view, khi vẽ theo lambda, có thể cắt nhau: một view có L1 thấp nhưng SSIM cũng thấp có thể vượt lên và trở thành view "tệ nhất" khi lambda tăng dần. Tỉ số giữa giá trị lớn nhất và nhỏ nhất qua các view cũng thay đổi rõ rệt giữa lambda bằng không và lambda bằng một. Điều này cho thấy lựa chọn lambda không chỉ là một siêu tham số vô hại, mà ảnh hưởng trực tiếp đến việc view nào được xem là đóng góp lỗi nhiều nhất — và do đó ảnh hưởng đến toàn bộ chuỗi tính điểm phía sau.

---
## Slide 87: Đóng góp của từng Gaussian vào điểm số tổng
**Nội dung chính trên slide:** Công thức raw_i = tổng counts nhân E_photo qua các view; heatmap cho thấy chỉ một vài Gaussian chiếm phần lớn tổng sai số.

**Lời thuyết trình:**
Bây giờ em ghép Importance và E_photo lại để tạo ra điểm thô raw, nền tảng của Pruning. Với mỗi Gaussian i và view v, ta đã có counts đếm số pixel lỗi mà Gaussian đó tham gia dựng nên. Đóng góp có trọng số của nó vào điểm thô là counts nhân với E_photo của view đó, và điểm thô tích luỹ raw_i là tổng của tích này qua toàn bộ V view. Lấy lại ví dụ trong 3.md: Gaussian G3 có counts bằng sáu ở cả ba view, nên raw của G3 bằng sáu nhân tổng ba giá trị E_photo, xấp xỉ một phẩy chín tám — cao nhất trong năm Gaussian. Ngược lại, G1 có counts bằng không ở cả ba view nên raw bằng không, không đóng góp gì vào sai số. Heatmap bên phải minh hoạ điều này rất trực quan: chỉ một vài hàng có màu đậm, nghĩa là chỉ một vài Gaussian thực sự chiếm phần lớn tổng sai số ảnh, còn phần lớn Gaussian còn lại gần như không đóng góp gì. Đây chính là cơ sở để bước tiếp theo chuẩn hoá raw thành điểm Pruning.

---
## Slide 88: Importance vs. Pruning: ranh giới không phải ngưỡng cứng
**Nội dung chính trên slide:** Importance và Pruning tương quan dương nhưng không phải hàm một-một; với cùng Importance, Pruning trải rộng do phụ thuộc view nào chứa lỗi.

**Lời thuyết trình:**
Đến đây có một câu hỏi quan trọng: Importance và Pruning có phải là hai cách nói cùng một thứ không? Câu trả lời là không hẳn. Importance là floor của trung bình counts qua các view — một số nguyên dùng để quyết định densify với ngưỡng lớn hơn năm. Pruning là minmax của raw — chuẩn hoá về khoảng không đến một qua toàn bộ N Gaussian, dùng cho quyết định prune cuối cùng với ngưỡng lớn hơn không phẩy chín. Trên tập mô phỏng ba trăm Gaussian, mười view, hai đại lượng này có tương quan hạng dương, thể hiện qua hệ số Spearman trên hình, nhưng rõ ràng không phải là hàm một-một của nhau. Nhìn vào đồ thị scatter, với cùng một giá trị Importance, các điểm Pruning trải rộng trên một khoảng khá lớn — bởi vì Pruning còn phụ thuộc vào việc lỗi xảy ra ở view nào, thông qua trọng số E_photo của view đó, trong khi Importance chỉ đơn thuần đếm số lượt xuất hiện. Kết luận rút ra là quyết định pruning có một vùng chuyển tiếp mờ quanh ranh giới, chứ không phải một phép cắt cứng theo Importance.

---
## Slide 89: Pruning $\neq 1 - $ Importance chuẩn hoá
**Nội dung chính trên slide:** Bốn kịch bản cùng tổng counts nhưng phân bố khác nhau qua view cho raw và Importance khác nhau, chứng minh Pruning giữ thông tin mà floor() của Importance làm mất.

**Lời thuyết trình:**
Slide này đào sâu thêm sự khác biệt giữa hai đại lượng bằng bốn kịch bản cụ thể, cùng tổng counts nhưng phân bố khác nhau qua ba view: kịch bản A là sáu, sáu, sáu; B là mười tám, không, không; C là không, mười tám, không; D là mười bảy, không, không. A, B, C đều có Importance bằng floor của mười tám chia ba, tức bằng sáu, giống hệt nhau. Nhưng điểm thô raw lại khác nhau rõ rệt: dồn lỗi vào view có E_photo cao, tức view một với giá trị không phẩy một bốn năm, cho raw lớn hơn hẳn so với dồn vào view tốt nhất là view hai. Kịch bản D có tổng thấp hơn một chút, mười bảy thay vì mười tám, nên Importance có thể giảm bậc dù raw vẫn gần với B — cho thấy phép làm tròn xuống của floor làm mất thông tin mà Pruning, không làm tròn và có trọng số theo view, vẫn giữ được. Trên ba trăm Gaussian mô phỏng, ứng với mỗi mức Importance cố định, Pruning trải trên cả một khoảng giá trị. Vì vậy xác suất bị pruning tăng dần khi Importance giảm, nhưng đây là một quan hệ mang tính thống kê, không phải phép cắt tất định.

---
## Slide 90: Tổng hợp: luồng dữ liệu 7 bước tính Importance / Pruning
**Nội dung chính trên slide:** Sơ đồ toàn cảnh 7 bước từ render đến Importance và Pruning, cùng độ phức tạp tính toán của từng bước.

**Lời thuyết trình:**
Sau khi đã phân tích chi tiết từng thành phần, em tổng hợp lại toàn bộ quy trình bảy bước trong một sơ đồ duy nhất. Các bước một đến bốn thực hiện trên từng view riêng lẻ: render ảnh, tính sai số trung bình theo kênh, chuẩn hoá minmax để ra sai số chuẩn hoá, rồi tạo mặt nạ lỗi nhị phân, sau đó đổ về từng Gaussian thành counts của view đó. Bước sáu tính E_photo cho mỗi view và tích luỹ hai đại lượng song song: full_counts cộng dồn counts, và full_score cộng dồn tích của E_photo với counts. Bước năm lấy full_counts chia V rồi floor để ra Importance, dùng cho densify khi lớn hơn năm. Bước bảy lấy minmax của full_score để ra Pruning, từ đó suy ra trọng số lấy mẫu w_i, và prune khi Pruning lớn hơn không phẩy chín. Về độ phức tạp, các bước trên ảnh có chi phí O của H nhân W mỗi view, bước đổ về Gaussian có chi phí theo tổng số pixel lỗi qua các view, còn bước năm và bảy chỉ tốn O của N. Sơ đồ này sẽ là nền cho các slide phân tích độ nhạy và chi phí tiếp theo.

---
## Slide 91: Độ nhạy tham số $\lambda$: ví dụ 5 Gaussian (3.md)
**Nội dung chính trên slide:** Đổi lambda chỉ đổi giá trị Pruning chứ không đổi thứ hạng; nhưng bỏ hoàn toàn trọng số E_photo có thể làm đảo ngược thứ hạng.

**Lời thuyết trình:**
Câu hỏi tiếp theo là: công thức này có ổn định khi thay đổi tham số lambda không? Em quét lambda trong khoảng từ không đến một và tính lại Pruning cho năm Gaussian trong ví dụ 3.md. Kết quả khá bất ngờ: đổi lambda chỉ làm thay đổi giá trị số của Pruning, nhưng không hề làm đổi thứ hạng giữa các Gaussian — các đường trên đồ thị không cắt nhau trong toàn bộ khoảng lambda. Tuy nhiên, nếu bỏ hoàn toàn trọng số theo view, tức coi E_v luôn bằng một và chỉ dùng tổng counts thô, thứ hạng có thể bị đảo ngược thực sự. Ví dụ minh hoạ: so sánh G5 với một Gaussian giả định G6 sao, có counts là không, sáu, không — tức lỗi tập trung đúng vào view tốt nhất — hai Gaussian này đổi chỗ thứ hạng cho nhau khi bỏ trọng số. Ngược lại, nếu chỉ đổi cách chuẩn hoá, ví dụ chia cho max thay vì minmax, hay dùng z-score, thứ hạng vẫn giữ nguyên vì đó là các phép biến đổi đơn điệu. Kết luận: công thức ổn định với lambda, nhưng nhạy cảm thực sự với việc có hay không có trọng số E_photo theo view.

---
## Slide 92: Kiểm tra tính robust ở quy mô lớn: 300 Gaussian mô phỏng
**Nội dung chính trên slide:** Mô phỏng quy mô lớn xác nhận: bỏ trọng số E_photo gây đảo hạng nhiều nhất (37.3%), trong khi thay đổi lambda chỉ gây đảo hạng rất nhỏ.

**Lời thuyết trình:**
Để kiểm chứng kết luận ở slide trước không chỉ đúng trên năm Gaussian mà còn đúng ở quy mô lớn, em mô phỏng ba trăm Gaussian với mười view và so sánh thứ hạng Pruning gốc, dùng lambda bằng không phẩy hai, với các biến thể khác. Khi lambda bằng không, hoàn toàn không có Gaussian nào đổi hạng, và tập Pruning lớn hơn không phẩy chín vẫn giữ nguyên một trăm phần trăm. Khi lambda bằng một, chỉ hai phẩy ba phần trăm đổi hạng, tập đó vẫn giữ trọn vẹn. Nhưng khi bỏ hoàn toàn trọng số, coi E_v luôn bằng một, có tới ba mươi bảy phẩy ba phần trăm Gaussian đổi hạng — con số lớn nhất trong tất cả các biến thể — và tập Pruning lớn hơn không phẩy chín cũng giảm nhẹ còn chín mươi chín phẩy ba phần trăm. Một thí nghiệm khác, thay E_v bằng thứ hạng của chính nó thay vì giá trị thô, chỉ gây ba phẩy bảy phần trăm đổi hạng — cho thấy chính giá trị tuyệt đối của E_photo, với tỉ lệ max trên min lên tới năm phẩy sáu mốt lần giữa các view, mới là yếu tố quyết định. Đây là bằng chứng định lượng khẳng định lại kết luận ở slide trước trên quy mô lớn.

---
## Slide 93: Chi phí tính toán của bước scoring có là nút thắt mới?
**Nội dung chính trên slide:** Overhead của việc tính Importance/Pruning mỗi lần densify chỉ chiếm dưới 10% tổng chi phí huấn luyện, ngay cả với interval nhỏ nhất.

**Lời thuyết trình:**
Một câu hỏi thực tế mà bất kỳ ai triển khai cũng sẽ đặt ra: liệu bước tính điểm số phức tạp này có làm chậm huấn luyện đáng kể không? Em phân tích với thiết lập chuẩn: densify bắt đầu từ vòng năm trăm, kết thúc ở vòng mười lăm nghìn, tổng ba mươi nghìn vòng huấn luyện. Mỗi lần densify cần thêm mười view, mỗi view render hai lần — cho ground truth và cho ảnh mới — tương đương hai mươi lần forward bổ sung. Với interval bằng năm trăm, tức hai mươi tám lần densify, tổng thêm năm trăm sáu mươi forward, so với tổng chi phí huấn luyện chín mươi nghìn forward-equivalent, overhead chỉ khoảng không phẩy sáu hai phần trăm, cận trên nếu tính chặt hơn là một phẩy tám bảy phần trăm. Với interval hai trăm, overhead tăng lên một phẩy sáu phần trăm; với interval một trăm — mức nhỏ nhất thường dùng — overhead là ba phẩy hai phần trăm, cận trên bốn phẩy tám phần trăm. Có thể thấy overhead giảm tỉ lệ nghịch với interval, và ngay cả ở mức nhỏ nhất, chi phí này vẫn dưới mười phần trăm tổng chi phí huấn luyện — kết luận là bước scoring này không phải là nút thắt cổ chai mới.

---
## Slide 94: Từ điểm số đến trọng số lấy mẫu $w_i$ (bước chuẩn bị cho Phần 5)
**Nội dung chính trên slide:** Công thức w_i = 1/(epsilon + 1 - Pruning_i) tăng siêu tuyến tính khi Pruning tiến về 1, khiến một Gaussian có thể chi phối gần như toàn bộ xác suất lấy mẫu.

**Lời thuyết trình:**
Slide này là cầu nối sang phần 5 của bài, nơi Pruning được dùng để lấy mẫu. Công thức chuyển đổi từ Pruning sang trọng số w_i là một chia cho, mười mũ âm sáu cộng một trừ Pruning_i. Vì mẫu số tiến về không khi Pruning tiến về một, w_i tăng siêu tuyến tính và phân kỳ gần giá trị đó — số mười mũ âm sáu chỉ để tránh chia cho không. Lấy lại ví dụ 3.md: Pruning của năm Gaussian là không, không phẩy không năm bảy, một phẩy không, không phẩy sáu một, không phẩy ba sáu bảy; tương ứng w là một, một phẩy không sáu, một triệu, hai phẩy năm sáu, một phẩy năm tám. Khi chuẩn hoá thành xác suất lấy mẫu bằng cách chia cho tổng, G3 — Gaussian có Pruning bằng một — chiếm tới chín mươi chín phẩy chín chín chín chín phần trăm xác suất được chọn ở lượt rút mẫu đầu tiên, các Gaussian còn lại gần như không bao giờ được chọn. Cảnh đồ chơi cũng cho kết quả tương tự, luôn có một Gaussian đạt Pruning bằng một khiến w tiến tới vô cực. Đây là hiện tượng cần lưu ý khi thiết kế cơ chế lấy mẫu dựa trên điểm số này.

---
## Slide 95: Outlier trong chính điểm SCORE tổng hợp — lặp lại vấn đề từ Phần 1
**Nội dung chính trên slide:** Một Gaussian outlier với raw score cực lớn kéo méo toàn bộ phân bố Pruning qua phép minmax, làm ngưỡng tuyệt đối và trọng số lấy mẫu mất ổn định.

**Lời thuyết trình:**
Slide cuối cùng của phần em phụ trách quay lại một vấn đề đã nhắc ở phần một: phép chuẩn hoá minmax rất nhạy với outlier ở cả hai đầu. Ở đây, nó áp dụng cho raw_score trước khi lấy minmax thành Pruning. Em làm thí nghiệm trên ba trăm Gaussian mô phỏng: nhân giá trị raw lớn nhất lên gấp ba lần, mô phỏng tình huống có một Gaussian sai rất nặng xuất hiện thêm. Trước khi thêm outlier, có một Gaussian đạt Pruning lớn hơn không phẩy chín, và mười ba Gaussian đạt Pruning lớn hơn không phẩy năm. Sau khi thêm outlier, vẫn chỉ đúng một Gaussian — chính nó — vượt ngưỡng không phẩy chín, nhưng số Gaussian vượt ngưỡng không phẩy năm giảm mạnh từ mười ba xuống chỉ còn một. Toàn bộ quần thể còn lại bị kéo dồn về gần không. Thứ hạng tương đối vẫn giữ nguyên vì minmax là phép biến đổi tuyến tính đơn điệu, nhưng ngưỡng tuyệt đối và trọng số lấy mẫu w_i mà em vừa trình bày ở slide trước đều bị méo nghiêm trọng chỉ bởi một Gaussian duy nhất. Đây là điểm yếu cần cân nhắc khi thiết kế cơ chế chuẩn hoá điểm số cho hệ thống thực tế. Đến đây em xin kết thúc phần trình bày của mình, xin chuyển cho phần tiếp theo về chính sách densify.

