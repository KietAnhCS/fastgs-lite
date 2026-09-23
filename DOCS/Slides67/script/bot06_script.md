# Kịch bản thuyết trình — Bot 06 (Slide 96–116)
## Phạm vi: ADC chuyên sâu P4 (Gradient Cancellation & Clone/Split), P5-A (Trọng số, Multinomial Resample & Opacity Reset)

---

## Slide 96: Triệt tiêu Gradient (Gradient Cancellation)
**Nội dung chính trên slide:** Hai tầng triệt tiêu gradient — trong một ảnh (cần lấy trị tuyệt đối theo trục u) và giữa nhiều view (cần lấy norm từng vector rồi mới cộng, thay vì cộng vector trực tiếp).

**Lời thuyết trình:**
Phần này em đi sâu vào một vấn đề kỹ thuật rất tinh vi trong bước tính gradient để quyết định densify, gọi là "triệt tiêu gradient". Có hai tầng triệt tiêu. Tầng một xảy ra ngay trong một ảnh: gradient vị trí theo trục u đổi dấu ở hai bên biên vật thể, nên nếu cộng có dấu thì hai phía gần như triệt tiêu nhau, phải lấy trị tuyệt đối mới giữ được tín hiệu. Tầng hai xảy ra giữa các view khác nhau: mỗi view cho một vector gradient hướng khác nhau, ví dụ bốn view với các góc và độ lớn khác nhau, nếu cộng trực tiếp các vector thì tổng chỉ còn khoảng 0.11 — gần như bằng không, sai hoàn toàn. Code gốc xử lý đúng bằng cách lấy norm của từng vector trước, rồi mới cộng dồn qua các iteration, cho ra accum bằng 3.95 và gradient trung bình 0.9875. Đây chính là lý do vì sao công thức chuẩn hoá phải "norm trước, cộng sau". Slide tiếp theo sẽ cho thấy quá trình tích luỹ này diễn ra như thế nào qua một chu kỳ densify.

---
## Slide 97: Đường cong tích luỹ: accum, denom, $\bar g$
**Nội dung chính trên slide:** Công thức tích luỹ accum/denom qua một chu kỳ 100 vòng lặp; ba Gaussian mô phỏng (A luôn hữu hình, B hữu hình 40%, C đã hội tụ) minh hoạ vai trò của denom.

**Lời thuyết trình:**
Tiếp theo, em minh hoạ cách gradient trung bình được tích luỹ qua một chu kỳ densification_interval bằng 100 vòng lặp. Mỗi Gaussian duy trì hai bộ đếm: accum là tổng norm gradient cộng dồn mỗi lần nó hữu hình trong một view, và denom là số lần nó thực sự hữu hình. Gradient trung bình chính là accum chia denom. Em mô phỏng ba trường hợp: Gaussian A luôn hữu hình với gradient trung bình khá thấp, Gaussian B chỉ hữu hình 40% số view nhưng gradient từng lần lại lớn, và Gaussian C đã hội tụ nên gradient rất nhỏ. Điểm thú vị là B có denom nhỏ hơn A nhưng gradient trung bình vẫn vượt ngưỡng, vì công thức chia theo số lần THẤY chứ không phải theo tổng số iteration đã trôi qua. Còn C thì gradient trung bình dưới ngưỡng nên không được densify. Điều này cho thấy tín hiệu densify phụ thuộc chặt vào denom, không chỉ vào accum. Sau khi có gradient trung bình đáng tin cậy, câu hỏi tiếp theo là: gradient thôi có đủ để quyết định densify không?

---
## Slide 98: Mặt phẳng AND: gradient truyền thống × Importance
**Nội dung chính trên slide:** FastGS-lite kết hợp điều kiện gradient truyền thống (3DGS gốc) với điều kiện Importance bằng phép AND, minh hoạ trên 5 Gaussian mẫu, chỉ G3 đạt cả hai điều kiện.

**Lời thuyết trình:**
Đây là điểm cải tiến cốt lõi của FastGS-lite so với 3DGS gốc. 3DGS gốc chỉ cần gradient vượt ngưỡng là đủ để clone hoặc split. FastGS-lite thêm một điều kiện thứ hai: điểm Importance, và nối hai điều kiện bằng phép AND logic. Em minh hoạ trên năm Gaussian mẫu đã dùng ở phần trước. Ở nhánh clone, G3 có gradient vượt ngưỡng và Importance bằng 6 lớn hơn 5, nên được clone. Nhưng G1 và G5 cũng có gradient vượt ngưỡng — nghĩa là 3DGS gốc sẽ clone cả hai — song Importance của chúng chỉ bằng 0 và 1, dưới ngưỡng 5, nên FastGS-lite chặn lại không cho densify. Ở nhánh split, G4 có gradient trị tuyệt đối vượt ngưỡng nhưng Importance chỉ bằng 3 nên cũng bị chặn. Kết quả cuối cùng: trong năm Gaussian, chỉ duy nhất G3 vượt qua cả hai điều kiện và được densify. Đây chính là cách FastGS-lite lọc bớt các Gaussian không thực sự quan trọng dù gradient của chúng có vẻ lớn. Slide sau sẽ mở rộng phép AND này ra quy mô năm trăm Gaussian.

---
## Slide 99: Vùng ứng viên clone trên quy mô 500 Gaussian
**Nội dung chính trên slide:** Mô phỏng 500 Gaussian trên mặt phẳng (gradient, kích thước); phép AND với điều kiện Importance chỉ giữ lại khoảng một nửa số ứng viên so với chỉ dùng gradient.

**Lời thuyết trình:**
Để thấy rõ hơn tác động của phép AND ở quy mô lớn, em mô phỏng 500 Gaussian với scene extent khoảng 1.69, từ đó suy ra ngưỡng kích thước để phân biệt nhánh clone và split. Điều kiện clone của 3DGS gốc chỉ cần Gaussian nhỏ và gradient đạt ngưỡng. FastGS-lite thêm điều kiện Importance lớn hơn 5. Trên mặt phẳng gradient theo trục hoành và kích thước lớn nhất theo trục tung, vùng dưới ngưỡng kích thước và bên phải ngưỡng gradient là toàn bộ ứng viên clone theo 3DGS gốc. Nhưng khi lấy Importance ngẫu nhiên trong khoảng từ 0 đến 11 rồi áp phép AND, chỉ khoảng một nửa số ứng viên đó còn được giữ lại — nửa còn lại bị đánh dấu là "3DGS sẽ densify nhưng FastGS-lite chặn". Điều này cho thấy rõ ràng bằng số liệu là phép lọc Importance cắt giảm đáng kể khối lượng tính toán densify mà không cần thay đổi ngưỡng gradient gốc. Trước khi đi vào sơ đồ quyết định đầy đủ, em cần giải thích scene extent được tính như thế nào.

---
## Slide 100: Scene Extent: chuẩn hoá ngưỡng kích thước theo camera
**Nội dung chính trên slide:** Công thức extent = 1.1 × khoảng cách xa nhất từ tâm camera trung bình; ví dụ tính toán trên cảnh 3 camera; extent dùng làm mốc cho các ngưỡng kích thước.

**Lời thuyết trình:**
Ngưỡng kích thước dùng để phân nhánh clone/split không phải một hằng số cố định, mà được chuẩn hoá theo quy mô thực tế của cảnh, gọi là scene extent. Cách tính: lấy tâm trung bình của tất cả vị trí camera, tìm khoảng cách xa nhất từ một camera bất kỳ đến tâm đó — gọi là diag — rồi nhân với hệ số 1.1 để có chút dư an toàn. Với ví dụ ba camera trong một cảnh đồ chơi, em tính ra tâm trung bình xấp xỉ (0, 0.167, -4), diag khoảng 1.5366, và extent khoảng 1.69. Điều quan trọng cần nhấn mạnh là: extent chỉ phụ thuộc vào vị trí camera, hoàn toàn không phụ thuộc vào điểm SfM hay vị trí các Gaussian. Từ extent này, ta suy ra ba mức tham chiếu: ngưỡng clone/split rất nhỏ khoảng 0.00169, ngưỡng prune Gaussian quá to khoảng 0.169, còn kích thước thực tế của Gaussian trong ví dụ lớn hơn nhiều, nên rơi hết vào nhánh split. Bây giờ ta đã có đủ ba mảnh ghép — gradient, Importance, và ngưỡng kích thước theo extent — để ráp thành sơ đồ quyết định densify hoàn chỉnh.

---
## Slide 101: Sơ đồ quyết định densify đầy đủ
**Nội dung chính trên slide:** Quy trình ba bước: kích thước chọn nhánh (nhỏ/to), nhánh nhỏ dùng gradient có dấu + Importance để clone, nhánh to dùng gradient trị tuyệt đối + Importance để split.

**Lời thuyết trình:**
Đây là sơ đồ tổng hợp toàn bộ logic densify của FastGS-lite, áp dụng cho mỗi Gaussian, mỗi một trăm hoặc năm trăm vòng lặp, trong khoảng từ iteration 500 đến 15000. Bước một: dựa vào kích thước lớn nhất của Gaussian so với ngưỡng delta nhân extent để chọn nhánh. Nếu Gaussian nhỏ — nghĩa là đang trong tình trạng "under-reconstruction", chưa phủ đủ vùng cần thiết — thì kiểm tra gradient trung bình có dấu vượt ngưỡng hai phẩy hai lần mười mũ âm bốn, và Importance lớn hơn 5; nếu cả hai đúng thì CLONE, tạo bản sao giữ nguyên tham số gốc. Nếu Gaussian to — "over-reconstruction", đang che lấp nhiều chi tiết — thì kiểm tra gradient trị tuyệt đối vượt ngưỡng cao hơn, và Importance lớn hơn 5; nếu đúng thì SPLIT thành hai Gaussian con nhỏ hơn, xoá Gaussian gốc. Điểm mấu chốt: nếu điều kiện Importance không thoả, 3DGS gốc vẫn sẽ densify chỉ cần gradient đạt, nhưng FastGS-lite kiên quyết chặn lại. Slide sau em sẽ đi sâu hơn vào bản chất khác nhau giữa hai tình trạng under và over-reconstruction.

---
## Slide 102: Under- vs. Over-reconstruction
**Nội dung chính trên slide:** Phân biệt hai "bệnh" khác nhau cùng biểu hiện gradient lớn nhưng khác nhau ở kích thước: under-reconstruction (Gaussian quá nhỏ) dẫn tới clone, over-reconstruction (Gaussian quá to) dẫn tới split.

**Lời thuyết trình:**
Ở đây em muốn làm rõ trực giác đằng sau việc chọn nhánh clone hay split. Cả hai tình huống đều biểu hiện qua cùng một tín hiệu — gradient vị trí lớn — nhưng bản chất vấn đề hoàn toàn khác nhau, và ta phân biệt bằng kích thước Gaussian. Trường hợp under-reconstruction: vùng cần phủ trong ảnh ground truth khá rộng, nhưng chỉ có một Gaussian quá nhỏ để phủ hết, nên gradient sẽ kéo nó giãn ra để cố lấp đầy — giải pháp đúng là CLONE thêm một bản để cùng chia nhau phủ vùng đó, và sau tối ưu hai bản sẽ tách ra hai phía. Trường hợp over-reconstruction thì ngược lại: có hai chi tiết nhỏ, hẹp, nằm cách nhau, nhưng chỉ một Gaussian to che lấp luôn cả hai; gradient ở hai nửa Gaussian đó kéo theo hai hướng ngược nhau, nên nếu cộng có dấu thì gần như triệt tiêu nhưng giá trị trị tuyệt đối lại rất lớn — giải pháp đúng là SPLIT thành hai Gaussian nhỏ hơn khớp đúng từng chi tiết. Đây chính là lý do sơ đồ quyết định dùng kích thước để chọn nhánh, còn dùng hai loại gradient khác nhau để xác nhận. Tiếp theo em sẽ trình bày cụ thể hình học của phép split.

---
## Slide 103: Hình học phép Split: sinh 2 Gaussian con
**Nội dung chính trên slide:** Công thức vị trí và scale của 2 Gaussian con sau split: tâm lấy mẫu quanh gốc theo chính hình dạng Gaussian gốc, scale chia 1.6, rotation giữ nguyên.

**Lời thuyết trình:**
Sang phần P4b, em trình bày chi tiết hình học của phép split. Khi một Gaussian bị split, nó bị xoá hoàn toàn và thay bằng hai Gaussian con. Vị trí của hai con không đặt tuỳ ý, mà được lấy mẫu ngẫu nhiên quanh tâm gốc theo đúng ma trận hiệp phương sai của chính Gaussian gốc — tức là lấy một epsilon từ phân phối chuẩn với phương sai bằng bình phương scale gốc, rồi xoay theo ma trận rotation của Gaussian gốc trước khi cộng vào tâm. Về scale, code dùng công thức scale chia cho 0.8 nhân N với N bằng 2 con, tương đương chia cho 1.6 — nghĩa là mỗi con nhỏ hơn gốc khoảng 1.6 lần. Rotation thì được giữ nguyên hoàn toàn cho cả hai con, chỉ có tâm và scale thay đổi, hướng ellipsoid không đổi. Kết quả là hai ellipsoid con nhỏ hơn, lệch tâm ngẫu nhiên nhưng luôn cùng hướng với ellipsoid cha ban đầu. Vậy tại sao lại chọn đúng hệ số 1.6 mà không phải một giá trị khác? Slide tiếp theo sẽ phân tích điều này.

---
## Slide 104: Phân bố lấy mẫu vị trí con và hệ số chia 1.6
**Nội dung chính trên slide:** So sánh các hệ số chia scale khác nhau (1, 1.6, 2, 3); hệ số 0.8 trong công thức được chọn để tổng khối lượng phủ của 2 con xấp xỉ khối lượng gốc.

**Lời thuyết trình:**
Em phân tích sâu hơn vì sao hệ số chia scale lại là 1.6 chứ không phải một con số khác. Nếu chia cho 1 thì hai Gaussian con gần như chồng lấn hoàn toàn lên nhau, không khác gì Gaussian gốc — vô nghĩa. Nếu chia cho 2 hoặc 3 thì hai con tách rời quá xa, để hở một khoảng trống ở giữa, làm mất liên tục vùng phủ. Hệ số 1.6 được chọn là điểm cân bằng: hai con vẫn phủ được gần hết vùng gốc nhưng đã đủ tách biệt để mỗi con có thể tự do tối ưu độc lập. Về mặt công thức, hệ số 0.8 trong scale chia cho 0.8 nhân N được thiết kế sao cho tổng "khối lượng" phủ của hai con xấp xỉ khối lượng phủ ban đầu của Gaussian gốc, tránh vừa để hở vừa chồng lấn quá mức. Em cũng minh hoạ trên cảnh đồ chơi với 4 Gaussian gốc tách thành 8 Gaussian con, bán kính con dao động khoảng 0.53 đến 0.70 đơn vị world. Còn với phép clone, do không thay đổi kích thước, làm sao hai bản sao giống hệt nhau lại có thể tách biệt được? Đó là nội dung slide tiếp theo.

---
## Slide 105: Clone: phá vỡ đối xứng nhờ trạng thái Adam
**Nội dung chính trên slide:** Cơ chế phá đối xứng giữa 2 bản clone không đến từ vị trí mà từ trạng thái Adam (m, v) khác nhau — bản gốc có lịch sử tích luỹ, bản clone khởi tạo lại bằng 0.

**Lời thuyết trình:**
Đây là một chi tiết rất tinh tế mà nhiều người bỏ qua. Khi CLONE, hai bản sao được tạo ra với tham số hoàn toàn giống nhau — cùng vị trí, scale, rotation, opacity, hệ số cầu điều hoà — tại thời điểm tạo ra. Câu hỏi đặt ra: nếu hai bản giống hệt nhau và nhận gradient giống hệt nhau ở mọi bước, thì làm sao chúng tách biệt để phát huy tác dụng? Câu trả lời không nằm ở việc dịch chuyển vị trí ban đầu, mà nằm ở trạng thái của bộ tối ưu Adam. Khi nối tensor tham số mới vào optimizer, bản clone được khởi tạo lại moment bậc một và bậc hai của Adam bằng 0, trong khi bản gốc đã mang theo toàn bộ lịch sử tích luỹ Adam từ trước đó. Vì bộ đếm bước dùng chung cho cả nhóm, công thức bias-correction áp dụng không đối xứng lên hai bản có m, v khác nhau, khiến hai quỹ đạo cập nhật lệch nhau ngay từ bước đầu tiên dù gradient ban đầu bằng nhau. Kết quả là khoảng cách giữa hai bản tăng dần theo thời gian tối ưu. Bây giờ, hãy xem việc liên tục clone và split như vậy sẽ khiến tổng số Gaussian tăng trưởng ra sao.

---
## Slide 106: Tăng trưởng N qua các vòng Densify
**Nội dung chính trên slide:** Công thức tăng trưởng luỹ thừa N_n = N_0 · q^n với q = (1+r_spawn)(1-r_prune); so sánh số vòng densify giữa interval 100 và interval 500.

**Lời thuyết trình:**
Sau mỗi vòng densify, số lượng Gaussian không tăng tuyến tính mà tăng theo hàm luỹ thừa. Công thức là N tại vòng k+1 bằng N tại vòng k nhân với (1 cộng tỉ lệ sinh mới) rồi nhân với (1 trừ tỉ lệ bị xoá), với tỉ lệ sinh mới bằng tổng tỉ lệ clone cộng tỉ lệ split. Gọi q là hệ số nhân mỗi vòng, thì sau n vòng ta có N bằng N ban đầu nhân q mũ n — đây là tăng trưởng luỹ thừa, không phải cộng dồn tuyến tính. Cửa sổ densify chạy từ iteration 500 đến 15000; nếu chu kỳ densify là 100 vòng thì có 144 lần densify, còn nếu chu kỳ là 500 vòng thì chỉ có 28 lần. Với ví dụ tỉ lệ sinh 0.15 và tỉ lệ xoá 0.05, q xấp xỉ 1.0925, và khi nâng lên luỹ thừa 144 so với luỹ thừa 28 thì chênh lệch tổng số Gaussian là nhiều bậc độ lớn — cho thấy tần suất densify ảnh hưởng cực kỳ mạnh đến kết quả cuối. Tốc độ tăng cũng tự nhiên chậm lại về cuối vì scene đã dày đặc, ít Gaussian còn đạt điều kiện. Để việc đếm gradient ở mỗi vòng chính xác, hệ thống cần một bước dọn dẹp quan trọng — đó là reset thống kê tích luỹ.

---
## Slide 107: Reset thống kê tích luỹ sau mỗi vòng Densify
**Nội dung chính trên slide:** Sau mỗi lần densify_and_prune_fastgs, ba tensor (xyz_gradient_accum, denom, max_radii2D) được đưa về 0 cho toàn bộ N, tránh cộng dồn sai giữa các chu kỳ.

**Lời thuyết trình:**
Trong một chu kỳ densify 100 vòng, có ba tensor tích luỹ theo từng Gaussian: xyz_gradient_accum là tổng norm gradient khi hữu hình, denom là số lần hữu hình, và max_radii2D là bán kính chiếu lên màn hình 2D lớn nhất từng đạt được. Ngay sau khi hàm densify_and_prune_fastgs thực thi xong, hàm densification_postfix sẽ đưa cả ba tensor này về 0 cho toàn bộ N Gaussian — kể cả những Gaussian không hề được clone hay split ở vòng đó. Đây là bước bắt buộc, vì nếu quên reset, thống kê của vòng trước sẽ cộng dồn sai lẫn vào vòng sau, khiến gradient trung bình bị lệch và toàn bộ quyết định densify hoặc prune ở vòng kế tiếp sai tín hiệu. Ngoài ra, ngưỡng max_screen_size 20 pixel chỉ được kích hoạt kiểm tra sau iteration 3000, dùng max_radii2D để loại bỏ Gaussian chiếm màn hình quá lớn. Sau khi hiểu việc reset, ta cần biết thêm: khi số lượng Gaussian thay đổi do clone/split, các tensor tham số và chỉ số được sắp xếp lại như thế nào — đó là nội dung slide cuối của phần P4.

---
## Slide 108: Bố trí lại Index/Tensor tham số sau Densify
**Nội dung chính trên slide:** Cách nối tensor theo mẫu repeat(N,1) sau split, prune_filter xoá Gaussian gốc, và hệ quả phụ về padded_importance chỉ áp cho lớp con đầu tiên theo chỉ số cũ.

**Lời thuyết trình:**
Slide cuối phần P4 này đi vào chi tiết cách cài đặt bên trong khi số lượng Gaussian thay đổi. Sau split, các tensor tham số được nối theo thứ tự cố định: toàn bộ Gaussian gốc trước, tiếp theo là lớp con thứ nhất của tất cả các Gaussian, rồi đến lớp con thứ hai — theo mẫu repeat N lần. Sau đó, một prune_filter được dùng để xoá sạch các Gaussian gốc đã split, chỉ giữ lại hai lớp con. Trạng thái Adam và ba tensor thống kê cũng được nối và cắt đồng bộ theo cùng chỉ số, rồi reset về 0 cho quần thể N mới — thao tác này thực hiện trực tiếp trên dữ liệu thô, không làm đứt gradient graph. Có một hệ quả phụ khá tinh vi: trọng số dùng cho việc xoá dựa trên padded_importance chỉ được gán đúng cho N phần tử đầu — tức lớp con thứ nhất — theo chỉ số cũ, chứ không theo danh tính Gaussian; kết quả là lớp con thứ hai nhận trọng số bằng 0, tạm thời miễn nhiễm với việc bị xoá ở vòng đó. Toàn bộ thứ tự thao tác gồm sáu bước: tính gradient, clone, split, prune kèm multinomial, ép opacity, rồi dọn biến tạm. Phần tiếp theo, P5, em sẽ đi sâu vào chính cơ chế multinomial pruning này.

---
## Slide 109: Từ điểm importance sang trọng số xoá: ví dụ N=10
**Nội dung chính trên slide:** Chuyển điểm Pruning_i thành trọng số w_i = 1/(1e-6 + (1-P_i)); trọng số phân kỳ mạnh khi P_i tiến gần 1.

**Lời thuyết trình:**
Bước sang phần P5, em trình bày cách hệ thống quyết định Gaussian nào bị xoá trong bước prune. Với mỗi Gaussian có một điểm số Pruning, gọi là P_i, thể hiện mức độ "nên bị xoá". Em minh hoạ với 10 Gaussian có điểm P cho trước, trong đó tập ứng viên bị xem xét xoá gồm 5 Gaussian, và ngân sách xoá được tính bằng làm tròn xuống của một nửa số ứng viên, ra kết quả 2. Trọng số bị xoá được tính tỉ lệ nghịch với (1 trừ P_i), cộng thêm một số rất nhỏ để tránh chia cho 0, sau đó chuẩn hoá thành xác suất bằng cách chia cho tổng. Điểm đáng chú ý là trọng số này phân kỳ rất mạnh khi P_i tiến gần tới 1: trong ví dụ, Gaussian có P bằng 1.0 chiếm tới 99.99% xác suất được rút ra ở lượt lấy mẫu đầu tiên — gần như chắc chắn bị xoá. Đây là cách hệ thống ưu tiên xoá những Gaussian có điểm pruning cao nhất một cách rất dứt khoát. Nhưng liệu công thức lý thuyết này có khớp với thực nghiệm khi lấy mẫu nhiều lần? Slide sau sẽ kiểm chứng bằng mô phỏng.

---
## Slide 110: Kiểm chứng bằng mô phỏng: 2000 lần lấy mẫu multinomial
**Nội dung chính trên slide:** Mô phỏng lấy mẫu multinomial không hoàn lại 2000 lần, so sánh xác suất lý thuyết P(i∈S) với tần suất thực nghiệm; phân biệt "được rút" và "thực sự bị xoá" (S∩C).

**Lời thuyết trình:**
Để kiểm chứng công thức trọng số ở slide trước có đúng như thiết kế hay không, em thực hiện lấy mẫu multinomial không hoàn lại với ngân sách bằng 2, lặp lại 2000 lần độc lập. Công thức xác suất lý thuyết để một Gaussian nằm trong tập được rút S, với ngân sách 2, phải tính cả trường hợp nó được rút ở lượt đầu và trường hợp được rút ở lượt thứ hai sau khi một phần tử khác đã bị loại — công thức này khá phức tạp nhưng tính được chính xác. Kết quả mô phỏng qua 2000 lần cho tần suất thực nghiệm khớp rất sát với xác suất lý thuyết, xác nhận cơ chế lấy mẫu hoạt động đúng như thiết kế. Một điểm cần phân biệt rõ: một Gaussian có thể được "rút" vào tập S nhưng nếu nó không nằm trong tập ứng viên C ban đầu thì vẫn không bị xoá thực sự — tập bị xoá thực tế là giao của S và C. Vì vậy số Gaussian bị xoá thực tế trung bình thường thấp hơn ngân sách 2. Sau khi xác nhận cơ chế đúng, câu hỏi tiếp theo là: tại sao lại chọn cách lấy mẫu ngẫu nhiên này thay vì đơn giản là cắt top-k theo điểm số?

---
## Slide 111: Top-k cứng vs. Multinomial: hai chiến lược pruning
**Nội dung chính trên slề:** So sánh ưu nhược điểm của cắt top-k cứng (deterministic, dễ đoán nhưng cắt sạch cả cụm) và multinomial (stochastic, đa dạng hơn nhưng có nhiễu).

**Lời thuyết trình:**
Ở đây em so sánh trực tiếp hai chiến lược pruning khả dĩ. Em mô phỏng 300 Gaussian với điểm số P được chuẩn hoá từ phân phối Beta, và một tập ứng viên chọn ngẫu nhiên 40% quần thể, độc lập với điểm số. Chiến lược thứ nhất là top-k cứng: sắp xếp tập ứng viên giảm dần theo điểm số rồi xoá đúng số lượng ngân sách những phần tử điểm thấp nhất. Ưu điểm là dễ đoán, luôn xoá đúng số lượng đã định. Nhược điểm là cắt sạch theo một ngưỡng cứng, dễ xoá nhầm cả một cụm liền kề có điểm số tương tự nhau cùng lúc. Chiến lược thứ hai là multinomial: rút mẫu không hoàn lại theo trọng số tỉ lệ nghịch với (1 trừ P) trên toàn bộ quần thể rồi giao với tập ứng viên. Ưu điểm là đa dạng hơn, tránh xoá đồng loạt một vùng; nhược điểm là có nhiễu, số lượng xoá thực tế thường ít hơn ngân sách vì phần giao có thể nhỏ hơn tập được rút. Vậy vì sao tính "tránh xoá đồng loạt một vùng" lại quan trọng đến vậy? Slide tiếp theo sẽ minh hoạ bằng một ví dụ cụ thể về cụm hình học.

---
## Slide 112: Vì sao lấy mẫu ngẫu nhiên tránh xoá sạch một cụm
**Nội dung chính trên slide:** Ví dụ một "cụm nóng" 2D có điểm P đồng đều cao; top-k cứng xoá gần hết cụm tạo lỗ hổng hình học, multinomial chỉ xoá một phần nhờ tính ngẫu nhiên.

**Lời thuyết trình:**
Đây là ví dụ trực quan nhất cho thấy vì sao cần tính ngẫu nhiên trong pruning. Em dựng kịch bản 300 điểm trong không gian 2D, trong đó có một "cụm nóng" gần một vị trí trung tâm, bán kính khoảng 1.6, mà mọi điểm trong cụm đều có điểm pruning cao và khá đồng đều với nhau; phần còn lại của quần thể có điểm số trải rộng hơn. Ngân sách xoá là 15% tổng quần thể, và toàn bộ quần thể đều là ứng viên. Với top-k cứng: vì mọi điểm trong cụm đều có điểm cao và tương tự nhau, thuật toán sẽ xoá gần như toàn bộ cụm cùng một lúc, tạo ra một lỗ hổng hình học rõ rệt tại đúng vị trí đó — có thể phá hỏng một vùng quan trọng của cảnh. Với multinomial: nhờ tính ngẫu nhiên trong lấy mẫu theo trọng số, chỉ một phần của cụm bị xoá ở mỗi lượt, phần còn lại được giữ lại để tái đánh giá ở lượt sau, tránh mất mát hình học đột ngột. Đây chính là lý do chính khiến hệ thống chọn lấy mẫu xác suất thay vì cắt ngưỡng cứng tuyệt đối. Bây giờ ta chuyển sang một cơ chế khác của ADC: reset opacity định kỳ, bắt đầu từ nền tảng toán học là hàm sigmoid nghịch đảo.

---
## Slide 113: Sigmoid nghịch đảo (logit) để reset opacity
**Nội dung chính trên slide:** Opacity thực chất được tối ưu qua logit; công thức sigmoid và nghịch đảo; các mốc opacity quan trọng (prune 0.005, reset 0.01, clamp 0.8).

**Lời thuyết trình:**
Một điểm kỹ thuật quan trọng cần làm rõ trước khi nói về reset opacity: tham số thực sự được tối ưu bên trong mô hình không phải alpha nằm trong khoảng (0,1), mà là một biến gọi là logit, và alpha chỉ là kết quả của việc đưa logit qua hàm sigmoid. Ngược lại, muốn biết logit tương ứng với một alpha cho trước, ta dùng hàm nghịch đảo, chính là hàm logarit của tỉ số alpha trên (1 trừ alpha). Vì sao phải làm vậy? Vì nếu muốn ép hoặc reset opacity về một giá trị nhỏ, ta không thể gán trực tiếp lên alpha — làm vậy sẽ phá vỡ ràng buộc alpha phải nằm trong (0,1) và làm sai gradient trong quá trình lan truyền ngược; thay vào đó phải gán lại giá trị logit tương ứng. Hệ thống có bốn mốc alpha quan trọng: ngưỡng prune 0.005, giá trị reset 0.01, ngưỡng prune cuối 0.1, và ngưỡng clamp 0.8. Một ví dụ cụ thể: bước từ alpha 0.9 xuống 0.8 chỉ là một bước nhỏ trong không gian logit, nhưng bước từ 0.9 xuống 0.01 là một bước rất xa — cho thấy reset opacity thực chất là một cú "giật lùi" mạnh trong không gian logit. Slide tiếp theo sẽ cho thấy quỹ đạo opacity thực tế biến đổi ra sao qua thời gian huấn luyện.

---
## Slide 114: Quỹ đạo opacity qua thời gian: tăng rồi bị reset định kỳ
**Nội dung chính trên slide:** Mô phỏng 4 Gaussian với các "sức kéo" gradient khác nhau; clamp 0.8 sau mỗi 100 vòng, reset về 0.01 mỗi 3000 vòng (opacity_reset_interval).

**Lời thuyết trình:**
Em mô phỏng bốn Gaussian với các mức độ hữu ích khác nhau để minh hoạ quỹ đạo opacity theo thời gian: một Gaussian hữu ích có gradient kéo logit tăng đều đặn, một Gaussian trung bình tăng chậm hơn, một Gaussian vô dụng có gradient kéo logit giảm dần, và một Gaussian "hồi sinh muộn" ban đầu không đổi nhưng bắt đầu tăng sau iteration 7000. Hai cơ chế can thiệp định kỳ diễn ra song song: mỗi 100 vòng trong khoảng cho phép densify, logit bị ép không vượt quá giá trị tương ứng với alpha 0.8, ngăn opacity tăng quá cao ngay sau densify. Còn mỗi opacity_reset_interval bằng 3000 vòng, logit của TOÀN BỘ Gaussian, bất kể trước đó cao thế nào, bị kéo về giá trị tương ứng alpha 0.01 — gần như về 0. Sau mỗi lần reset như vậy, Gaussian hữu ích nhờ gradient dương mạnh sẽ nhanh chóng "leo" trở lại mức cao, trong khi Gaussian vô dụng rơi xuống dưới ngưỡng prune 0.005 và bị xoá ở lần prune multinomial kế tiếp. Cơ chế này về bản chất cho mọi Gaussian "bị nghi ngờ" một cơ hội chứng minh lại giá trị của mình, thay vì bị xoá vĩnh viễn ngay lập tức. Để thấy tác động ở quy mô toàn bộ quần thể, slide sau sẽ trình bày histogram phân bố opacity trước và sau một lần reset.

---
## Slide 115: Histogram phân bố opacity: trước và sau một lần reset
**Nội dung chính trên slide:** Mô phỏng 20000 Gaussian; ngay sau reset toàn bộ phân bố dồn về sát 0.01, sau đó tách lại thành hai nhóm hữu ích/vô dụng theo tốc độ leo lại khác nhau.

**Lời thuyết trình:**
Ở quy mô lớn, em mô phỏng 20000 Gaussian, trong đó 60% được xem là "hữu ích" với opacity ban đầu thiên về giá trị cao, và 40% được xem là "lơ lửng" với opacity ban đầu thiên về giá trị thấp. Trước khi reset, phân bố opacity khá đa dạng, trải rộng trên toàn khoảng (0,1). Nhưng ngay sau khi áp dụng công thức reset cho mọi Gaussian, toàn bộ phân bố gần như dồn cứng về sát giá trị 0.01, gần như đồng nhất — mất hết mọi sự đa dạng đã tích luỹ trước đó. Điều thú vị xảy ra khoảng 500 vòng sau: nhóm hữu ích, nhờ gradient dương mạnh, leo lại rất nhanh với mức tăng trung bình đáng kể trong không gian logit; còn nhóm vô dụng leo rất chậm hoặc gần như đứng yên, thậm chí còn giảm nhẹ. Kết quả là phân bố opacity tách trở lại thành hai nhóm rõ rệt — đúng như mục đích thiết kế của cơ chế reset: buộc mọi Gaussian phải "chứng minh lại" giá trị của mình một cách công bằng. Tỉ lệ Gaussian dưới ngưỡng prune được ghi trực tiếp trên mỗi histogram để so sánh ba giai đoạn. Nhưng việc reset opacity chỉ là nửa câu chuyện — nó còn ảnh hưởng tới trạng thái của bộ tối ưu Adam, đó là nội dung slide cuối cùng của phần em phụ trách.

---
## Slide 116: Trạng thái Adam sau reset: có nên xoá luôn moment?
**Nội dung chính trên slide:** Khi reset opacity, code chỉ đặt lại (m,v)=(0,0) nhưng KHÔNG reset bước đếm step của Adam, khiến bias-correction gần như không còn hiệu lực và bước cập nhật đầu tiên lớn bất thường.

**Lời thuyết trình:**
Slide cuối cùng của phần em phụ trách đi vào một chi tiết cài đặt tinh vi liên quan đến bộ tối ưu Adam sau khi reset opacity. Nhắc lại nhanh: Adam cập nhật hai moment m và v theo trung bình mũ, rồi áp dụng bias-correction chia cho (1 trừ beta mũ t) để bù trừ cho việc m, v ban đầu bằng 0. Trong code thật, khi reset opacity, hàm replace_tensor_to_optimizer chỉ đặt lại m và v về 0 cho nhóm tham số opacity, nhưng KHÔNG reset bộ đếm bước t của Adam — bước t vẫn tiếp tục tính từ giá trị cũ, ví dụ 6000. Hệ quả là gì? Với t lớn, beta mũ t gần như bằng 0, nên bias-correction gần như mất tác dụng đúng nghĩa vốn được thiết kế cho t nhỏ — khiến bước cập nhật đầu tiên ngay sau reset lớn bất thường, xấp xỉ gấp hơn 3 lần learning rate thông thường theo ước tính của em. So với việc khởi tạo Adam hoàn toàn mới với step bằng 0, phiên bản "giữ step cũ" này tạo ra một cú nhảy tham số lớn hơn nhiều ngay sau reset, rồi mới hội tụ dần về mức ổn định. Nếu không reset cả trạng thái Adam thì optimizer vẫn "nhớ" quán tính cũ, có thể khiến opacity hồi phục lệch hướng hoặc chậm hơn dự kiến so với một khởi động hoàn toàn sạch. Đến đây là hết phần ADC chuyên sâu P4 và P5-A mà em phụ trách, xin chuyển tiếp sang phần P5-B của nhóm kế tiếp.
