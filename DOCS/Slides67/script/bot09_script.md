# Kịch bản thuyết trình — Bot 09 (Slide 155–173)
## Phạm vi: Cơ chế theo dòng thời gian - phần B (partADC_TLb.tex), Phần 5 (part5.tex) — Ba đòn bẩy tăng tốc FastGS & Mô hình chi phí

---

## Slide 155: Split: mẫu Monte Carlo vị trí con quanh gốc
**Nội dung chính trên slide:** Mô phỏng Monte Carlo 2000 cặp Gaussian con sinh ra khi split một Gaussian gốc có scale (0.06, 0.02), góc xoay 40°; kiểm chứng độ lệch chuẩn mẫu khớp covariance lý thuyết, và diện tích ellipse mỗi con co lại còn 0,391 lần gốc.

**Lời thuyết trình:**
Tiếp tục timeline của ADC, ở phần này em đi sâu vào bước prune và reset opacity — hai cơ chế "dọn dẹp" diễn ra sau clone/split. Trước tiên em kiểm chứng lại việc sinh Gaussian con bằng mô phỏng Monte Carlo: với Gaussian gốc có scale (0,06; 0,02) và góc xoay 40 độ, em lấy mẫu 2000 cặp vị trí con theo phân phối chuẩn trong khung cục bộ rồi xoay về khung thế giới. Kết quả đo được cho thấy độ lệch chuẩn mẫu theo trục dài và trục ngắn khớp gần như tuyệt đối với giá trị lý thuyết 0,06 và 0,02, xác nhận công thức sinh con là đúng. Đáng chú ý, diện tích ellipse của mỗi Gaussian con chỉ còn 0,391 lần gốc, nghĩa là thể tích 3D co lại khoảng 4,1 lần — đúng như thiết kế để tránh chồng lấn khi tách đôi. Đây là nền tảng để bước sang phần kế toán số lượng Gaussian.

---
## Slide 156: Kế toán số Gaussian N qua một bước densify + prune
**Nội dung chính trên slide:** Công thức cân bằng N_{t+1} = N_t + clone + split − prune; so sánh tăng trưởng lũy thừa N(t) = N0(1+r)^t với tăng tuyến tính qua 144 bước densify.

**Lời thuyết trình:**
Câu hỏi đặt ra là: số lượng Gaussian N thay đổi thế nào qua mỗi vòng densify và prune? Em xây dựng một phép "kế toán" đơn giản: với N ban đầu là 1000, clone thêm 120, split 80 — mỗi Gaussian split sinh hai con và xóa một gốc nên đóng góp ròng dương 80 — rồi trừ đi 45 Gaussian bị prune, ta được N mới là 1155. Điều quan trọng hơn là nếu tỉ lệ densify giữ nguyên mỗi bước, N sẽ tăng theo cấp số nhân chứ không phải tuyến tính. Với N0 là 100.000 và 144 bước densify trải từ vòng 500 đến 15.000, mô hình lũy thừa cho ra N gấp khoảng 8,2 lần N0, trong khi mô hình tuyến tính chỉ gấp 3,16 lần. Sự chênh lệch này chính là lý do vì sao ADC bắt buộc phải có các tiêu chí prune cứng và reset opacity định kỳ — nếu không, số Gaussian sẽ bùng nổ vượt ngân sách bộ nhớ và tốc độ render. Slide sau em sẽ trình bày tiêu chí prune đầu tiên.

---
## Slide 157: Tiêu chí prune 1: opacity thấp α_i < 0,005
**Nội dung chính trên slide:** Ngưỡng min_opacity mặc định của Inria; mô phỏng trên 20.000 Gaussian cho thấy 981 Gaussian (4,91%) bị xóa vì opacity quá thấp.

**Lời thuyết trình:**
Tiêu chí prune đầu tiên và cũng là dễ hiểu nhất: xóa mọi Gaussian có opacity alpha nhỏ hơn 0,005, đây là ngưỡng min_opacity mặc định của Inria. Về mặt vật lý, một Gaussian gần như trong suốt hoàn toàn thì đóng góp không đáng kể vào công thức alpha-blending khi render ảnh — giữ lại nó chỉ tốn bộ nhớ và tính toán vô ích. Em mô phỏng trên quần thể 20.000 Gaussian với opacity trộn từ nhiều nguồn khác nhau: phần lớn là Gaussian hữu ích, một cụm nhỏ vừa mới reset opacity, và một số ở mức trung gian. Kết quả có 981 trên 20.000 Gaussian, tức khoảng 4,91%, bị xóa theo tiêu chí này. Đặc biệt, phân phối alpha có một cụm nhỏ dồn sát về 0 — đó chính là các Gaussian vừa bị reset opacity nhưng chưa kịp hồi phục, và chúng có nguy cơ bị prune ngay ở lần densify kế tiếp nếu gradient không đủ mạnh để kéo alpha lên lại. Tiếp theo là tiêu chí prune thứ hai, liên quan đến kích thước chiếu lên ảnh 2D.

---
## Slide 158: Tiêu chí prune 2: bán kính chiếu 2D r_i^2D > 20 px
**Nội dung chính trên slide:** Công thức xấp xỉ hình học r^2D ≈ f·s/z; ngưỡng 20px chỉ áp dụng khi t > 3000; ví dụ số cho thấy cùng scale 3D nhưng độ sâu khác nhau dẫn tới giữ hoặc xóa khác nhau.

**Lời thuyết trình:**
Tiêu chí thứ hai nhìn vào kích thước Gaussian sau khi chiếu lên ảnh 2D, xấp xỉ bằng công thức r 2D gần bằng f nhân s chia z, với f là tiêu cự, s là scale 3D và z là độ sâu tới camera. Ngưỡng của Inria là xóa nếu bán kính chiếu vượt quá 20 pixel, nhưng chỉ áp dụng sau vòng 3000, tức sau lần reset opacity đầu tiên — trước đó điều kiện này bị bỏ qua để tránh xóa nhầm những Gaussian to đang trong quá trình hình thành. Giá trị max_radii2D được cập nhật liên tục mỗi vòng bởi rasterizer, lấy giá trị lớn nhất quan sát được qua mọi góc nhìn. Ví dụ minh họa rất trực quan: với scale 0,01 ở độ sâu 1, bán kính chiếu là 10 pixel nên được giữ; nhưng scale 0,03 ở cùng độ sâu cho ra 30 pixel nên bị xóa; tuy nhiên cũng scale 0,03 đó nếu ở độ sâu 3 thì bán kính chỉ còn 10 pixel và lại được giữ. Điều này cho thấy cùng một scale 3D có thể bị xóa hay giữ tùy vào khoảng cách tới camera — đây chính là điểm khác biệt so với tiêu chí scale tuyệt đối mà em sẽ trình bày ngay sau đây.

---
## Slide 159: Tiêu chí prune 3: scale tuyệt đối max(s_i) > 0,1 · extent
**Nội dung chính trên slide:** Ngưỡng scale tuyệt đối không phụ thuộc camera, dùng để xóa các "floater" to; ví dụ 4 Gaussian với tỉ lệ khác nhau, so sánh với ngưỡng khởi tạo split/clone nhỏ hơn 10 lần.

**Lời thuyết trình:**
Khác với tiêu chí trước phụ thuộc vào góc nhìn camera, tiêu chí thứ ba hoàn toàn dựa trên không gian thế giới: nếu bán trục lớn nhất của Gaussian vượt quá 10% kích thước cảnh, gọi là extent, thì bị xóa ngay lập tức. Về mặt diễn giải, một Gaussian có tỉ lệ max scale trên extent lớn hơn 0,1 thường là dấu hiệu của artefact "sương" hoặc floater to, phủ lên nhiều vật thể khác nhau mà không mô tả bề mặt cụ thể nào — nên bị xóa ngay bất kể opacity hay gradient của nó ra sao. Em minh họa với 4 Gaussian có tỉ lệ lần lượt là 0,02, 0,05, 0,10 và 0,18: chỉ hai Gaussian đầu được giữ lại, còn hai Gaussian sau bị xóa, với quy ước ranh giới 0,10 tính là vượt ngưỡng. Một điểm thú vị là ngưỡng khởi tạo cho split và clone chỉ dùng delta bằng 0,01 lần extent — nhỏ hơn ngưỡng prune này đúng 10 lần, cho thấy hai cơ chế đang nhắm vào hai thái cực kích thước hoàn toàn khác nhau. Slide tiếp theo em sẽ trình bày cách ba tiêu chí này kết hợp lại với nhau.

---
## Slide 160: Kết hợp tiêu chí prune bằng phép OR
**Nội dung chính trên slide:** Công thức hợp ba điều kiện xóa (A: alpha thấp, B: bán kính 2D lớn, C: scale tuyệt đối lớn); số liệu tại t=9000 và t=2000 cho thấy vùng hợp nhỏ hơn tổng ba vùng do chồng lấn.

**Lời thuyết trình:**
Ba tiêu chí prune vừa nêu không hoạt động độc lập mà được kết hợp bằng phép OR logic: một Gaussian bị xóa nếu vi phạm ít nhất một trong ba điều kiện A, B, hoặc C. Về mặt hình học, vùng quyết định xóa trong không gian tham số chính là hợp của ba tập hợp A hợp B hợp C. Em mô phỏng trên 20.000 Gaussian tại thời điểm t bằng 9000, khi cả ba điều kiện đều hoạt động: tập A có 981 phần tử, tập B có 1869, tập C có 237, nhưng tổng vùng hợp chỉ là 2751 Gaussian, tức khoảng 13,8% — nhỏ hơn hẳn tổng cộng từng phần vì các điều kiện có chồng lấn nhau, ví dụ một Gaussian to thường cũng bị mờ do hiệu ứng sương, nên vùng giao giữa A và C khác rỗng. Ngược lại, tại t bằng 2000, do chưa vượt mốc 3000 nên điều kiện B bị bỏ qua, chỉ còn A và C hoạt động, tổng xóa giảm xuống còn 6,0%. Cần nhấn mạnh đây là prune cứng — xóa ngay lập tức, không cần ứng viên hay ngân sách như trong quá trình densify. Bây giờ em chuyển sang lý do vì sao opacity được lưu dưới dạng logit thay vì giá trị trực tiếp.

---
## Slide 161: Vì sao lưu opacity dưới dạng logit: sigmoid và nghịch đảo
**Nội dung chính trên slide:** Tham số hóa alpha qua x không chặn bằng sigmoid và nghịch đảo logit; các mốc giá trị cụ thể; công thức reset opacity qua ánh xạ ngược.

**Lời thuyết trình:**
Một chi tiết kỹ thuật quan trọng: opacity alpha luôn phải nằm trong khoảng từ 0 đến 1, nhưng nếu tối ưu trực tiếp trên alpha bằng gradient descent thì rất dễ vi phạm ràng buộc này. Giải pháp là tham số hóa gián tiếp qua biến x không bị chặn, với alpha bằng sigma của x theo công thức sigmoid, và chiều ngược lại x bằng logarit của alpha chia cho 1 trừ alpha, gọi là hàm logit. Nhờ vậy, việc tối ưu x bằng gradient descent không cần ràng buộc gì cả, vì alpha bằng sigma của x sẽ tự động nằm trong khoảng 0 đến 1. Một vài mốc cụ thể: alpha bằng 0,005 tương ứng x xấp xỉ âm 5,293; alpha bằng 0,01 tương ứng x xấp xỉ âm 4,595; và alpha bằng 0,99 tương ứng x xấp xỉ dương 4,595. Khi reset opacity, công thức là lấy giá trị nhỏ hơn giữa alpha hiện tại và 0,01, rồi ánh xạ ngược về không gian x thông qua hàm logit nghịch đảo. Hiểu được cơ chế này sẽ giúp em giải thích rõ hơn quỹ đạo opacity theo thời gian ở slide tiếp theo.

---
## Slide 162: Quỹ đạo opacity theo thời gian: nhảy xuống rồi phục hồi
**Nội dung chính trên slide:** Mô phỏng 4 Gaussian qua 4 lần reset tại t = 3000, 6000, 9000, 12000; các loại quỹ đạo khác nhau — hồi phục nhanh, không hồi phục và bị prune, hồi phục chậm, Gaussian mới sinh.

**Lời thuyết trình:**
Slide này minh họa trực quan điều gì xảy ra với opacity của từng Gaussian qua nhiều lần reset liên tiếp tại các mốc t bằng 3000, 6000, 9000 và 12000, với giá trị khởi tạo alpha0 bằng 0,1. Em theo dõi bốn trường hợp điển hình. Trường hợp thứ nhất là Gaussian thực sự hữu ích: sau mỗi lần reset, nó hồi phục rất nhanh về khoảng 0,9, chứng tỏ gradient liên tục đẩy nó lên. Trường hợp thứ hai là Gaussian "ảo" hoặc dư thừa: sau lần reset tại t bằng 6000, nó không còn hồi phục được nữa vì gradient không đủ mạnh, alpha giảm dần và bị prune ngay khi chạm ngưỡng 0,005 ở lần densify kế tiếp. Trường hợp thứ ba là Gaussian hữu ích vừa phải, hồi phục chậm hơn và chỉ đạt khoảng 0,4. Trường hợp thứ tư là Gaussian mới sinh ra do clone hoặc split tại t bằng 9000, bắt đầu tham gia chu trình reset từ mốc 12000 trở đi. Mỗi lần reset giống như một cú nhảy xuống tức thời về alpha bằng 0,01 trên thang log, sau đó quỹ đạo phân nhánh rõ rệt — hoặc leo trở lại nếu hữu ích, hoặc tụt xuống dưới ngưỡng xóa nếu vô dụng. Đây chính là cơ chế "sàng lọc" cốt lõi của ADC.

---
## Slide 163: Toàn cảnh: histogram opacity trước/sau một lần reset
**Nội dung chính trên slide:** Ba histogram alpha tại thời điểm trước reset, ngay sau reset (dồn về 0,01), và 500 vòng sau; tỉ lệ nhóm hữu ích và vô dụng hồi phục khác nhau.

**Lời thuyết trình:**
Để tổng kết toàn bộ cơ chế reset opacity, em nhìn vào bức tranh toàn cảnh trên 20.000 Gaussian quanh thời điểm reset t bằng 6000, trong đó 72% là nhóm "hữu ích" với alpha trước reset khoảng 0,80, và 28% là nhóm "vô dụng" với alpha khoảng 0,15. Trước khi reset, phân phối alpha trải rộng và đã có một lượng nhỏ nằm dưới ngưỡng xóa 0,005. Ngay sau khi reset, công thức alpha bằng min của alpha cũ và 0,01 được áp dụng cho toàn bộ, khiến gần như mọi Gaussian dồn về đúng giá trị 0,01 — một cú "reset toàn cục". Rồi 500 vòng sau, nhóm hữu ích hồi phục rất mạnh, vượt lại trên 0,5, trong khi nhóm vô dụng tiếp tục giảm, một phần rơi xuống dưới 0,005 và bị prune, nhưng đáng chú ý là 12% của nhóm vô dụng vẫn hồi phục vừa phải và được giữ lại — cho thấy cơ chế không tuyệt đối cứng nhắc. Đây chính là cơ chế cốt lõi khép lại toàn bộ chu trình ADC theo thời gian: buộc mọi Gaussian phải liên tục "chứng minh lại" giá trị tồn tại của mình qua gradient. Đến đây em đã hoàn tất phần cơ chế ADC, tiếp theo em chuyển sang phần thứ năm nói về ba đòn bẩy tăng tốc chính của FastGS.

---
## Slide 164: Vì sao 3DGS gốc chậm?
**Nội dung chính trên slide:** Bốn khâu chính mỗi iteration (render, tính loss, backward, densify/prune) và tỉ trọng thời gian ước lượng: render 45%, backward 35%, loss 10%, densify/prune 10% nhưng đột biến.

**Lời thuyết trình:**
Bắt đầu phần 5, em phân tích tại sao 3DGS gốc lại chậm, làm nền tảng để hiểu FastGS cải tiến ở đâu. Mỗi iteration huấn luyện gồm bốn khâu chính: render tức là chiếu và rasterize, tính loss, backward tức là lan truyền ngược gradient, và định kỳ chạy densify/prune. Với cảnh thực tế có hàng trăm nghìn đến hàng triệu Gaussian, chi phí rasterize và backward chiếm phần lớn thời gian mỗi bước. Theo bảng ước lượng, render chiếm khoảng 45% thời gian, backward khoảng 35%, tính loss chỉ khoảng 10%, còn densify/prune tuy chỉ chiếm khoảng 10% tổng thời gian huấn luyện nhưng lại chạy định kỳ chứ không phải mỗi iteration, và mỗi lần chạy thì gây đột biến chi phí đáng kể. Từ bảng phân tích này ta thấy rõ: nếu muốn tăng tốc toàn bộ hệ thống một cách hiệu quả, phải nhắm đúng vào những khâu chiếm tỉ trọng lớn nhất — đó là render và backward. Đây chính là tiền đề để em giới thiệu định luật Amdahl ở slide tiếp theo.

---
## Slide 165: Định luật Amdahl áp dụng cho 3DGS
**Nội dung chính trên slide:** Công thức Speedup = 1/((1-p) + p/s); giải thích p là tỉ trọng phần được tối ưu, s là hệ số tăng tốc riêng; kết luận FastGS nên tối ưu đúng phần rasterize + backward.

**Lời thuyết trình:**
Để định lượng việc nên tối ưu ở đâu, em áp dụng định luật Amdahl kinh điển trong khoa học máy tính. Ý tưởng cốt lõi là: tăng tốc một phần của hệ thống chỉ mang lại lợi ích bị giới hạn bởi tỉ trọng thời gian mà phần đó chiếm trong toàn bộ pipeline. Công thức Speedup bằng 1 chia cho tổng của (1 trừ p) cộng với p chia s, trong đó p là tỉ trọng thời gian của phần được tối ưu, còn s là hệ số tăng tốc riêng đạt được trên phần đó. Áp dụng vào 3DGS: rasterize cộng backward chiếm tỉ trọng p lớn nhất trong toàn bộ pipeline, nên đây chính là nơi FastGS tập trung tối ưu. Ngược lại, nếu ta tối ưu một phần chỉ chiếm tỉ trọng p nhỏ, ví dụ như bước tính loss, thì dù có đạt hệ số tăng tốc s rất lớn cho riêng phần đó, speedup tổng thể của toàn hệ thống vẫn không đáng kể. Đây là lý do khoa học vì sao FastGS không cố gắng tối ưu mọi thứ dàn trải mà chọn lọc rất kỹ ba đòn bẩy cụ thể, mà em sẽ trình bày chi tiết ngay sau khi nói về mô hình chi phí ở slide kế tiếp.

---
## Slide 166: Mô hình chi phí chi tiết của một iteration
**Nội dung chính trên slide:** Công thức chi phí rasterize C_rasterize ~ O(N × T̄) và chi phí densify/prune C_densify ~ O(N); hai hướng giảm chi phí là giảm N hoặc giảm T̄.

**Lời thuyết trình:**
Để hiểu rõ hơn các đòn bẩy của FastGS, em xây dựng một mô hình chi phí toán học đơn giản cho một iteration. Gọi N là số Gaussian hiện có trong cảnh, và T_i là số tile mà Gaussian thứ i chạm tới trên ảnh, gọi là tile_touched. Chi phí rasterize tỉ lệ với tổng T_i trên toàn bộ N Gaussian, tương đương với N nhân T trung bình, tức số tile trung bình mà mỗi Gaussian chạm tới. Tương tự, chi phí densify và prune cũng tỉ lệ tuyến tính với N. Từ mô hình này, ta rút ra kết luận rất quan trọng: có hai hướng độc lập để giảm chi phí tổng thể — một là giảm N bằng cách prune hiệu quả hơn, hai là giảm T trung bình bằng cách thu nhỏ bounding box của mỗi Gaussian. Đây chính là cơ sở lý thuyết cho hai trong ba đòn bẩy tăng tốc mà FastGS sử dụng. Slide tiếp theo em sẽ giới thiệu tổng quan cả ba đòn bẩy này cùng một lúc.

---
## Slide 167: Ba đòn bẩy tăng tốc của FastGS (tổng quan)
**Nội dung chính trên slide:** Ba cột: Multi-view score (pruning thông minh hơn), Compact Box (giảm hệ số bounding-box), Sparse Adam (lịch cập nhật phân tầng, giảm tần suất optimizer.step()).

**Lời thuyết trình:**
Đây là slide tổng quan quan trọng nhất của phần 5: ba đòn bẩy tăng tốc chính của FastGS. Đòn bẩy thứ nhất là Multi-view score, đánh giá tầm quan trọng của mỗi Gaussian dựa trên đóng góp thực tế qua nhiều góc nhìn khác nhau thay vì chỉ một view, giúp pruning thông minh và chính xác hơn. Đòn bẩy thứ hai là Compact Box, giảm hệ số nhân bounding-box khi xác định vùng ảnh hưởng của Gaussian trên ảnh, từ đó giảm trực tiếp số tile mà mỗi Gaussian chạm tới. Đòn bẩy thứ ba là Sparse Adam, áp dụng lịch cập nhật phân tầng cho các tham số, giảm tần suất gọi optimizer.step() một cách có chọn lọc. Điều đáng chú ý là cả ba đòn bẩy này không phải chọn ngẫu nhiên, mà đều nhắm chính xác vào những khâu chiếm tỉ trọng lớn nhất theo phân tích Amdahl vừa trình bày: rasterize, densify/prune, và bước cập nhật tham số. Trong ba slide tiếp theo, em sẽ đi sâu vào từng đòn bẩy một, bắt đầu với Multi-view consistency score.

---
## Slide 168: Đòn bẩy 1: Multi-view consistency score
**Nội dung chính trên slide:** So sánh cách cũ (điểm quan trọng từ một view, dễ bị gradient cancellation) với FastGS (tổng hợp Contribution qua nhiều view có trọng số w_v).

**Lời thuyết trình:**
Đòn bẩy đầu tiên giải quyết một vấn đề đã nêu ở phần trước: cách tính điểm quan trọng kiểu cũ dựa trên opacity hoặc gradient tích lũy chỉ từ một view tại một thời điểm, nên rất dễ bị nhiễu bởi hiện tượng gradient cancellation — khi gradient từ các view khác nhau triệt tiêu lẫn nhau. FastGS khắc phục bằng cách tổng hợp đóng góp thực tế của mỗi Gaussian vào chất lượng ảnh qua nhiều góc camera khác nhau trước khi ra quyết định prune hay giữ. Công thức Importance của Gaussian g bằng tổng trên tất cả các view v thuộc tập Views, của trọng số w_v nhân với Contribution của g tại view đó, trong đó trọng số w_v có thể chọn đồng đều hoặc theo độ tin cậy của từng view. Cách tiếp cận này giải quyết trực tiếp vấn đề một view "khử" tín hiệu gradient của view khác, giúp việc pruning trở nên chính xác hơn nhiều, từ đó giảm N một cách hiệu quả hơn so với phương pháp cũ. Tiếp theo là đòn bẩy thứ hai, tác động trực tiếp vào chi phí rasterize.

---
## Slide 169: Đòn bẩy 2: Compact Box multiplier
**Nội dung chính trên slide:** Giảm hệ số nhân --mult từ 3.0 mặc định xuống 0.5 (cảnh nhỏ/vừa) hoặc 0.7 (cảnh lớn); diện tích ellipse giảm theo bình phương hệ số.

**Lời thuyết trình:**
Đòn bẩy thứ hai, Compact Box, là đòn bẩy rẻ nhất về mặt cài đặt nhưng lại rất hiệu quả. Nhắc lại từ phần 2, bounding-box mặc định của mỗi Gaussian được lấy theo 3 sigma để xác định vùng ảnh hưởng khi rasterize. FastGS đơn giản là giảm hệ số nhân, gọi là tham số mult, xuống còn 0,5 đối với cảnh nhỏ và vừa, hoặc 0,7 đối với cảnh lớn, thay vì giá trị mặc định 3,0. Vì diện tích ellipse chiếu lên ảnh — và do đó số tile mà Gaussian chạm tới — giảm theo bình phương của hệ số nhân, nên chỉ cần giảm hệ số này là chi phí rasterize giảm đi rất đáng kể. Bảng so sánh cho thấy rõ: với mult bằng 3,0 số tile trung bình mỗi Gaussian cao, còn với compact box mult từ 0,5 đến 0,7 thì số tile giảm hẳn, kéo theo thời gian rasterize giảm rõ rệt. Điều đặc biệt là toàn bộ cải tiến này chỉ cần đổi đúng một tham số duy nhất, không cần thay đổi kiến trúc hay thuật toán gì cả. Slide kế tiếp là đòn bẩy thứ ba, nhắm vào bước cập nhật tham số.

---
## Slide 170: Đòn bẩy 3: Sparse Adam schedule
**Nội dung chính trên slide:** Lịch step phân tầng theo bậc SH: mỗi 16 iteration đến 15k, mỗi 32 iteration đến 20k, mỗi 64 iteration sau đó; giảm mạnh số lần gọi optimizer.step().

**Lời thuyết trình:**
Đòn bẩy thứ ba là Sparse Adam schedule, nhắc lại từ phần 3 về lịch tối ưu hóa phân tầng theo bậc hệ số cầu điều hòa, gọi tắt là SH. Ý tưởng là: các hệ số SH bậc cao, mô tả chi tiết màu sắc phụ thuộc góc nhìn, biến thiên rất chậm sau giai đoạn đầu huấn luyện, nên gọi optimizer.step() dày đặc cho chúng là lãng phí tài nguyên. Cụ thể, trong 15.000 iteration đầu, các tham số SH bậc cao chỉ được cập nhật mỗi 16 iteration; từ 15.000 đến 20.000, tần suất giảm còn mỗi 32 iteration; và sau 20.000, giảm tiếp còn mỗi 64 iteration cho các tham số ít quan trọng. Bảng số liệu cho thấy rõ mức giảm: thay vì gọi optimizer 15.000 lần trong giai đoạn đầu, Sparse Adam chỉ gọi khoảng 938 lần; thay vì 5.000 lần ở giai đoạn giữa, chỉ còn khoảng 156 lần. Điều quan trọng là chất lượng ảnh gần như không đổi, trong khi số lần gọi optimizer giảm mạnh, kéo theo giảm đáng kể chi phí cập nhật tham số. Sau khi đã trình bày cả ba đòn bẩy riêng lẻ, slide tiếp theo em sẽ cho thấy hiệu quả tích lũy khi kết hợp chúng lại.

---
## Slide 171: Hiệu năng tích lũy: Biểu đồ thác nước
**Nội dung chính trên slide:** Waterfall chart minh họa tốc độ tích lũy khi lần lượt bật từng đòn bẩy: baseline → +Compact Box → +Multi-view score → +Sparse Adam → FastGS đầy đủ.

**Lời thuyết trình:**
Slide này tổng hợp hiệu quả của cả ba đòn bẩy bằng một biểu đồ thác nước, minh họa mức tăng tốc tích lũy khi bật lần lượt từng đòn bẩy lên nền baseline là 3DGS gốc. Thứ tự trình bày là: bắt đầu từ baseline, sau đó cộng thêm Compact Box, rồi cộng thêm Multi-view score, tiếp theo cộng thêm Sparse Adam, và cuối cùng là FastGS đầy đủ với cả ba đòn bẩy cùng hoạt động. Điểm mấu chốt cần nhấn mạnh là mỗi đòn bẩy đóng góp một cách độc lập vào tổng tốc độ, và khi tổng hợp cả ba lại, mức tăng tốc đạt được vượt trội hẳn so với chỉ áp dụng riêng lẻ từng đòn bẩy một. Điều này phù hợp hoàn toàn với dự đoán từ định luật Amdahl: vì ba đòn bẩy nhắm vào ba khâu chi phí khác nhau và gần như không chồng lấn nhau — rasterize, densify/prune, và optimizer step — nên hiệu quả của chúng cộng dồn lại một cách tự nhiên chứ không triệt tiêu lẫn nhau. Slide tiếp theo em sẽ chỉ rõ vị trí cụ thể của ba đòn bẩy này trong toàn bộ pipeline FastGS.

---
## Slide 172: Pipeline FastGS: ba đòn bẩy nằm ở đâu?
**Nội dung chính trên slide:** Sơ đồ pipeline với ba chú thích: Compact Box tại bước rasterize, Multi-view score tại bước densify/prune, Sparse Adam tại bước optimizer step.

**Lời thuyết trình:**
Để hình dung rõ ràng hơn, slide này đặt cả ba đòn bẩy lên đúng vị trí của chúng trong pipeline huấn luyện FastGS. Compact Box tác động tại bước rasterize, trực tiếp làm giảm số tile trung bình mà mỗi Gaussian chạm tới, từ đó giảm chi phí render. Multi-view score tác động tại bước densify/prune, giúp việc chọn lọc Gaussian nào cần giữ, nào cần xóa trở nên chính xác hơn nhờ tổng hợp thông tin từ nhiều góc nhìn. Sparse Adam tác động tại bước optimizer step, làm giảm tần suất cập nhật tham số cho những thành phần ít quan trọng. Ba đòn bẩy này không cạnh tranh tài nguyên với nhau vì chúng tác động vào ba giai đoạn hoàn toàn tách biệt trong vòng lặp huấn luyện, đó cũng chính là lý do hiệu quả của chúng cộng dồn được như biểu đồ thác nước ở slide trước đã cho thấy. Bây giờ, để khép lại toàn bộ phần 5, em sẽ tổng kết bằng một bảng so sánh tham số giữa 3DGS gốc và FastGS.

---
## Slide 173: Tổng kết: từ 3DGS gốc đến FastGS
**Nội dung chính trên slide:** Bảng so sánh các tham số then chốt giữa 3DGS gốc và FastGS (loss_thresh, grad_abs_thresh, --dense, --highfeature_lr, --mult); kết luận FastGS tối ưu tham số hóa và lịch huấn luyện, giữ nguyên chất lượng.

**Lời thuyết trình:**
Slide cuối cùng của phần 5 tổng kết lại toàn bộ hành trình từ 3DGS gốc đến FastGS thông qua bảng so sánh các tham số then chốt. Ba DGS gốc không có loss_thresh, trong khi FastGS thêm mới với giá trị 0,1. Về gradient, 3DGS gốc dùng gradient thường, còn FastGS dùng Abs-GS với ngưỡng 0,0012 để khắc phục vấn đề gradient cancellation đã nói ở các phần trước. Tham số dense cố định ở 3DGS gốc, nhưng ở FastGS dao động từ 0,001 đến 0,013 tùy theo kích thước cảnh. Learning rate cho tham số bậc cao cũng được điều chỉnh xuống còn 0,005 đến 0,02, chia thêm 20 lần trong code thực tế. Và cuối cùng, hệ số nhân bounding-box mult giảm từ 3,0 xuống còn 0,5 đến 0,7 như đã trình bày. Điểm mấu chốt cần nhấn mạnh là: FastGS hoàn toàn không thay đổi mô hình biểu diễn cảnh bằng Gaussian, mà chỉ tối ưu tham số hóa và lịch huấn luyện dựa trên mô hình chi phí và định luật Amdahl đã phân tích. Kết quả cuối cùng là tăng tốc huấn luyện đáng kể trong khi chất lượng tái tạo đo bằng PSNR và SSIM gần như không đổi so với bản gốc. Đến đây em xin kết thúc phần trình bày của mình, xin cảm ơn hội đồng đã lắng nghe.

---
