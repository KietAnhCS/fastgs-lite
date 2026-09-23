# Kịch bản thuyết trình 45 phút — 3D Gaussian Splatting & FastGS-lite (41 slide)


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


---
## Slide 9: Compact Box — đòn bẩy tăng tốc #1 của FastGS
**Lời thuyết trình (rút gọn):**
Sau khi chiếu Gaussian 3D thành ellipse 2D, hệ thống cần một bounding-box bao quanh để biết vùng ảnh hưởng trên ảnh. Cài đặt gốc dùng bounding-box 3-sigma — an toàn về thống kê nhưng khá rộng, khiến Gaussian chạm nhiều tile hơn mức cần thiết, gây lãng phí tính toán. FastGS thay bằng compact box nhỏ gọn hơn, điều chỉnh qua tham số mult, mặc định 0.5, có thể tăng lên 0.7 cho cảnh lớn để tránh cắt mất phần đuôi Gaussian quan trọng. Kết quả là giảm đáng kể số tile mỗi Gaussian phải rasterize, tức giảm tải tính toán mà không đánh đổi nhiều chất lượng hình ảnh. Đây là đòn bẩy tăng tốc đầu tiên và quan trọng nhất của FastGS. Tiếp theo, ta xem công thức alpha blending — bước kết hợp các Gaussian này thành màu pixel cuối cùng.

---
## Slide 10: Alpha blending (front-to-back)
**Lời thuyết trình (rút gọn):**
Đây là công thức quyết định màu cuối cùng của mỗi pixel: C bằng tổng của ci nhân alpha_i nhân Ti, theo đúng thứ tự độ sâu đã sắp xếp. Trong đó ci là màu Gaussian giải mã từ hệ số SH, alpha_i là độ mờ tại điểm đó, còn Ti là transmittance — độ truyền sáng còn lại, bằng tích của (1 trừ alpha_j) cho mọi Gaussian j gần camera hơn i. Nói cách khác, Ti đo "còn bao nhiêu ánh sáng lọt qua" sau khi các lớp phía trước đã hấp thụ một phần. Gaussian càng gần camera thì đóng góp càng lớn vào màu cuối cùng, vì chưa bị lớp nào che khuất. Đây là mô hình front-to-back alpha compositing kinh điển. Vì Ti giảm theo cấp số nhân khi có nhiều lớp che, khi Ti đủ nhỏ, rasterizer có thể dừng sớm (early stopping) để tiết kiệm tính toán mà không ảnh hưởng chất lượng. Slide sau sẽ tổng kết toàn bộ phương trình render.

---
## Slide 11: Tổng kết phương trình render
**Lời thuyết trình (rút gọn):**
Slide này tổng kết toàn bộ phần chiếu phối cảnh và rasterizer bằng một phương trình duy nhất: C tại p bằng tổng ci nhân alpha_i nhân tích (1 trừ alpha_j) với j nhỏ hơn i — chính là công thức blend vừa nêu, viết gọn thành phương trình render hoàn chỉnh cho cả pipeline. Điều cốt lõi là toàn bộ phương trình này khả vi theo mọi tham số: vị trí tâm mu, hiệp phương sai Sigma, độ mờ alpha, và hệ số SH quyết định màu. Nhờ tính khả vi đầy đủ này, ta huấn luyện trực tiếp bằng gradient descent: so sánh ảnh render với ảnh RGB thực tế, rồi lan truyền ngược sai số qua toàn bộ chuỗi Gaussian 3D, chiếu 2D, trường alpha, blend, để cập nhật từng tham số. Đây là điểm khép lại phần render, và cũng là lúc chuyển sang phần tiếp theo: hàm mất mát và cách tối ưu các tham số này.

---
## Slide 12: Hàm mất mát: kết hợp L1 và D-SSIM
**Lời thuyết trình (rút gọn):**
Chuyển sang phần huấn luyện: hàm mất mát dùng để tối ưu tham số Gaussian. Loss kết hợp hai thành phần bổ trợ nhau. Thứ nhất là L1, đo sai khác trung bình theo từng pixel giữa ảnh render và ảnh gốc — tín hiệu cơ bản, đơn giản nhưng nhạy với nhiễu. Thứ hai là D-SSIM, bằng 1 trừ SSIM, phản ánh sai lệch về cấu trúc chứ không chỉ độ sáng từng điểm. Công thức tổng là trung bình có trọng số: L bằng (1 trừ lambda) nhân L1 cộng lambda nhân (1 trừ SSIM). Giá trị lambda_dssim mặc định của 3DGS gốc là 0.2, ưu tiên L1 nhiều hơn; FastGS-lite tăng lên 0.25 để nhấn mạnh cấu trúc, giữ chi tiết hình học tốt hơn khi giảm số Gaussian. Tiếp theo ta sẽ đi sâu vào cơ chế của SSIM.


---
## Slide 13: Gradient tập trung tại biên Gaussian
**Lời thuyết trình (rút gọn):**
Một quan sát quan trọng: gradient trong không gian view không phân bố đều trên Gaussian mà tập trung mạnh nhất ở vùng biên — nơi ranh giới với nền hoặc Gaussian khác, cũng là nơi ảnh render dễ lệch nhiều nhất so với ảnh thật. Ý nghĩa thực tiễn rất lớn: gradient tại biên chính là tín hiệu để hệ thống quyết định có densify hay không, tức có thêm Gaussian mới vào vùng đó không, ở bước Adaptive Density Control. Ngược lại, vùng gradient thấp nghĩa là Gaussian đã khớp tốt với dữ liệu, không cần chia tách thêm. Đây là nền tảng kết nối phần loss, gradient với phần điều chỉnh mật độ Gaussian. Tiếp theo, ta tổng kết lại toàn bộ vòng lặp huấn luyện trước khi đi sâu vào phần densify/prune.

---
## Slide 14: Tổng kết vòng lặp huấn luyện
**Lời thuyết trình (rút gọn):**
Toàn bộ vòng lặp huấn luyện gồm năm bước lặp lại liên tục: Render ảnh từ tập Gaussian hiện tại theo góc camera; Tính loss kết hợp L1 và D-SSIM; Backward — lan truyền ngược gradient qua rasterizer khả vi về từng tham số; optimizer.step() dùng sparse Adam để cập nhật tham số hiệu quả; và cuối cùng, định kỳ chứ không phải mọi iteration, là Densify/Prune — điều chỉnh mật độ Gaussian. Sau bước năm, vòng lặp quay lại bước một, lặp hàng chục nghìn iteration tới khi hội tụ. Đây là bước chuyển rất quan trọng: từ đây, em sẽ đi sâu vào chính phần trọng tâm của đồ án — Adaptive Density Control, cơ chế quyết định trực tiếp chất lượng và hiệu năng của FastGS-lite.

---
## Slide 15: Tổng quan Adaptive Density Control (ADC)
**Lời thuyết trình (rút gọn):**
ADC giải quyết vấn đề: point cloud khởi tạo từ COLMAP thường không hoàn hảo — có vùng quá thưa, thiếu điểm ở khu vực chi tiết phức tạp, trong khi có Gaussian quá to che khuất chi tiết nhỏ. ADC chạy định kỳ sau mỗi khoảng densification_interval iteration, điều chỉnh động mật độ Gaussian theo độ phức tạp hình học thực tế của scene. ADC gồm hai nhóm thao tác: Densify, gồm clone (nhân bản) và split (tách), dùng để thêm Gaussian ở nơi thiếu hình học; và Prune, cắt tỉa Gaussian thừa, mờ hoặc không còn đóng góp giá trị. Mục tiêu là cân bằng chất lượng tái tạo với số lượng Gaussian, vì số lượng lớn ảnh hưởng trực tiếp tới bộ nhớ và tốc độ render. Tiếp theo, ta xem tín hiệu cụ thể để quyết định densify: gradient view-space tích lũy.

---
## Slide 16: Tích luỹ gradient view-space
**Lời thuyết trình (rút gọn):**
Để quyết định Gaussian nào cần densify, hệ thống cần tín hiệu định lượng: gradient view-space tích lũy qua thời gian. Với mỗi Gaussian, gradient vị trí 2D sinh ra từ mỗi lần render ở góc camera khác nhau được cộng dồn lại, đồng thời đếm denom — số lần Gaussian thực sự được nhìn thấy. Gradient trung bình g-bar bằng tổng gradient view-space chia cho denom. Nếu g-bar lớn, nghĩa là Gaussian liên tục gây sai số đáng kể qua nhiều góc nhìn, nó trở thành ứng viên clone hoặc split. Cách tích lũy qua nhiều view giúp tín hiệu ổn định hơn so với dùng một lần render đơn lẻ. Tuy nhiên, chính cách lấy trung bình vector này cũng tiềm ẩn một vấn đề quan trọng — gradient cancellation — mà slide tiếp theo sẽ phân tích.


---
## Slide 17: Vấn đề triệt tiêu gradient (gradient cancellation)
**Lời thuyết trình (rút gọn):**
Đây là một hạn chế đã biết của 3DGS gốc: khi lấy trung bình gradient từ nhiều góc nhìn, nếu các vector này ngược hướng nhau, chúng có thể triệt tiêu lẫn nhau thay vì cộng dồn. Hệ quả là dù mỗi góc nhìn riêng lẻ đều báo Gaussian đó có sai số lớn, gradient trung bình tổng hợp lại có thể rất nhỏ — Gaussian bị bỏ sót khỏi danh sách densify dù đáng lẽ cần xử lý. Đây là hạn chế nội tại của việc chỉ dùng gradient trung bình làm tín hiệu duy nhất. FastGS khắc phục bằng multi-view consistency score, tức điểm nhất quán đa góc nhìn, sẽ trình bày kỹ hơn ở phần sau. Tiếp theo, chúng ta xem cơ chế densify với hai thao tác cụ thể: clone và split.

---
## Slide 18: Clone (nhân bản)
**Lời thuyết trình (rút gọn):**
Clone áp dụng cho Gaussian nhỏ nhưng thiếu hình học — vùng đó cần thêm chi tiết, nhưng bản thân Gaussian đã đủ nhỏ nên không cần tách. Cơ chế: nhân đôi Gaussian, bản sao đặt lệch một khoảng nhỏ theo đúng hướng gradient vị trí, tức về phía sai số lớn nhất. Quan trọng là clone không đổi kích thước hay ma trận hiệp phương sai — cả bản gốc lẫn bản sao giữ nguyên hình dạng. Mục đích duy nhất là tăng mật độ điểm để lấp đầy vùng thiếu chi tiết, như thêm quân tiếp viện vào đúng khu vực thiếu người chứ không thay đổi cấu trúc đã có. Tiếp theo là thao tác còn lại trong nhóm densify: split.

---
## Slide 19: Split (tách)
**Lời thuyết trình (rút gọn):**
Ngược với clone, split áp dụng cho Gaussian quá to, thuộc trường hợp over-reconstruction — một Gaussian đang cố biểu diễn vùng quá nhiều chi tiết so với kích thước của nó. Khi split, Gaussian cha tách thành hai Gaussian con nhỏ hơn, scale giảm theo hệ số cố định. Điểm đáng chú ý: vị trí các Gaussian con không cố định mà được lấy mẫu ngẫu nhiên theo phân phối chuẩn, xây dựng dựa trên chính ma trận hiệp phương sai của Gaussian cha — nên các điểm con phân bố theo đúng hình dạng, hướng và độ trải rộng mà cha vốn biểu diễn, tránh lệch lạc ngẫu nhiên. Sau khi có clone và split để tăng mật độ, ta chuyển sang chiều ngược lại: prune, để cắt giảm Gaussian dư thừa.

---
## Slide 20: Prune theo độ quan trọng đa góc nhìn (FastGS)
**Lời thuyết trình (rút gọn):**
Đây là một đóng góp chính của FastGS. Trong 3DGS gốc, prune chỉ dựa trên ngưỡng opacity thấp hoặc kích thước quá lớn — cách này đơn giản nhưng không phản ánh đúng mức đóng góp thực tế của từng Gaussian vào chất lượng ảnh. FastGS cải tiến bằng cách tính điểm importance cho mỗi Gaussian, dựa trên đóng góp thực tế vào chất lượng render, và tổng hợp điểm này qua nhiều góc nhìn khác nhau chứ không chỉ một góc. Gaussian có importance thấp nhất bị cắt trước tiên, bất kể opacity hay kích thước — ngay cả Gaussian opacity cao nhưng đóng góp thực tế thấp vẫn có thể bị loại. Kết quả: mô hình gọn hơn đáng kể về số lượng Gaussian nhưng vẫn giữ chất lượng tái tạo — đây là một trong những đòn bẩy chính giúp FastGS tăng tốc độ render. Tiếp theo, ta xem số lượng Gaussian biến đổi ra sao theo thời gian huấn luyện.


---
## Slide 21: Reset opacity định kỳ
**Lời thuyết trình (rút gọn):**
Cơ chế cuối trong Adaptive Density Control: cứ mỗi opacity_reset_interval, mặc định 3000 iteration, toàn bộ opacity của mọi Gaussian bị đặt lại về giá trị thấp qua sigmoid-inverse. Mục đích là buộc optimizer đánh giá lại vai trò từng Gaussian, thay vì để những Gaussian đã đạt opacity cao giữ nguyên trạng thái dù không còn cần thiết. Sau reset, Gaussian thực sự quan trọng sẽ nhanh chóng tăng opacity trở lại nhờ gradient; Gaussian dư thừa không kịp hồi phục sẽ bị prune ở vòng kế tiếp. Đây là bước dọn dẹp định kỳ giúp scene không tích tụ Gaussian "chết" theo thời gian.

Đến đây, em xin kết thúc phần tổng quan Loss, Gradient, Optimizer và ADC. Bây giờ ta đi sâu vào cách tính điểm Importance đa view — đóng góp cốt lõi của FastGS-lite.

---
## Slide 22: Tổng hợp counts qua nhiều view: đầu vào cho Importance
**Lời thuyết trình (rút gọn):**
Sau khi có counts cho từng Gaussian ở từng view riêng lẻ, bước tiếp theo là tổng hợp lại. Mỗi Gaussian có một vector counts ứng với các view nó xuất hiện. Lưu ý: tổng counts trên một view luôn ≥ số pixel lỗi thật của view đó, vì một pixel lỗi có thể nằm trong footprint của nhiều Gaussian chồng lấn nên bị đếm lặp — điều này có chủ đích, vì mục tiêu là quy trách nhiệm cho tất cả Gaussian liên quan, không chia đều lỗi. Về cài đặt, mỗi lần render một view, kernel CUDA cộng dồn atomic vào tensor metricCount, rồi lũy kế qua tất cả view thành full_metric_counts. Vector counts tổng hợp qua toàn bộ V view này chính là đầu vào trực tiếp cho bước tính Importance.

---
## Slide 23: Importance score: trung bình counts qua các view, làm tròn xuống
**Lời thuyết trình (rút gọn):**
Đây là công thức trung tâm, khớp với hàm compute_gaussian_score_fastgs trong code. Importance của một Gaussian bằng trung bình cộng counts qua tất cả view, rồi làm tròn xuống (torch.div chế độ floor). Một Gaussian được coi là "quan trọng" — giữ lại hoặc ưu tiên densify — khi Importance vượt quá 5. Dùng trung bình thay vì max hay tổng có lý do: nó đảm bảo Gaussian phải nhất quán gây lỗi qua nhiều góc nhìn, chứ không chỉ nổi bật ở một view đơn lẻ, mới được coi là thực sự quan trọng — tránh việc một góc nhìn bất thường quyết định số phận Gaussian.

Từ điểm Importance này, ta chuyển sang xem nó được dùng ở đâu trong quyết định densify.

---
## Slide 24: Sơ đồ quyết định densify đầy đủ
**Lời thuyết trình (rút gọn):**
Đây là sơ đồ tổng hợp toàn bộ logic densify của FastGS-lite, áp dụng mỗi 100-500 vòng lặp, trong khoảng iteration 500-15000. Bước một: dựa vào kích thước Gaussian so với ngưỡng để chọn nhánh. Nếu nhỏ — "under-reconstruction" — thì kiểm tra gradient có dấu vượt ngưỡng và Importance > 5; nếu đúng cả hai thì CLONE, tạo bản sao giữ nguyên tham số gốc. Nếu to — "over-reconstruction" — thì kiểm tra gradient trị tuyệt đối vượt ngưỡng cao hơn và Importance > 5; nếu đúng thì SPLIT thành hai Gaussian con, xóa Gaussian gốc. Điểm mấu chốt: nếu Importance không thỏa, 3DGS gốc vẫn densify chỉ cần gradient đạt, nhưng FastGS-lite kiên quyết chặn lại — đây chính là đóng góp cốt lõi giúp tránh sinh thừa Gaussian không cần thiết.


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


---
## Slide 29: Tổng kết 5 cải tiến của FastGS-lite trong ADC
**Lời thuyết trình (rút gọn):**
Em tổng kết nhanh 5 cải tiến FastGS-lite đưa vào chu trình ADC so với 3DGS gốc. Một: quyết định densify không chỉ dựa gradient mà AND thêm với Importance đa view, lọc bớt ứng viên sai. Hai: split dùng tổng trị tuyệt đối gradient thay vì tổng có dấu, tránh bị triệt tiêu khi Gaussian đúng vị trí nhưng sai kích thước. Ba: bước prune không xoá cứng toàn bộ ứng viên mà rút mẫu multinomial theo trọng số, giữ lại phần còn hữu ích. Bốn: chặn trần opacity ở 0,8 để không "chôn" gradient của các lớp Gaussian phía sau. Và cuối cùng, cải tiến năm: thêm bước final prune sau khi densify kết thúc, tại t=1000 và t=1100, xoá theo alpha thấp hoặc xác suất Pruning cao — tạo ra hai bậc thang giảm N rõ rệt mà 3DGS gốc không có. Kết quả chung: loss L1 tương đương 3DGS gốc nhưng N nhỏ hơn hẳn — mô hình gọn nhất trong ba kịch bản so sánh. Tiếp theo, em chuyển sang phần ba đòn bẩy tăng tốc của FastGS.

---
## Slide 30: Vì sao 3DGS gốc chậm?
**Lời thuyết trình (rút gọn):**
Trước khi nói về tăng tốc, em phân tích 3DGS gốc chậm ở đâu. Mỗi iteration gồm bốn khâu: render, tính loss, backward, và định kỳ densify/prune. Với cảnh thực có hàng trăm nghìn đến hàng triệu Gaussian, ước tính render chiếm khoảng 45% thời gian, backward khoảng 35%, tính loss chỉ 10%, còn densify/prune tuy chỉ khoảng 10% tổng thời gian nhưng chạy định kỳ và gây đột biến chi phí mỗi lần kích hoạt. Nhìn vào tỉ trọng này, rõ ràng muốn tăng tốc hiệu quả thì phải nhắm đúng hai khâu chiếm phần lớn nhất là render và backward, thay vì tối ưu dàn trải. Đây chính là tiền đề để áp dụng định luật Amdahl: tăng tốc một phần hệ thống chỉ có lợi tương ứng với tỉ trọng thời gian phần đó chiếm trong toàn pipeline. Dựa trên phân tích này, FastGS chọn lọc đúng ba đòn bẩy tăng tốc, em sẽ giới thiệu tổng quan ngay sau đây.

---
## Slide 31: Ba đòn bẩy tăng tốc của FastGS (tổng quan)
**Lời thuyết trình (rút gọn):**
Đây là slide tổng quan quan trọng nhất của phần này: ba đòn bẩy tăng tốc chính của FastGS. Một, Multi-view score: đánh giá tầm quan trọng của Gaussian qua nhiều góc nhìn thay vì một view, giúp pruning chính xác hơn, giảm N hiệu quả hơn. Hai, Compact Box: giảm hệ số nhân bounding-box từ 3,0 mặc định xuống 0,5-0,7, khiến số tile mỗi Gaussian chạm tới giảm theo bình phương hệ số, kéo giảm trực tiếp chi phí rasterize. Ba, Sparse Adam: áp dụng lịch cập nhật phân tầng, giảm tần suất gọi optimizer.step() cho các tham số ít quan trọng như SH bậc cao. Cả ba đòn bẩy đều nhắm đúng những khâu chiếm tỉ trọng lớn theo phân tích Amdahl vừa nêu — rasterize, densify/prune, và bước cập nhật tham số — và gần như không chồng lấn nhau. Slide tiếp theo em sẽ cho thấy hiệu quả tích luỹ khi kết hợp cả ba.

---
## Slide 32: Hiệu năng tích luỹ — Biểu đồ thác nước
**Lời thuyết trình (rút gọn):**
Slide này tổng hợp hiệu quả ba đòn bẩy bằng biểu đồ thác nước, thể hiện mức tăng tốc tích luỹ khi bật lần lượt từng đòn bẩy lên nền baseline 3DGS gốc: bắt đầu từ baseline, cộng Compact Box, cộng tiếp Multi-view score, cộng tiếp Sparse Adam, và cuối cùng là FastGS đầy đủ với cả ba cùng hoạt động. Điểm mấu chốt là mỗi đòn bẩy đóng góp tăng tốc độc lập, và khi kết hợp cả ba thì mức tăng tốc tổng vượt trội hẳn so với dùng riêng lẻ. Điều này khớp hoàn toàn với dự đoán từ định luật Amdahl: vì ba đòn bẩy tác động vào ba khâu chi phí khác nhau, gần như không chồng lấn — rasterize, densify/prune, optimizer step — nên hiệu quả cộng dồn tự nhiên chứ không triệt tiêu lẫn nhau. Đây là bằng chứng thực nghiệm cho toàn bộ chiến lược thiết kế của FastGS.


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

