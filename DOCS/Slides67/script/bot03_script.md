# Kịch bản thuyết trình — Bot 03 (Slide 40–59)
## Phạm vi: Phần 3 (Loss, Gradient & Optimizer), Phần 4 (Adaptive Density Control — Tổng quan)

---

## Slide 40: Hàm mất mát: kết hợp L1 và D-SSIM
**Nội dung chính trên slide:** Loss huấn luyện = kết hợp L1 (sai khác pixel) và D-SSIM (sai khác cấu trúc), công thức $\mathcal{L} = (1-\lambda)\mathcal{L}_1 + \lambda(1-\text{SSIM})$, hệ số lambda_dssim mặc định 0.2, FastGS-lite dùng 0.25.

**Lời thuyết trình:**
Bây giờ em xin chuyển sang phần huấn luyện, cụ thể là hàm mất mát — hay loss function — dùng để tối ưu các tham số Gaussian. Loss huấn luyện trong 3D Gaussian Splatting không chỉ dùng một thành phần duy nhất mà kết hợp hai thành phần bổ trợ nhau. Thứ nhất là L1, đo sai khác trung bình theo từng pixel giữa ảnh render và ảnh gốc — đây là tín hiệu cơ bản, đơn giản nhưng rất nhạy với nhiễu. Thứ hai là D-SSIM, bằng 1 trừ SSIM, phản ánh sai lệch về mặt cấu trúc chứ không chỉ độ sáng từng điểm ảnh. Công thức tổng là trung bình có trọng số giữa hai thành phần này, với hệ số lambda gọi là lambda_dssim. Giá trị mặc định của 3DGS gốc là 0.2, nghĩa là ưu tiên L1 nhiều hơn; còn trong FastGS-lite, nhóm em tăng lên 0.25 để nhấn mạnh hơn vào cấu trúc, giúp giữ chi tiết hình học tốt hơn khi giảm số Gaussian. Tiếp theo, em sẽ giải thích rõ hơn cơ chế của SSIM.

---
## Slide 41: SSIM: so sánh theo cửa sổ trượt
**Nội dung chính trên slide:** SSIM so sánh theo cửa sổ trượt (thường 11x11), kết hợp 3 yếu tố: luminance, contrast, structure; nhạy với sai lệch cấu trúc cục bộ hơn L1/L2.

**Lời thuyết trình:**
SSIM khác biệt căn bản so với L1 hay L2 ở chỗ nó không so sánh từng pixel riêng lẻ, mà trượt một cửa sổ nhỏ, thường có kích thước 11 nhân 11, qua toàn bộ ảnh render và ảnh gốc để so sánh theo từng vùng cục bộ. Trong mỗi cửa sổ đó, SSIM kết hợp ba yếu tố: độ sáng trung bình, độ tương phản, và cấu trúc — tức là mối tương quan giữa các giá trị pixel sau khi đã chuẩn hóa độ sáng và tương phản. Nhờ cách tiếp cận theo cửa sổ này, SSIM nhạy hơn hẳn với các sai lệch về cấu trúc cục bộ, ví dụ như răng cưa ở viền vật thể hay mất chi tiết texture, những lỗi mà L1 hay L2 thuần túy khó phát hiện vì chúng chỉ nhìn từng điểm ảnh độc lập. Chính vì tính bổ trợ này mà SSIM, thông qua thành phần D-SSIM, được kết hợp cùng L1 trong hàm loss tổng, giúp mô hình vừa khớp giá trị pixel vừa giữ được cấu trúc hình ảnh. Sau khi hiểu về SSIM, ta xem tiếp một metric per-pixel khác là PSNR.

---
## Slide 42: PSNR: metric per-pixel
**Nội dung chính trên slide:** PSNR đo bằng dB dựa trên MSE, công thức $\text{PSNR}=10\log_{10}(\text{MAX}^2/\text{MSE})$; PSNR cao = sai số nhỏ; nhược điểm là không phản ánh tốt cảm nhận thị giác.

**Lời thuyết trình:**
PSNR, viết tắt của Peak Signal-to-Noise Ratio, là một metric đánh giá chất lượng ảnh rất phổ biến, đo bằng đơn vị decibel và dựa trực tiếp trên MSE — sai số bình phương trung bình giữa ảnh render và ảnh gốc. Công thức là 10 nhân log cơ số 10 của MAX bình phương chia cho MSE, trong đó MAX là giá trị pixel lớn nhất có thể, thường là 255 hoặc 1 tùy chuẩn hóa. Vì nằm trong mẫu số theo dạng nghịch đảo và logarit, PSNR càng cao thì MSE càng nhỏ, tức là sai số giữa hai ảnh càng thấp, chất lượng tái tạo càng tốt. Tuy nhiên, nhược điểm của PSNR là nó thuần túy đo theo pixel, coi mọi sai lệch có cùng độ lớn là như nhau về mặt cảm nhận, trong khi mắt người lại nhạy cảm hơn với một số loại sai lệch, ví dụ như mất cấu trúc, hơn là sai lệch màu sắc nhỏ đồng đều. Vì vậy PSNR không phản ánh tốt chất lượng cảm nhận thị giác như SSIM hay LPIPS. Điều này dẫn tới nhu cầu có một điểm số tổng hợp cân bằng cả ba metric, mà em sẽ trình bày ở slide tiếp theo.

---
## Slide 43: Điểm số tổng hợp (Score) đánh giá benchmark
**Nội dung chính trên slide:** Score = 0.4(1-LPIPS) + 0.3·SSIM + 0.3·clamp(PSNR/PSNR_max, 0, 1), PSNR_max = 30dB; cân bằng giữa độ chính xác pixel, cấu trúc, cảm nhận thị giác.

**Lời thuyết trình:**
Vì mỗi metric riêng lẻ — PSNR, SSIM, LPIPS — chỉ phản ánh một khía cạnh chất lượng, các benchmark và cuộc thi về 3D Gaussian Splatting thường dùng một điểm số tổng hợp, gọi là Score, để đánh giá công bằng hơn. Công thức Score kết hợp ba thành phần theo trọng số: 0.4 nhân với 1 trừ LPIPS — LPIPS là metric dựa trên mạng neural đo khoảng cách cảm nhận, nên LPIPS càng thấp càng tốt, do đó lấy 1 trừ đi để điểm càng cao càng tốt; cộng 0.3 nhân SSIM, phản ánh cấu trúc; và cộng 0.3 nhân giá trị PSNR đã chuẩn hóa, chia cho PSNR tối đa quy ước là 30 decibel và giới hạn trong khoảng 0 đến 1 bằng hàm clamp để tránh điểm vượt quá thang đo. Mục tiêu của công thức này là cân bằng giữa ba khía cạnh: độ chính xác pixel qua PSNR, cấu trúc qua SSIM, và cảm nhận thị giác qua LPIPS, tránh trường hợp một mô hình chỉ tối ưu tốt một metric nhưng lại kém ở các mặt khác. Sau khi đã nắm rõ các thước đo chất lượng, ta chuyển sang cơ chế lan truyền gradient để hiểu quá trình tối ưu diễn ra như thế nào.

---
## Slide 44: Lan truyền gradient qua rasterizer khả vi
**Nội dung chính trên slide:** Rasterizer 3DGS hoàn toàn khả vi; chuỗi backprop từ pixel qua alpha blend, alpha field, covariance chiếu về (μ, Σ); chain rule tổng quát cho mỗi tham số θ.

**Lời thuyết trình:**
Một điểm mấu chốt khiến 3D Gaussian Splatting tối ưu được bằng gradient descent là toàn bộ phép render — từ chiếu Gaussian 3D xuống mặt phẳng ảnh 2D cho đến alpha blending — đều là các phép toán khả vi, nghĩa là có thể tính đạo hàm liên tục. Nhờ vậy, ta có thể lan truyền ngược, hay backprop, từ giá trị loss tính trên ảnh render, đi ngược trở lại từng tham số của từng Gaussian. Chuỗi lan truyền cụ thể đi từ pixel, qua bước alpha blend, tới alpha field, tới covariance đã chiếu xuống 2D, rồi cuối cùng về lại cặp tham số gốc mu và sigma trong không gian 3D. Về mặt toán học, đây chính là quy tắc chuỗi quen thuộc: đạo hàm của loss theo một tham số theta bất kỳ, có thể là vị trí mu, ma trận hiệp phương sai sigma, độ mờ alpha, hay hệ số cầu điều hòa SH, đều bằng đạo hàm của loss theo màu pixel C nhân với đạo hàm của C theo theta. Nhờ cơ chế này, mọi tham số Gaussian đều được cập nhật trực tiếp từ tín hiệu sai số ở không gian ảnh. Tiếp theo, ta sẽ xem gradient này phân bố tập trung ở đâu trên mỗi Gaussian.

---
## Slide 45: Gradient tập trung tại biên Gaussian
**Nội dung chính trên slide:** Gradient view-space lớn nhất ở vùng biên Gaussian — nơi sai số cao nhất; đây là tín hiệu quyết định densify; vùng gradient thấp nghĩa là đã khớp tốt.

**Lời thuyết trình:**
Một quan sát quan trọng khi phân tích gradient trong quá trình huấn luyện là gradient trong không gian view, tức không gian ảnh chiếu, không phân bố đều trên toàn bộ Gaussian mà tập trung mạnh nhất ở vùng biên, hay rìa, của Gaussian đó. Điều này khá trực quan: vùng biên là nơi ranh giới giữa Gaussian này với nền hoặc với Gaussian khác, nên cũng chính là nơi ảnh render dễ lệch nhiều nhất so với ảnh thật, tạo ra sai số cao. Ý nghĩa thực tiễn của quan sát này rất lớn, vì gradient tại biên chính là tín hiệu được hệ thống dùng để quyết định có nên densify hay không, tức là có nên thêm Gaussian mới vào vùng đó hay không, ở giai đoạn Adaptive Density Control mà chúng ta sẽ tìm hiểu kỹ ở phần sau. Ngược lại, những vùng có gradient thấp cho thấy Gaussian ở đó đã khớp tốt với dữ liệu thật, không cần chia tách hay bổ sung thêm. Đây chính là nền tảng lý thuyết kết nối phần loss và gradient với phần điều chỉnh mật độ Gaussian. Bây giờ, ta sẽ nói về cách các gradient này được dùng để cập nhật tham số thông qua optimizer Adam.

---
## Slide 46: Adam optimizer cho từng tham số Gaussian
**Nội dung chính trên slide:** Mỗi tham số Gaussian có trạng thái Adam riêng (moment bậc 1, bậc 2); learning rate thích ứng theo từng chiều; hội tụ mượt hơn SGD thuần.

**Lời thuyết trình:**
Sau khi có gradient, bước tiếp theo là cập nhật tham số, và 3D Gaussian Splatting sử dụng optimizer Adam thay vì SGD thông thường. Điểm đặc biệt là mỗi tham số của mỗi Gaussian — bao gồm vị trí mu, hiệp phương sai sigma, độ mờ alpha, và hệ số cầu điều hòa SH — đều có trạng thái Adam riêng biệt, được lưu và cập nhật độc lập. Trạng thái này gồm hai đại lượng: moment bậc một, là trung bình động của gradient, giúp giữ hướng cập nhật ổn định; và moment bậc hai, là trung bình động của bình phương gradient, dùng để điều chỉnh biên độ bước cập nhật theo từng chiều tham số. Nhờ cơ chế này, learning rate trở nên thích ứng theo từng chiều, tham số nào có gradient dao động lớn sẽ được điều chỉnh bước nhỏ lại, còn tham số ổn định thì được cập nhật nhanh hơn. Kết quả là quỹ đạo hội tụ mượt hơn nhiều so với SGD thuần, tránh được hiện tượng dao động mạnh khi gradient bị nhiễu — điều rất dễ xảy ra trong bài toán này vì gradient được tính từ nhiều góc camera khác nhau. Tiếp theo là cách learning rate thay đổi theo thời gian huấn luyện.

---
## Slide 47: Lịch suy giảm learning rate
**Nội dung chính trên slide:** Learning rate của vị trí mu giảm theo hàm mũ qua position_lr_max_steps; mục đích: học nhanh đầu, ổn định tinh chỉnh cuối.

**Lời thuyết trình:**
Bên cạnh việc mỗi tham số có trạng thái Adam riêng, learning rate của từng nhóm tham số — đặc biệt là learning rate cho vị trí mu — còn được điều chỉnh theo một lịch suy giảm dạng hàm mũ, gọi là exponential decay. Quá trình suy giảm này diễn ra trong suốt số bước được quy định bởi tham số position lr max steps. Lý do thiết kế như vậy xuất phát từ đặc điểm hai giai đoạn của quá trình huấn luyện: ở giai đoạn đầu, các Gaussian còn ở xa vị trí tối ưu, cần học nhanh với bước cập nhật lớn để nhanh chóng đưa hình học về đúng khung hình cơ bản. Nhưng càng về sau, khi Gaussian đã gần vị trí đúng, nếu vẫn giữ bước lớn sẽ dễ gây dao động, mất ổn định, thậm chí làm hỏng kết quả đã đạt được. Vì vậy learning rate giảm dần để cho phép các bước tinh chỉnh nhỏ, ổn định hơn về cuối quá trình huấn luyện, giúp mô hình hội tụ chính xác đến chi tiết nhỏ. Đây là một kỹ thuật lập lịch rất phổ biến trong deep learning nói chung, được áp dụng phù hợp vào bài toán tối ưu hình học 3D này. Tiếp theo, chúng ta sẽ xem một cải tiến quan trọng về tốc độ: Sparse Adam schedule.

---
## Slide 48: Sparse Adam schedule (đòn bẩy tăng tốc #3)
**Nội dung chính trên slide:** features_rest chỉ step mỗi 16 iteration đến 15k; sau đó mọi tham số step mỗi 32 iter (15k-20k), mỗi 64 iter (sau 20k); giảm chi phí optimizer.step().

**Lời thuyết trình:**
Đây là đòn bẩy tăng tốc thứ ba mà nhóm em muốn nhấn mạnh: Sparse Adam schedule. Ý tưởng cốt lõi là không phải mọi tham số đều cần được cập nhật ở mọi iteration. Cụ thể, features_rest — tức là các hệ số cầu điều hòa SH bậc cao, chi phối màu sắc thay đổi theo góc nhìn, gọi là view-dependent color — chỉ được step, tức cập nhật, mỗi 16 iteration một lần, cho tới mốc 15 nghìn iteration. Sau mốc đó, toàn bộ tham số, không chỉ SH bậc cao, chuyển sang chế độ step thưa: mỗi 32 iteration trong khoảng từ 15 nghìn đến 20 nghìn, rồi giãn ra mỗi 64 iteration sau mốc 20 nghìn. Lý do kỹ thuật này khả thi là vì thành phần SH bậc cao thường thay đổi rất chậm trong quá trình huấn luyện, không cần cập nhật liên tục ở mọi bước mà vẫn không ảnh hưởng đáng kể tới chất lượng. Kết quả là chi phí gọi optimizer.step(), vốn là một trong những phần tốn tài nguyên tính toán nhất, được giảm mạnh, góp phần đáng kể vào việc tăng tốc độ huấn luyện tổng thể. Sau khi đã đi qua từng thành phần, slide tiếp theo sẽ tổng kết lại toàn bộ vòng lặp huấn luyện.

---
## Slide 49: Tổng kết vòng lặp huấn luyện
**Nội dung chính trên slide:** Sơ đồ tikz 5 bước: Render → Tính loss → Backward → optimizer.step() → Densify/Prune (định kỳ) → lặp lại; chạy hàng chục nghìn iteration.

**Lời thuyết trình:**
Slide này tổng kết lại toàn bộ những gì em vừa trình bày về loss, gradient và optimizer, gói gọn thành một vòng lặp huấn luyện hoàn chỉnh gồm năm bước, thể hiện qua sơ đồ trên màn hình. Bước một là render: từ tập Gaussian hiện tại, dựng ảnh 2D theo góc camera được chọn. Bước hai là tính loss, kết hợp L1 và D-SSIM như đã trình bày ở đầu phần này. Bước ba là backward, tức lan truyền ngược gradient qua rasterizer khả vi, từ loss về tới từng tham số Gaussian. Bước bốn là optimizer.step(), sử dụng sparse Adam để cập nhật tham số một cách hiệu quả. Và bước năm, diễn ra định kỳ chứ không phải mọi iteration, là densify và prune — tức điều chỉnh mật độ Gaussian, thêm vào nơi thiếu chi tiết và loại bỏ Gaussian dư thừa. Sau bước năm, vòng lặp quay trở lại bước một để tiếp tục render với tập Gaussian đã được cập nhật. Toàn bộ chu trình năm bước này lặp lại hàng chục nghìn iteration cho tới khi mô hình hội tụ, đạt chất lượng tái tạo mong muốn. Với nền tảng vòng lặp huấn luyện đã rõ, phần tiếp theo em sẽ đi sâu vào bước densify và prune — hay còn gọi là Adaptive Density Control.

---
## Slide 50: Tổng quan Adaptive Density Control (ADC)
**Nội dung chính trên slide:** Point cloud COLMAP quá thưa ở vùng thiếu chi tiết, Gaussian quá to che chi tiết nhỏ; ADC chạy định kỳ mỗi densification_interval; hai nhóm: Densify (clone/split), Prune.

**Lời thuyết trình:**
Bây giờ em xin chuyển sang một phần rất quan trọng của quy trình huấn luyện 3D Gaussian Splatting: Adaptive Density Control, viết tắt là ADC. Vấn đề xuất phát từ chỗ point cloud khởi tạo từ COLMAP thường không hoàn hảo: có vùng quá thưa, thiếu điểm ở những khu vực chi tiết phức tạp, trong khi lại có những Gaussian quá to, che khuất mất các chi tiết nhỏ cần tái tạo chính xác. ADC được thiết kế để giải quyết vấn đề này, chạy định kỳ sau mỗi khoảng densification_interval iteration trong suốt quá trình huấn luyện, nhằm điều chỉnh động mật độ Gaussian sao cho phù hợp với độ phức tạp hình học thực tế của scene. ADC gồm hai nhóm thao tác chính, như sơ đồ trên màn hình thể hiện: nhóm Densify, bao gồm clone tức nhân bản và split tức tách, dùng để thêm Gaussian ở nơi thiếu hình học; và nhóm Prune, dùng để cắt tỉa những Gaussian thừa, mờ hoặc không còn đóng góp giá trị. Mục tiêu tổng thể là cân bằng giữa chất lượng tái tạo và số lượng Gaussian, vì số lượng Gaussian càng lớn thì chi phí bộ nhớ và tốc độ render càng bị ảnh hưởng. Slide tiếp theo sẽ trình bày dòng thời gian cụ thể của các thao tác này.

---
## Slide 51: Dòng thời gian densify/prune
**Nội dung chính trên slide:** densify_from_iter (bắt đầu), densification_interval (clone/split định kỳ), opacity_reset_interval (reset opacity), densify_until_iter (dừng thêm), final_prune (chỉ 3DGS gốc).

**Lời thuyết trình:**
Để hiểu rõ ADC vận hành theo thời gian như thế nào, ta cần nắm năm mốc và khoảng quan trọng. Đầu tiên là densify_from_iter, đánh dấu thời điểm bắt đầu cho phép densify — thường hệ thống sẽ bỏ qua vài trăm iteration đầu tiên để cho scene có thời gian ổn định trước khi can thiệp thêm bớt Gaussian, tránh gây nhiễu loạn ngay từ đầu. Sau đó, cứ mỗi khoảng densification_interval iteration, hệ thống chạy một vòng clone và split, dựa trên gradient đã tích lũy được từ các lần render trước đó. Song song đó, cứ mỗi khoảng opacity_reset_interval, toàn bộ opacity của các Gaussian được reset định kỳ, nhằm loại bỏ những Gaussian dư thừa không thực sự cần thiết — phần này em sẽ giải thích kỹ hơn ở slide sau. Quá trình densify không kéo dài mãi mãi mà dừng lại ở mốc densify_until_iter, sau mốc này hệ thống không còn thêm Gaussian mới nữa. Và cuối cùng, riêng với 3DGS gốc, còn có một bước final_prune, tức một lượt cắt tỉa cuối cùng sau khi densify đã kết thúc hoàn toàn, nhằm dọn dẹp lần chót trước khi kết thúc huấn luyện. Tiếp theo, ta đi vào chi tiết cách gradient được tích lũy để làm tín hiệu densify.

---
## Slide 52: Tích luỹ gradient view-space
**Nội dung chính trên slide:** Gradient vị trí 2D cộng dồn qua nhiều lần render; đếm denom (số lần visible); công thức $\bar{g} = \sum_v \nabla_{view} / \text{denom}$; g lớn → ứng viên clone/split.

**Lời thuyết trình:**
Để quyết định Gaussian nào cần densify, hệ thống cần một tín hiệu định lượng, và tín hiệu đó chính là gradient view-space được tích lũy qua thời gian. Cụ thể, với mỗi Gaussian, gradient vị trí 2D trong không gian ảnh, sinh ra từ mỗi lần render ở một góc camera khác nhau, được cộng dồn lại chứ không dùng riêng lẻ từng lần. Đồng thời, hệ thống đếm một biến gọi là denom, ghi lại số lần Gaussian đó thực sự được nhìn thấy, tức visible, trong batch các camera đã render. Từ hai đại lượng này, gradient trung bình g-bar được tính bằng tổng các gradient view-space chia cho denom, như công thức trên màn hình. Giá trị g-bar này chính là tín hiệu chính được dùng để quyết định densify: nếu g-bar lớn, nghĩa là Gaussian đó liên tục gây ra sai số đáng kể qua nhiều góc nhìn khác nhau, nó trở thành ứng viên cho việc clone hoặc split. Cách tích lũy qua nhiều view như thế này giúp tín hiệu ổn định hơn so với chỉ dùng một lần render đơn lẻ. Tuy nhiên, chính cách lấy trung bình này cũng tiềm ẩn một vấn đề mà slide tiếp theo sẽ phân tích.

---
## Slide 53: Vấn đề triệt tiêu gradient (gradient cancellation)
**Nội dung chính trên slide:** Gradient từ nhiều view ngược hướng có thể triệt tiêu nhau; Gaussian bị bỏ sót dù mỗi view báo lỗi lớn; hạn chế của 3DGS gốc; FastGS cải thiện bằng multi-view consistency score.

**Lời thuyết trình:**
Đây là một hạn chế quan trọng đã được biết đến của 3DGS gốc, gọi là hiện tượng triệt tiêu gradient, hay gradient cancellation. Vấn đề nằm ở chỗ khi lấy trung bình vector gradient từ nhiều góc nhìn khác nhau, nếu các vector này có hướng ngược nhau, chúng có thể triệt tiêu lẫn nhau trong phép cộng vector, chứ không đơn thuần cộng độ lớn. Hệ quả rất đáng lo ngại: dù mỗi góc nhìn riêng lẻ đều cho thấy Gaussian đó có sai số lớn — tức là vị trí hoặc kích thước của nó thực sự chưa chính xác — nhưng khi tính trung bình theo vector, g-bar tổng hợp lại có thể rất nhỏ. Kết quả là Gaussian đó bị bỏ sót, không được đưa vào danh sách densify, mặc dù đáng lẽ nó cần được xử lý. Đây chính là một hạn chế nội tại của việc chỉ dùng gradient trung bình làm tín hiệu duy nhất trong 3DGS gốc. Để khắc phục, FastGS đề xuất một hướng cải thiện, sử dụng multi-view consistency score, tức điểm nhất quán đa góc nhìn, thay vì chỉ dựa vào gradient trung bình — nội dung chi tiết về cải tiến này sẽ được trình bày ở phần sau của bài. Tiếp theo, ta quay lại cơ chế clone/split cơ bản với các ngưỡng quyết định cụ thể.

---
## Slide 54: Mặt phẳng quyết định clone/split
**Nội dung chính trên slide:** grad_thresh = 0.0002 (kích hoạt clone nếu Gaussian nhỏ); grad_abs_thresh = 0.0012 (kích hoạt split nếu Gaussian lớn); hai ngưỡng chia không gian thành 3 vùng.

**Lời thuyết trình:**
Sau khi có tín hiệu gradient, hệ thống cần quy tắc cụ thể để quyết định nên clone hay split Gaussian nào, và điều này dựa trên hai ngưỡng số học. Ngưỡng thứ nhất là grad_thresh, có giá trị 0.0002, là ngưỡng gradient trung bình: nếu gradient vượt ngưỡng này và Gaussian đang có kích thước nhỏ, hệ thống sẽ kích hoạt thao tác clone. Ngưỡng thứ hai là grad_abs_thresh, giá trị 0.0012, là ngưỡng absolute-gradient theo kiểu Abs-GS — một biến thể gradient được tính theo giá trị tuyệt đối thay vì cộng vector thông thường, giúp giảm bớt vấn đề triệt tiêu gradient vừa nói ở slide trước; nếu vượt ngưỡng này và Gaussian đang lớn, hệ thống sẽ kích hoạt split. Về bản chất hình học, hai ngưỡng này chia không gian hai chiều gồm trục gradient và trục kích thước Gaussian thành ba vùng quyết định: vùng giữ nguyên không thay đổi, vùng clone, và vùng split, như minh họa mặt phẳng quyết định trên màn hình. Cách phân vùng rõ ràng này giúp thuật toán tự động hóa hoàn toàn quá trình điều chỉnh mật độ mà không cần can thiệp thủ công. Slide tiếp theo em sẽ mô tả chi tiết thao tác clone.

---
## Slide 55: Clone (nhân bản)
**Nội dung chính trên slide:** Áp dụng cho Gaussian nhỏ nhưng thiếu hình học; nhân đôi theo hướng gradient vị trí; không thay đổi kích thước/covariance, chỉ tăng mật độ.

**Lời thuyết trình:**
Clone là thao tác đầu tiên trong nhóm densify, được áp dụng cho những Gaussian có kích thước nhỏ nhưng lại đang bị thiếu hình học, tức thuộc trường hợp under-reconstruction — nghĩa là vùng đó cần thêm chi tiết nhưng bản thân Gaussian hiện tại đã đủ nhỏ, không cần tách nhỏ hơn nữa. Cơ chế của clone là nhân đôi Gaussian đó, và bản sao mới được đặt lệch đi một khoảng nhỏ theo đúng hướng của gradient vị trí g-bar đã tính được — tức là dịch chuyển về phía có sai số lớn nhất, nơi cần bổ sung thêm mật độ. Điểm quan trọng cần lưu ý là clone không hề thay đổi kích thước hay ma trận hiệp phương sai của Gaussian, cả bản gốc lẫn bản sao đều giữ nguyên hình dạng ban đầu. Mục đích duy nhất của clone là lấp đầy vùng thiếu chi tiết bằng cách tăng mật độ điểm, giống như việc thêm quân tiếp viện vào đúng khu vực đang thiếu người, chứ không phải thay đổi cấu trúc của quân đã có. Hình minh họa bên trái cho thấy trạng thái trước và sau khi clone, còn hình bên phải khoanh vùng cụ thể khu vực được clone trong scene thực tế. Tiếp theo là thao tác còn lại trong nhóm densify: split.

---
## Slide 56: Split (tách)
**Nội dung chính trên slide:** Áp dụng cho Gaussian quá to (over-reconstruction); tách thành 2 Gaussian con nhỏ hơn; vị trí con lấy mẫu ngẫu nhiên theo phân phối chuẩn dựa trên covariance cha.

**Lời thuyết trình:**
Ngược lại với clone, split được áp dụng cho những Gaussian quá to, thuộc trường hợp over-reconstruction, tức là một Gaussian đang cố gắng biểu diễn một vùng có quá nhiều chi tiết cho kích thước của nó, vượt qua ngưỡng liên quan đến phạm vi scene, ký hiệu là dense. Khi bị split, Gaussian cha được tách thành hai Gaussian con nhỏ hơn, với scale thường giảm theo một hệ số cố định so với Gaussian gốc, giúp mỗi Gaussian con biểu diễn một vùng nhỏ hơn, chi tiết hơn. Điểm thú vị về mặt kỹ thuật là vị trí của các Gaussian con không được đặt cố định mà được lấy mẫu ngẫu nhiên theo một phân phối chuẩn, và phân phối này được xây dựng dựa trên chính ma trận hiệp phương sai gốc của Gaussian cha — nghĩa là các điểm con có xu hướng phân bố theo đúng hình dạng, hướng và độ trải rộng mà Gaussian cha vốn đang biểu diễn. Cách làm này đảm bảo Gaussian con sinh ra vẫn nằm trong vùng không gian hợp lý về mặt hình học, không bị lệch lạc ngẫu nhiên. Hai hình minh họa cho thấy hình học tách và cách lấy mẫu theo covariance. Sau khi đã có clone và split để tăng mật độ, ta chuyển sang chiều ngược lại: prune, để cắt giảm Gaussian dư thừa.

---
## Slide 57: Prune theo độ quan trọng đa góc nhìn (FastGS)
**Nội dung chính trên slide:** 3DGS gốc prune theo opacity thấp hoặc kích thước lớn; FastGS tính importance score tổng hợp qua nhiều góc nhìn; Gaussian importance thấp nhất bị cắt trước.

**Lời thuyết trình:**
Slide này giới thiệu một cải tiến quan trọng khác của FastGS liên quan đến bước prune. Trong 3DGS gốc, việc prune, tức cắt tỉa Gaussian, được thực hiện khá đơn giản: chỉ dựa trên ngưỡng opacity thấp, tức Gaussian gần như trong suốt không còn đóng góp gì, hoặc dựa trên kích thước quá lớn. Cách làm này tuy đơn giản nhưng không phản ánh chính xác mức độ đóng góp thực tế của từng Gaussian vào chất lượng ảnh cuối cùng. FastGS cải tiến bằng cách tính một điểm số gọi là importance, tức độ quan trọng, cho mỗi Gaussian, dựa trên đóng góp thực tế của nó vào chất lượng ảnh render, và điểm này được tổng hợp qua nhiều góc nhìn khác nhau chứ không chỉ một góc đơn lẻ. Với cách tiếp cận này, những Gaussian có điểm importance thấp nhất sẽ bị cắt tỉa trước tiên, bất kể giá trị opacity hay kích thước của chúng là bao nhiêu — nghĩa là ngay cả một Gaussian có opacity cao nhưng đóng góp thực tế thấp vẫn có thể bị loại bỏ. Kết quả cuối cùng là mô hình trở nên gọn hơn đáng kể về số lượng Gaussian nhưng vẫn giữ được chất lượng tái tạo, và đây chính là một trong những đòn bẩy chính giúp FastGS tăng tốc độ render. Tiếp theo, ta xem số lượng Gaussian biến đổi ra sao theo thời gian huấn luyện.

---
## Slide 58: Số lượng Gaussian tăng theo iteration
**Nội dung chính trên slide:** N tăng dần qua mỗi vòng densify (clone+split), bước nhảy rõ rệt tại densification_interval; N ngừng tăng sau densify_until_iter, chỉ còn giảm do prune.

**Lời thuyết trình:**
Slide này minh họa trực quan sự thay đổi số lượng Gaussian, ký hiệu N, xuyên suốt quá trình huấn luyện. Nhìn vào biểu đồ bên trái, ta thấy N tăng dần theo thời gian, và đặc điểm nổi bật là có những bước nhảy khá rõ rệt tại các mốc densification_interval — đúng như đã trình bày, vì mỗi lần chạm mốc này, hệ thống thực hiện một vòng clone và split, làm số Gaussian tăng đột ngột thay vì tăng đều đặn liên tục. Sau khi vượt qua mốc densify_until_iter, đường biểu diễn N gần như đi ngang, không còn tăng nữa vì hệ thống đã dừng hẳn việc thêm Gaussian mới, và từ điểm này trở đi N chỉ còn có thể giảm do tác động của prune, chứ không tăng thêm được nữa. Biểu đồ bên phải thể hiện chi tiết hơn việc hạch toán, tức accounting, cho từng nguồn đóng góp vào sự thay đổi N: bao nhiêu từ clone, bao nhiêu từ split, và bao nhiêu bị trừ đi do prune ở mỗi giai đoạn. Việc theo dõi N theo thời gian rất hữu ích để kiểm chứng hành vi thuật toán có đúng như thiết kế hay không. Slide cuối cùng trong phần của em sẽ nói về một cơ chế đặc biệt: reset opacity định kỳ.

---
## Slide 59: Reset opacity định kỳ
**Nội dung chính trên slide:** Mỗi opacity_reset_interval (mặc định 3000 iter), opacity reset về giá trị thấp qua sigmoid-inverse; buộc optimizer đánh giá lại; Gaussian dư thừa không hồi phục kịp sẽ bị prune.

**Lời thuyết trình:**
Đây là cơ chế cuối cùng trong phần Adaptive Density Control mà em muốn trình bày: reset opacity định kỳ. Cứ sau mỗi khoảng opacity_reset_interval iteration, mặc định là 3000, toàn bộ giá trị opacity của tất cả Gaussian trong scene đều bị đặt lại về một giá trị thấp, thông qua phép biến đổi sigmoid-inverse để đảm bảo phù hợp với cách opacity được tham số hóa nội bộ. Mục đích của việc này là buộc optimizer phải, có thể nói là, đánh giá lại từ đầu vai trò của từng Gaussian trong scene, thay vì để những Gaussian đã đạt opacity cao từ trước cứ mãi giữ nguyên trạng thái đó dù có thể chúng không còn thực sự cần thiết. Sau khi reset, những Gaussian thực sự quan trọng, đóng góp thật sự vào chất lượng ảnh, sẽ nhanh chóng có opacity tăng trở lại trong quá trình huấn luyện tiếp theo, nhờ gradient liên tục đẩy chúng lên. Ngược lại, những Gaussian dư thừa, không còn đóng góp rõ ràng, sẽ không kịp hồi phục opacity, và do đó sẽ bị prune ở vòng cắt tỉa kế tiếp. Hai biểu đồ trên màn hình minh họa quỹ đạo thay đổi opacity theo thời gian, và histogram so sánh phân bố opacity trước và sau khi reset. Đến đây, em xin kết thúc phần trình bày về Loss, Gradient, Optimizer và tổng quan Adaptive Density Control, xin nhường lời cho phần tiếp theo đi sâu vào các cải tiến cụ thể của FastGS-lite.

---
