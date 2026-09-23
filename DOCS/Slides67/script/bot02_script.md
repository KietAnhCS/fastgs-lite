# Kịch bản thuyết trình — Bot 02 (Slide 22–39)
## Phạm vi: Phần SH-B (Spherical Harmonics, tiếp theo), Phần 2 (Chiếu phối cảnh & Rasterizer khả vi)

---

## Slide 22: Real SH bậc thấp: $Y_{00}, Y_{1,-1}, Y_{10}, Y_{11}$
**Nội dung chính trên slide:** Định nghĩa cụ thể các hàm SH thực bậc 0 và bậc 1 theo tọa độ $x,y,z$ trên mặt cầu đơn vị, phân biệt hằng số chuẩn hóa cố định với hệ số học được.

**Lời thuyết trình:**
Tiếp tục phần nền tảng toán học, slide này đi vào cụ thể các hàm SH bậc thấp nhất. Với mỗi điểm trên mặt cầu đơn vị, ta biểu diễn qua góc theta, phi thành tọa độ x, y, z. Ở bậc l bằng 0, ta chỉ có một hàm duy nhất là hằng số Y00, không phụ thuộc hướng nhìn — đây chính là thành phần "màu trung bình". Ở bậc l bằng 1, có ba hàm tuyến tính theo x, y, z, tương ứng ba hướng trong không gian. Điểm quan trọng cần nhấn mạnh là các hằng số C0, C1 này hoàn toàn cố định, được tính sẵn bằng công thức toán, không phải tham số học được. Cái thực sự học được trong 3D Gaussian Splatting là các hệ số nhân klm — chính là f_dc cho bậc 0 và f_rest cho các bậc cao hơn, được lưu trong mỗi Gaussian. Đây là nền tảng để sang slide sau ta so sánh SH với chuỗi Fourier quen thuộc.

---
## Slide 23: Chuỗi Fourier (1D) so với khai triển SH (2D cầu)
**Nội dung chính trên slide:** So sánh khai triển Fourier trên đường tròn với khai triển SH trên mặt cầu, cùng chung ý tưởng phân tích theo tần số không gian.

**Lời thuyết trình:**
Để hiểu bản chất của spherical harmonics, ta liên hệ với một công cụ quen thuộc hơn: chuỗi Fourier. Chuỗi Fourier phân tích một hàm tuần hoàn trên đường tròn thành tổng các thành phần e mũ i n phi, với n càng lớn thì tần số dao động càng nhanh. SH làm điều tương tự nhưng trên mặt cầu hai chiều: một hàm bất kỳ trên mặt cầu được phân tích thành tổng các hàm cơ sở Ylm, với bậc l đóng vai trò giống n bên Fourier — l càng lớn, hàm càng dao động nhanh, càng mang nhiều chi tiết tần số cao. Cả hai đều dựa trên hai tính chất cốt lõi: tính đầy đủ, nghĩa là tổng vô hạn tái tạo được mọi hàm bình phương khả tích, và tính trực chuẩn giữa các hàm cơ sở. Trong ngữ cảnh 3D Gaussian Splatting, hàm f chính là hàm màu ci của mỗi Gaussian theo hướng nhìn d. Hiểu được sự tương đồng này giúp ta dễ hình dung vì sao phải cắt cụt chuỗi SH ở một bậc hữu hạn, điều sẽ trình bày ngay sau đây.

---
## Slide 24: Cắt cụt chuỗi SH ở bậc hữu hạn $D$
**Nội dung chính trên slide:** Giải thích việc xấp xỉ chuỗi vô hạn bằng tổng hữu hạn đến bậc D, hiện tượng mất chi tiết và Gibbs phenomenon, lý do chọn D=3.

**Lời thuyết trình:**
Trong thực tế không thể tính tổng vô hạn, nên ta phải cắt cụt chuỗi SH ở một bậc hữu hạn D, chỉ giữ lại các số hạng từ l bằng 0 đến l bằng D. Khi D nhỏ, hàm tái dựng chỉ còn thành phần tần số thấp, giống như làm mượt ảnh, mất đi các chi tiết sắc nét — ví dụ các đốm sáng phản xạ đặc trưng của bề mặt bóng. Hiện tượng này tương tự Gibbs phenomenon quen thuộc trong xử lý tín hiệu Fourier: khi cắt cụt sớm, phần tái dựng có thể xuất hiện dao động gợn sóng tại các vùng biến thiên nhanh. Nhìn vào biểu đồ, có thể thấy sai số L2 giảm dần đơn điệu khi D tăng từ 0 lên 1, 2, 3, hình dạng tái dựng ngày càng chi tiết, sắc nét hơn. Đây chính là lý do 3D Gaussian Splatting chọn D bằng 3 làm điểm dừng: đủ để biểu diễn các hiệu ứng phản xạ phụ thuộc góc nhìn như specular, nhưng vẫn giữ số lượng hệ số phải lưu và tối ưu ở mức chấp nhận được. Slide tiếp theo sẽ tính cụ thể con số này.

---
## Slide 25: Số hệ số SH: từ $2l+1$ đến $(D+1)^2$
**Nội dung chính trên slide:** Công thức đếm số hàm cơ sở mỗi bậc và tổng số hệ số khi cắt cụt đến D, áp dụng cho D=3 ra 48 số thực mỗi Gaussian.

**Lời thuyết trình:**
Slide này lượng hóa chi phí bộ nhớ của việc chọn bậc D. Mỗi bậc l có đúng 2l cộng 1 hàm cơ sở độc lập, tương ứng các giá trị m chạy từ âm l đến dương l. Khi cộng dồn số hàm này từ l bằng 0 đến D, ta được công thức rất gọn: tổng bằng (D cộng 1) bình phương. Cụ thể, D bằng 0 cho 1 hệ số, D bằng 1 cho 4, D bằng 2 cho 9, và D bằng 3 — bậc mà 3D Gaussian Splatting sử dụng — cho 16 hệ số. Nhưng đây mới là cho một kênh màu; vì ảnh có ba kênh RGB, nên tổng cộng ta cần 16 nhân 3 bằng 48 số thực cho mỗi Gaussian chỉ riêng phần màu. Trong đó, một hệ số DC nhân ba kênh chính là f_dc, còn 45 hệ số còn lại là f_rest. Đây là cái giá phải trả về bộ nhớ và băng thông để đổi lấy khả năng biểu diễn màu phụ thuộc góc nhìn — một đánh đổi quan trọng khi nhân với hàng triệu Gaussian trong một cảnh. Tiếp theo, ta sẽ xem công thức đầy đủ để giải mã màu từ các hệ số này.

---
## Slide 26: Công thức computeColorFromSH: giải mã màu view-dependent
**Nội dung chính trên slide:** Công thức đầy đủ tính màu quan sát từ tổng các bậc SH đến D=3, vai trò của phép dịch +0.5 và chặn dưới max(0,.).

**Lời thuyết trình:**
Đây là công thức trung tâm để "giải mã" màu thực tế nhìn thấy từ các hệ số SH đã học. Màu ci theo hướng nhìn d được tính bằng cách cộng dồn đóng góp của từng bậc: bậc 0 cho thành phần màu cơ bản C0 nhân k00, bậc 1 cộng thêm ba số hạng tuyến tính theo x, y, z, rồi đến bậc 2 và bậc 3 là các đa thức bậc cao hơn Pl(m) nhân với hệ số klm tương ứng. Các hằng số Clm ở đây là hằng số chuẩn hóa cố định tính sẵn theo công thức giai thừa, còn klm chính là tham số học được — f_dc và f_rest của từng Gaussian. Có hai chi tiết kỹ thuật quan trọng: cộng thêm 0.5 để dịch tâm phân bố màu, vốn có thể âm dương, về khoảng 0 đến 1 phù hợp không gian RGB; và lấy max với 0 để chặn dưới, vì tổng hữu hạn của SH có thể cho giá trị âm không hợp lệ. Điểm bị chặn này được lưu lại trong mảng clamped để xử lý đúng khi lan truyền ngược gradient. Đây chính là bước giải mã màu phụ thuộc góc nhìn từ các hệ số đã tối ưu.

---
## Slide 27: Lịch tăng bậc SH khi huấn luyện: $D(t)$
**Nội dung chính trên slide:** Bậc SH sử dụng tăng dần theo hàm bậc thang D(t) trong quá trình huấn luyện, từ D=0 lên D=3 sau mỗi 1000 iteration.

**Lời thuyết trình:**
Một chi tiết thú vị trong huấn luyện là bậc D không cố định ngay từ đầu, mà tăng dần theo một hàm bậc thang: D của t bằng min của 3 và phần nguyên của t chia 1000. Ban đầu, khi t còn nhỏ, D bằng 0, nghĩa là chỉ dùng Y00 — màu lúc này hoàn toàn đẳng hướng, không phụ thuộc góc nhìn, chỉ có f_dc hoạt động. Cứ sau mỗi 1000 bước huấn luyện, D tăng thêm 1, mở khóa thêm các hệ số bậc cao hơn, cho đến khi đạt tối đa D bằng 3. Lý do thiết kế này rất hợp lý về mặt tối ưu hóa: ở giai đoạn đầu, hình học của Gaussian — vị trí, hiệp phương sai — còn rất thô và chưa hội tụ. Nếu cho phép tối ưu ngay các hệ số bậc cao mô tả hiệu ứng phản xạ, mô hình dễ bị overfit vào nhiễu quan sát thay vì học đúng hình học. Bằng cách tăng dần bậc, quá trình huấn luyện tách biệt rõ hai giai đoạn: học hình học ổn định trước, rồi mới học các hiệu ứng phản xạ phức tạp theo góc nhìn sau. Tiếp theo ta sẽ xem một công cụ kiểm chứng toán học quan trọng: định lý Parseval.

---
## Slide 28: Định lý Parseval trên mặt cầu: năng lượng và hệ số
**Nội dung chính trên slide:** Tính trực chuẩn của SH cho phép năng lượng tín hiệu bằng tổng bình phương hệ số, phân tách năng lượng theo bậc l và ước lượng sai số cắt cụt.

**Lời thuyết trình:**
Nhờ tính trực chuẩn của hệ hàm cơ sở Ylm, ta có một công cụ mạnh gọi là định lý Parseval trên mặt cầu: tích phân của bình phương hàm màu trên toàn mặt cầu bằng đúng tổng bình phương tất cả các hệ số khai triển klm. Điều này cho phép kiểm chứng bằng số học — tính tích phân số trên lưới theta, phi rồi so với tổng bình phương hệ số, hai bên phải khớp nhau. Quan trọng hơn, định lý này cho phép ta phân tách năng lượng tín hiệu theo từng bậc l, xem bậc nào đang mang nhiều thông tin nhất. Thực nghiệm cho thấy bậc l bằng 0, tức thành phần DC, chiếm ưu thế áp đảo về năng lượng — điều này khá trực quan vì màu trung bình luôn là thành phần lớn nhất. Ứng dụng thực tế của công thức này là đánh giá được chính xác bao nhiêu năng lượng bị mất khi cắt chuỗi ở bậc D hữu hạn: phần dư, tức tổng bình phương các hệ số ở bậc lớn hơn D, chính là sai số bị bỏ qua. Từ quan sát này, slide sau sẽ đi sâu vào ý nghĩa của riêng hệ số DC k00.

---
## Slide 29: Hệ số DC $k_{00}$: chính là màu trung bình
**Nội dung chính trên slide:** Quan hệ giữa hệ số DC và màu trung bình theo mọi hướng nhìn, cơ sở khởi tạo f_dc, và lý do tách learning rate cho f_dc và f_rest.

**Lời thuyết trình:**
Slide cuối phần SH này giải thích ý nghĩa vật lý của riêng hệ số k00. Vì Y00 là hằng số C0, còn mọi Ylm với l lớn hơn hoặc bằng 1 đều có trung bình bằng 0 trên mặt cầu — do trực giao với hàm hằng — nên nếu lấy trung bình màu ci theo mọi hướng nhìn đều nhau, kết quả đúng bằng C0 nhân k00. Nói cách khác, k00 chính là màu trung bình, không hơn không kém. Đây là lý do khi khởi tạo Gaussian, ta dùng công thức RGB2SH ngược lại: lấy màu quan sát trung bình chia cho C0 để có f_dc ban đầu khớp chính xác với dữ liệu quan sát. Vì k00 chiếm phần lớn năng lượng tín hiệu màu như vừa thấy ở slide Parseval, nó cần learning rate thấp để tránh làm dao động màu toàn ảnh trong quá trình huấn luyện. Ngược lại, 45 hệ số còn lại ở bậc 1 đến 3, tức f_rest, chỉ là các nhiễu loạn bậc cao mã hóa hiệu ứng phản xạ nhỏ, nên được tách ra optimizer riêng với tần suất cập nhật khác. Đến đây ta đã hoàn tất phần nền tảng toán học về Spherical Harmonics; tiếp theo chúng ta chuyển sang phần chiếu phối cảnh và rasterizer khả vi — nơi các hệ số màu này thực sự được dùng để tạo ra ảnh cuối cùng.

---
## Slide 30: Tổng quan pipeline render
**Nội dung chính trên slide:** Sơ đồ tổng thể năm bước của pipeline render 3DGS: world-to-camera, projection, tiling, sorting, alpha blending.

**Lời thuyết trình:**
Bây giờ ta chuyển sang phần thứ hai: chiếu phối cảnh và rasterizer khả vi — đây chính là bước biến các Gaussian 3D với các hệ số SH vừa học ở phần trước thành một bức ảnh 2D cụ thể. Pipeline gồm năm bước chính. Đầu tiên là chuyển từ hệ tọa độ thế giới sang hệ tọa độ camera bằng phép biến đổi view gồm ma trận quay R và vector tịnh tiến t. Tiếp theo là bước projection, chiếu các Gaussian 3D xuống mặt phẳng ảnh 2D. Sau đó là tiling, xác định mỗi Gaussian ảnh hưởng đến những tile ảnh nào. Rồi đến sorting, sắp xếp các Gaussian trong từng tile theo độ sâu để blend đúng thứ tự. Cuối cùng là alpha blending tuần tự để ra màu pixel cuối cùng. Toàn bộ chuỗi năm bước này đều khả vi, nghĩa là ta có thể lan truyền gradient ngược từ ảnh render về lại các tham số Gaussian. Các slide tiếp theo sẽ đi vào chi tiết từng bước, bắt đầu từ phép chiếu pinhole camera.

---
## Slide 31: Chiếu camera pinhole
**Nội dung chính trên slide:** Mô hình camera pinhole, phép biến đổi world-to-camera bằng R, t, và công thức chiếu phối cảnh cơ bản x'=fx/z, y'=fy/z.

**Lời thuyết trình:**
Bước đầu tiên trong quá trình chiếu là đưa Gaussian từ hệ tọa độ thế giới về hệ tọa độ camera, bằng cách nhân với ma trận quay R và cộng vector tịnh tiến t — đây chỉ đơn giản là đổi hệ quy chiếu, chưa liên quan gì đến ống kính. Sau đó, ta áp dụng mô hình chiếu pinhole — mô hình camera lỗ kim kinh điển trong thị giác máy tính — với tiêu cự f và điểm chính principal point. Công thức chiếu cơ bản rất đơn giản về mặt hình học: tọa độ x phẩy bằng f nhân x chia z, và y phẩy bằng f nhân y chia z. Ý nghĩa là điểm càng xa camera theo trục z thì càng bị "thu nhỏ" lại gần tâm ảnh, đúng như nguyên lý phối cảnh mà mắt người quan sát. Kết quả của bước này là tọa độ tâm của mỗi Gaussian trên ảnh 2D — tức là ta biết Gaussian đó sẽ xuất hiện ở đâu trên màn hình. Tuy nhiên, Gaussian không phải một điểm mà là một khối hình elip có hình dạng, nên bước tiếp theo ta cần chiếu cả phần hiệp phương sai để biết hình dạng elip đó trên ảnh.

---
## Slide 32: Chiếu hiệp phương sai (EWA splatting)
**Nội dung chính trên slide:** Xấp xỉ tuyến tính phép chiếu phi tuyến bằng Jacobian, công thức Sigma' = J W Sigma W^T J^T, kỹ thuật EWA splatting.

**Lời thuyết trình:**
Phép chiếu phối cảnh vừa nêu ở slide trước là phi tuyến do có phép chia cho z, nên không thể áp dụng trực tiếp cho toàn bộ hình dạng elip 3D của Gaussian. Giải pháp là xấp xỉ tuyến tính cục bộ quanh tâm Gaussian bằng ma trận Jacobian J của phép chiếu. Khi đó, hiệp phương sai 2D sau khi chiếu được tính bằng công thức Sigma phẩy bằng J nhân W nhân Sigma nhân W chuyển vị nhân J chuyển vị, trong đó W là phần quay của ma trận view. Đây thực chất là quy tắc biến đổi hiệp phương sai qua một phép biến đổi tuyến tính gần đúng — giống như biến đổi ellipsoid qua một ánh xạ tuyến tính thì vẫn ra ellipsoid, chỉ khác hình dạng. Kết quả cuối cùng là mỗi Gaussian 3D trong không gian sau khi chiếu sẽ trở thành một ellipse 2D cụ thể trên ảnh, với tâm và hình dạng xác định rõ ràng. Kỹ thuật này có tên gọi chuyên môn là Elliptical Weighted Average splatting, viết tắt EWA splatting, vốn là nền tảng kinh điển của các phương pháp splatting trước cả Gaussian Splatting hiện đại. Tiếp theo, ta sẽ xem FastGS tối ưu vùng ảnh hưởng của elip này như thế nào.

---
## Slide 33: Compact Box — đòn bẩy tăng tốc #1 của FastGS
**Nội dung chính trên slide:** So sánh bounding-box 3-sigma mặc định với compact box của FastGS, tham số --mult điều chỉnh kích thước vùng ảnh hưởng.

**Lời thuyết trình:**
Sau khi có ellipse 2D, hệ thống cần xác định một bounding-box bao quanh nó để biết vùng ảnh hưởng trên ảnh. Cài đặt gốc dùng bounding-box 3-sigma, tức phạm vi bao phủ 3 lần độ lệch chuẩn theo mọi hướng — khá an toàn về mặt thống kê nhưng cũng khá rộng, khiến Gaussian chạm vào nhiều tile hơn mức thực sự cần thiết, gây lãng phí tính toán. Đây chính là điểm mà FastGS tối ưu: thay bounding-box mặc định bằng một compact box nhỏ gọn hơn, với kích thước điều chỉnh được qua tham số dòng lệnh mult. Giá trị mặc định là 0.5, và có thể tăng lên 0.7 cho các cảnh lớn hoặc phức tạp để tránh cắt mất phần đuôi Gaussian quan trọng. Kết quả trực tiếp là giảm đáng kể số tile mà mỗi Gaussian phải rasterize, tức giảm tải tính toán mà không đánh đổi nhiều về chất lượng hình ảnh. Đây được xem là đòn bẩy tăng tốc đầu tiên và quan trọng của FastGS. Slide sau sẽ định lượng cụ thể mức giảm tải này qua tỉ lệ diện tích.

---
## Slide 34: Tỉ lệ diện tích ảnh hưởng tile
**Nội dung chính trên slide:** Số tile bị ảnh hưởng tỉ lệ thuận với diện tích ellipse chiếu; compact box giảm diện tích này mà không giảm chất lượng đáng kể.

**Lời thuyết trình:**
Slide này làm rõ mối quan hệ định lượng giữa kích thước bounding-box và chi phí tính toán. Về bản chất, số lượng tile mà một Gaussian "chạm" tới tỉ lệ thuận với diện tích của ellipse chiếu trên ảnh — box càng lớn thì diện tích càng lớn, số tile cần xử lý càng nhiều, kéo theo khối lượng công việc rasterize tăng lên tương ứng. Biểu đồ ở đây cho thấy khi áp dụng compact box thay cho bounding-box 3-sigma mặc định, tỉ lệ diện tích ảnh hưởng giảm đi rõ rệt, trong khi chất lượng hình ảnh render ra gần như không thay đổi đáng kể, vì phần đuôi Gaussian bị cắt bớt chỉ đóng góp giá trị rất nhỏ vào màu pixel cuối cùng. Đây chính là cơ sở lý thuyết vững chắc cho việc tăng tốc rasterization mà FastGS đề xuất — không phải một mẹo tối ưu tùy tiện mà dựa trên quan sát định lượng về mối quan hệ diện tích và chi phí. Từ đây, ta chuyển sang phần lõi kỹ thuật: rasterizer khả vi chạy trên GPU.

---
## Slide 35: Rasterizer khả vi theo tile (CUDA)
**Nội dung chính trên slide:** Chia ảnh thành lưới tile, mỗi tile lưu danh sách Gaussian đã sort theo depth, xử lý song song CUDA, toàn bộ khả vi.

**Lời thuyết trình:**
Đây là trái tim kỹ thuật của toàn bộ pipeline render: rasterizer khả vi cài đặt trên CUDA. Ảnh output được chia thành một lưới các tile nhỏ, ví dụ mỗi tile kích thước 16 nhân 16 pixel. Với mỗi tile, hệ thống duy trì một danh sách các Gaussian có vùng ảnh hưởng chạm tới tile đó — đây chính là kết quả của bước tiling và compact box đã nói ở các slide trước. Danh sách này sau đó được sắp xếp theo độ sâu trước khi tiến hành blend, đảm bảo thứ tự trước sau đúng về mặt vật lý. Về mặt triển khai phần cứng, quá trình này được xử lý song song trên GPU, trong đó mỗi thread-block CUDA đảm nhận xử lý một tile độc lập — đây là mô hình song song hóa rất tự nhiên và hiệu quả cho bài toán này. Phần cài đặt cụ thể nằm trong module diff-gaussian-rasterization_fastgs của mã nguồn. Điều quan trọng nhất cần nhấn mạnh là toàn bộ quá trình này được thiết kế khả vi hoàn toàn, tức là hỗ trợ lan truyền ngược gradient, cho phép huấn luyện end-to-end. Tiếp theo, ta sẽ xem cụ thể công thức trường alpha được dùng trong bước blend này.

---
## Slide 36: Trường alpha (opacity field)
**Nội dung chính trên slide:** Công thức trường alpha alpha(p) = alpha_i * G'_i(p), với G'_i là Gaussian 2D chuẩn hóa theo Sigma'.

**Lời thuyết trình:**
Sau khi mỗi Gaussian 3D đã được chiếu thành một ellipse 2D với hiệp phương sai Sigma phẩy, nó tạo ra một trường độ mờ liên tục — hay còn gọi là trường alpha — trên toàn bộ ảnh, chứ không chỉ tại một điểm duy nhất. Công thức của trường này là alpha tại điểm p bằng độ mờ cơ bản alpha_i của Gaussian nhân với G phẩy i tại p, trong đó G phẩy i là hàm Gaussian 2D đã chuẩn hóa theo hiệp phương sai Sigma phẩy vừa tính được. Về mặt trực quan, giá trị alpha tại một điểm p sẽ càng lớn khi p càng gần tâm của ellipse, và giảm dần mượt mà theo hình chuông khi đi ra xa tâm, tuân theo đúng phân bố Gaussian chuẩn. Đây chính là cơ chế khiến mỗi Gaussian không có biên cứng như một hình khối rời rạc, mà "hòa tan" mượt mà vào ảnh xung quanh nó, tạo hiệu ứng render tự nhiên, khử răng cưa tốt. Trường alpha này của từng Gaussian chính là nguyên liệu đầu vào cho bước tiếp theo: kết hợp nhiều Gaussian chồng lấn qua alpha blending.

---
## Slide 37: Alpha blending (front-to-back)
**Nội dung chính trên slide:** Công thức blend C = sum ci*alpha_i*Ti với Ti là transmittance tích lũy từ các Gaussian gần camera hơn.

**Lời thuyết trình:**
Đây là công thức trung tâm quyết định màu cuối cùng của mỗi pixel. Màu pixel C được tính bằng tổng có trọng số của tất cả Gaussian chồng lên nhau tại điểm đó, theo đúng thứ tự độ sâu đã sort ở bước trước: C bằng tổng của ci nhân alpha_i nhân Ti. Ở đây ci là màu của Gaussian thứ i, được giải mã từ các hệ số SH mà ta đã học ở phần trước; alpha_i là độ mờ tại điểm đó; còn Ti là transmittance — độ truyền sáng còn lại trước khi tới Gaussian thứ i, được tính bằng tích của (1 trừ alpha_j) cho tất cả các Gaussian j nằm gần camera hơn Gaussian i. Nói cách khác, Ti đo lường "còn bao nhiêu ánh sáng lọt qua được" sau khi đã bị các lớp Gaussian phía trước hấp thụ một phần. Ý nghĩa vật lý quan trọng là các Gaussian càng gần camera thì đóng góp trọng số càng lớn vào màu cuối cùng, vì ánh sáng từ chúng chưa bị che khuất bởi lớp nào. Đây chính là mô hình front-to-back alpha compositing kinh điển trong đồ họa máy tính. Slide sau sẽ khai thác tính chất suy giảm của Ti để tăng tốc render.

---
## Slide 38: Suy giảm độ truyền sáng (transmittance decay)
**Nội dung chính trên slide:** Ti giảm theo cấp số nhân khi có nhiều Gaussian che phía trước, cho phép early stopping để tăng tốc.

**Lời thuyết trình:**
Nhìn vào công thức Ti là tích của các thừa số (1 trừ alpha_j), ta thấy ngay Ti sẽ giảm dần theo cấp số nhân khi số lượng Gaussian che phía trước tăng lên — mỗi Gaussian thêm vào đều nhân thêm một thừa số nhỏ hơn 1. Về mặt thực tế, khi Ti giảm xuống gần bằng 0, nghĩa là hầu như toàn bộ ánh sáng đã bị các lớp phía trước hấp thụ, thì các Gaussian nằm phía sau — dù có màu sắc gì đi nữa — cũng gần như không còn đóng góp đáng kể vào màu pixel cuối cùng nữa. Đây chính là cơ sở cho một kỹ thuật tối ưu tốc độ rất hiệu quả gọi là early stopping: khi phát hiện Ti đã đủ nhỏ dưới một ngưỡng cho trước, rasterizer có thể dừng xử lý sớm, bỏ qua toàn bộ các Gaussian còn lại trong danh sách của tile đó mà không ảnh hưởng đáng kể tới chất lượng ảnh. Đây là một trong những kỹ thuật tăng tốc quan trọng nhất của rasterizer, giúp tiết kiệm đáng kể tài nguyên tính toán, đặc biệt ở các tile có mật độ Gaussian cao. Slide cuối cùng sẽ tổng kết lại toàn bộ phương trình render.

---
## Slide 39: Tổng kết phương trình render
**Nội dung chính trên slide:** Phương trình render đầy đủ C(p), tính khả vi theo mọi tham số, sơ đồ tikz tóm tắt luồng xử lý từ Gaussian 3D đến pixel.

**Lời thuyết trình:**
Slide này tổng kết lại toàn bộ phần chiếu phối cảnh và rasterizer khả vi bằng một phương trình duy nhất: C tại p bằng tổng của ci nhân alpha_i nhân tích (1 trừ alpha_j) với j nhỏ hơn i — chính là công thức alpha blending front-to-back đã trình bày, viết gọn lại thành phương trình render hoàn chỉnh cho toàn pipeline. Điều cốt lõi cần nhấn mạnh là toàn bộ phương trình này khả vi theo mọi tham số đầu vào: vị trí tâm mu, hiệp phương sai Sigma, độ mờ alpha, và các hệ số SH quyết định màu sắc. Chính nhờ tính khả vi đầy đủ này mà ta có thể huấn luyện trực tiếp toàn bộ mô hình bằng gradient descent, chỉ cần so sánh ảnh render ra với ảnh RGB quan sát thực tế, rồi lan truyền ngược sai số qua toàn bộ chuỗi Gaussian 3D, chiếu 2D, trường alpha, blend, để cập nhật từng tham số. Sơ đồ ở dưới tóm tắt trực quan luồng xử lý năm bước này thành một chuỗi liên tục. Đây cũng là điểm khép lại phần chiếu phối cảnh và rasterizer, chuẩn bị chuyển sang phần loss và gradient — nơi ta sẽ đi sâu vào cách tính sai số và cập nhật tham số cụ thể.

---
