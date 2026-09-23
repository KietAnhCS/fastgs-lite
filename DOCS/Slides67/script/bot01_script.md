# Kịch bản thuyết trình — Bot 01 (Slide 1–21)
## Phạm vi: Trang bìa, Mục lục, Phần 1 (Nền tảng 3D Gaussian), Phần SH-A (Spherical Harmonics)

---

## Slide 1: Trang bìa
**Nội dung chính trên slide:** Tiêu đề "Quy trình Render 3D Gaussian Splatting và Cải tiến FastGS (FastGS-lite)", phụ đề "Từ toán học nền tảng đến Adaptive Density Control chuyên sâu", nhóm thực hiện Digital Twin GS, ngày báo cáo.

**Lời thuyết trình:**
Kính chào thầy cô và các bạn. Em xin phép bắt đầu phần trình bày đồ án với chủ đề "Quy trình Render 3D Gaussian Splatting và Cải tiến FastGS", tên gọi tắt là FastGS-lite. Phụ đề của đồ án là "Từ toán học nền tảng đến Adaptive Density Control chuyên sâu" — điều này phản ánh đúng cấu trúc bài báo cáo hôm nay: chúng em sẽ đi từ những kiến thức toán nền tảng nhất như hàm Gaussian, hiệp phương sai, hàm điều hòa cầu, cho tới cơ chế kỹ thuật phức tạp nhất là Adaptive Density Control — cơ chế quyết định việc nhân bản, tách và loại bỏ các Gaussian trong quá trình huấn luyện. Đồ án được thực hiện bởi nhóm Digital Twin GS, dựa trên việc tái hiện và rút gọn thuật toán FastGS gốc để có thể chạy được trên phần cứng phổ thông. Sau đây, em xin trình bày nội dung chi tiết.

---

## Slide 2: Mục lục
**Nội dung chính trên slide:** Danh sách các section chính của bài (tableofcontents tự động): Nền tảng 3D Gaussian, Spherical Harmonics, Chiếu phối cảnh & Rasterizer, Loss/Gradient/Optimizer, ADC tổng quan, ADC chuyên sâu (7 phần con), Ba đòn bẩy tăng tốc, Kết quả thực nghiệm, Kết luận.

**Lời thuyết trình:**
Đây là mục lục tổng thể của bài báo cáo, được sinh tự động từ các section đã khai báo. Bài trình bày gồm bảy phần lớn. Chúng em bắt đầu với nền tảng: 3D Gaussian là gì, sau đó bổ sung phần toán học đầy đủ về Spherical Harmonics — vì đây là công cụ toán quan trọng để biểu diễn màu sắc phụ thuộc góc nhìn. Tiếp theo là quy trình chiếu phối cảnh và rasterizer khả vi, rồi đến hàm mất mát, gradient và bộ tối ưu. Phần trọng tâm của đồ án là Adaptive Density Control, được chia thành nhiều phần chuyên sâu: từ tín hiệu sai số pixel, footprint đa view, công thức score và pruning, cho đến cơ chế gradient cancellation và quyết định clone/split. Cuối bài là ba đòn bẩy tăng tốc của FastGS, kết quả thực nghiệm trên Colab T4, và phần quan sát, kết luận. Bây giờ, em xin đi vào phần đầu tiên: Nền tảng 3D Gaussian là gì.

---

## Slide 3: Trang bìa (nhắc lại trước khi vào Phần 1)
**Nội dung chính trên slide:** Slide lặp lại trang bìa (title page) ngay trước khi bắt đầu nội dung Phần 1 trong file part1.tex.

**Lời thuyết trình:**
Trước khi đi sâu vào nội dung, slide này nhắc lại một lần nữa tiêu đề đồ án và tên nhóm thực hiện, đóng vai trò như một dấu mốc chuyển tiếp giữa phần mục lục và phần nội dung chính thức. Từ đây, chúng em chính thức bắt đầu Phần 1 của bài: Nền tảng 3D Gaussian là gì. Đây là phần nền móng cho toàn bộ các phần sau, bởi vì tất cả các kỹ thuật render, tối ưu hóa và Adaptive Density Control mà chúng em sẽ trình bày về sau đều xây dựng trên khái niệm cơ bản là một "quả cầu Gaussian" trong không gian ba chiều. Nếu không nắm vững phần này, các công thức phức tạp ở những phần sau sẽ rất khó theo dõi. Vì vậy em xin dành vài slide tiếp theo để giới thiệu lại toàn bộ nội dung sẽ trình bày, trước khi đi vào động lực ra đời của 3D Gaussian Splatting.

---

## Slide 4: Nội dung trình bày
**Nội dung chính trên slide:** Danh sách đánh số 7 mục: Nền tảng 3D Gaussian; Chiếu phối cảnh & Rasterizer khả vi; Loss, Metric, Gradient & Optimizer; Adaptive Density Control; Ba đòn bẩy tăng tốc FastGS; FastGS-lite trên Colab T4; Quan sát, giới hạn & kết luận.

**Lời thuyết trình:**
Slide này trình bày lại dàn ý của toàn bộ báo cáo dưới dạng danh sách đánh số, cụ thể và dễ theo dõi hơn so với mục lục tự động ở trước. Có bảy mục chính. Thứ nhất, nền tảng 3D Gaussian là gì — trả lời câu hỏi một Gaussian được biểu diễn bằng những tham số nào. Thứ hai, cách chiếu các Gaussian ba chiều này lên mặt phẳng ảnh hai chiều thông qua rasterizer khả vi. Thứ ba, cách tính loss, gradient và cập nhật tham số bằng optimizer. Thứ tư — và đây là phần trọng tâm — Adaptive Density Control, cơ chế tự động thêm bớt Gaussian trong quá trình huấn luyện. Thứ năm, ba đòn bẩy giúp FastGS tăng tốc so với 3DGS gốc. Thứ sáu, kết quả thực nghiệm khi triển khai FastGS-lite trên phần cứng giới hạn là Colab T4. Và cuối cùng là những quan sát, giới hạn và kết luận rút ra được. Tiếp theo, chúng ta sẽ tìm hiểu động lực vì sao cần một biểu diễn cảnh mới.

---

## Slide 5: Động lực: vì sao cần một biểu diễn cảnh mới?
**Nội dung chính trên slide:** So sánh NeRF (implicit, MLP, chậm) với 3DGS (explicit, rasterization, nhanh); bảng so sánh tiêu chí biểu diễn, render, tốc độ, huấn luyện.

**Lời thuyết trình:**
Trước khi 3D Gaussian Splatting ra đời, NeRF — Neural Radiance Fields — là hướng tiếp cận thống trị trong tái tạo cảnh 3D. NeRF biểu diễn cảnh theo kiểu "implicit", nghĩa là toàn bộ thông tin về màu sắc và mật độ được mã hóa ẩn bên trong trọng số của một mạng nơ-ron MLP. Chất lượng ảnh dựng ra rất cao, nhưng để render một khung hình, ta phải thực hiện ray-marching — bắn hàng loạt tia sáng và truy vấn mạng nơ-ron liên tục dọc theo tia — khiến tốc độ cực kỳ chậm, có thể mất tới vài giây cho một khung hình. 3D Gaussian Splatting giải quyết vấn đề này bằng cách chuyển sang biểu diễn "explicit": cảnh được mô tả tường minh bằng hàng trăm nghìn quả cầu Gaussian mờ có vị trí, hình dạng rõ ràng trong không gian. Nhờ vậy, việc render chỉ cần rasterization — một kỹ thuật đồ họa truyền thống, nhanh hơn ray-marching rất nhiều — đạt tốc độ hàng chục đến hàng trăm khung hình mỗi giây, gần thời gian thực. Bảng so sánh trên slide tóm tắt rõ sự khác biệt này. Tiếp theo, em xin giới thiệu về dự án cụ thể mà nhóm đã thực hiện.

---

## Slide 6: Dự án Digital Twin GS / FastGS-lite
**Nội dung chính trên slide:** FastGS-lite là bản fork rút gọn của FastGS, giữ nguyên thuật toán gốc nhưng tối ưu để chạy trên Colab T4 (~15GB VRAM) thay vì RTX 4090 (24GB VRAM); bảng so sánh phần cứng, CUDA extension, entry point.

**Lời thuyết trình:**
Dự án của nhóm mang tên FastGS-lite, là một bản fork rút gọn từ mã nguồn gốc của FastGS, nhưng quan trọng là chúng em giữ nguyên toàn bộ thuật toán cốt lõi — không thay đổi bản chất phương pháp, mà tập trung vào việc làm cho nó chạy được trên phần cứng hạn chế hơn nhiều. Bản FastGS gốc được thiết kế và kiểm thử trên GPU RTX 4090 với 24GB VRAM — một card đồ họa cao cấp. Mục tiêu của nhóm là chạy toàn bộ pipeline này chỉ với một GPU T4 miễn phí trên Google Colab, vốn chỉ có khoảng 15GB VRAM và 12.7GB RAM hệ thống. Đây là một thách thức kỹ thuật đáng kể vì bộ nhớ eo hẹp hơn nhiều. Nhóm cũng đã vendor sẵn CUDA extension trực tiếp trong repo thay vì để dưới dạng submodule ngoài như bản gốc, giúp việc cài đặt và biên dịch trên Colab thuận tiện hơn, đồng thời bổ sung notebook có ghi log để dễ theo dõi quá trình huấn luyện. Sau khi đã rõ bối cảnh dự án, ta sẽ đi vào câu hỏi cốt lõi: một 3D Gaussian thực chất là gì.

---

## Slide 7: 3D Gaussian là gì?
**Nội dung chính trên slide:** Công thức hàm mật độ Gaussian 3D $G(x) = \exp(-\frac12(x-\mu)^T\Sigma^{-1}(x-\mu))$; 4 nhóm tham số: vị trí $\mu$, hiệp phương sai $\Sigma$, opacity $\alpha$, màu qua SH.

**Lời thuyết trình:**
Đây là công thức trung tâm của toàn bộ phương pháp. Một Gaussian 3D được mô tả bằng hàm mật độ $G(x)$, có dạng hàm mũ của một biểu thức toàn phương âm. Về mặt trực quan, công thức này cho biết: tại điểm x càng gần tâm $\mu$, giá trị của $G(x)$ càng lớn, gần bằng 1; càng ra xa tâm theo hướng nào đó, giá trị càng giảm dần về 0, tạo ra hình dạng giống một "quả chuông" mờ dần — như hình minh họa bên phải. Ma trận $\Sigma^{-1}$ trong số mũ, gọi là nghịch đảo hiệp phương sai, quyết định quả chuông này giãn nở nhanh hay chậm theo từng hướng, tức là hình dạng và độ định hướng của nó. Mỗi Gaussian trong cảnh được đặc trưng bởi bốn nhóm tham số có thể học được qua huấn luyện: vị trí tâm $\mu$ trong không gian ba chiều, ma trận hiệp phương sai $\Sigma$ quy định hình dạng và hướng, độ mờ opacity $\alpha$ quyết định mức độ trong suốt, và cuối cùng là màu sắc được biểu diễn thông qua các hệ số Spherical Harmonics. Slide tiếp theo sẽ đi sâu vào cách tham số hóa ma trận hiệp phương sai này.

---

## Slide 8: Tham số hoá hiệp phương sai
**Nội dung chính trên slide:** Không học trực tiếp $\Sigma$ để đảm bảo positive semi-definite; học scale $s$ và quaternion $q$: $\Sigma = RSS^TR^T$; khởi tạo scale từ khoảng cách điểm lân cận COLMAP.

**Lời thuyết trình:**
Một vấn đề kỹ thuật quan trọng nảy sinh: nếu để mô hình học trực tiếp ma trận $\Sigma$ gồm 6 giá trị độc lập, thì trong quá trình cập nhật gradient, hoàn toàn có khả năng $\Sigma$ trở thành một ma trận không hợp lệ về mặt toán học — tức là mất tính positive semi-definite, khiến công thức Gaussian ở slide trước không còn ý nghĩa vật lý. Để tránh điều này, người ta không học $\Sigma$ trực tiếp mà học hai đại lượng trung gian: một vector scale $s$ ba chiều biểu diễn độ giãn theo từng trục, và một quaternion $q$ bốn chiều biểu diễn phép xoay. Từ đó, hiệp phương sai được dựng lại theo công thức $\Sigma = R S S^T R^T$, trong đó $S$ là ma trận đường chéo từ $s$, còn $R$ là ma trận xoay dựng từ quaternion $q$. Cách phân tích này — tương tự phân tích SVD — đảm bảo $\Sigma$ luôn hợp lệ dù $s$ và $q$ nhận giá trị gì. Về khởi tạo, scale ban đầu được tính dựa trên khoảng cách tới các điểm lân cận trong point cloud do COLMAP tạo ra, như minh họa ở hình bên. Tiếp theo, ta sẽ hình dung trực quan hơn ý nghĩa hình học của $\Sigma$.

---

## Slide 9: Ellipsoid hiệp phương sai
**Nội dung chính trên slide:** $\Sigma$ tương đương một ellipsoid 3D; trục ellipsoid = eigenvector, độ dài trục = eigenvalue của $\Sigma$; ellipsoid càng dẹt/kéo dài thì Gaussian càng bất đẳng hướng.

**Lời thuyết trình:**
Về mặt hình học, ma trận hiệp phương sai $\Sigma$ không phải là một khái niệm trừu tượng khó hình dung — nó tương đương chính xác với một ellipsoid, tức là một khối elip ba chiều, biểu diễn vùng không gian mà Gaussian có ảnh hưởng đáng kể. Cụ thể, nếu ta phân tích trị riêng của $\Sigma$, thì các eigenvector — tức các vector riêng — chính là hướng của ba trục chính của ellipsoid, còn các eigenvalue — trị riêng tương ứng — cho biết độ dài của từng trục đó. Nói cách khác, hướng mà Gaussian "giãn ra" nhiều nhất ứng với eigenvalue lớn nhất. Khi ba trục xấp xỉ bằng nhau, ellipsoid gần giống hình cầu, Gaussian gần như đẳng hướng — ảnh hưởng đều theo mọi phương. Ngược lại, khi ellipsoid càng dẹt hoặc càng kéo dài theo một trục, Gaussian càng bất đẳng hướng — điều này rất hữu ích để mô hình hóa các bề mặt phẳng, mỏng như tường, lá cây, hay các cạnh sắc nét trong cảnh thực. Hình minh họa bên phải thể hiện rõ ellipsoid ứng với một $\Sigma$ cụ thể. Sau khi đã hiểu hình dạng, ta chuyển sang câu hỏi: màu sắc của mỗi Gaussian được biểu diễn như thế nào.

---

## Slide 10: Màu sắc qua Spherical Harmonics (SH)
**Nội dung chính trên slide:** Mỗi Gaussian lưu hệ số SH thay vì RGB cố định; màu phụ thuộc góc nhìn (view-dependent); hệ số SH bậc 0 khởi tạo từ RGB: $f_{dc} = \text{RGB2SH}(\text{color})$.

**Lời thuyết trình:**
Một điểm đặc biệt của 3D Gaussian Splatting là mỗi Gaussian không lưu một màu RGB cố định duy nhất, mà lưu một tập hệ số Spherical Harmonics, gọi tắt là SH. Lý do là vì trong thực tế, màu sắc mà ta quan sát tại một điểm bề mặt thường thay đổi tùy theo góc nhìn — ví dụ hiệu ứng phản xạ ánh sáng, độ bóng trên kim loại hay mặt nước. Nếu chỉ lưu một màu cố định, mô hình sẽ không tái hiện được các hiệu ứng specular này. Bằng cách biểu diễn màu như một hàm phụ thuộc hướng nhìn thông qua hệ số SH, mô hình có thể "tính toán lại" màu sắc phù hợp theo từng góc camera khác nhau khi render. Phần dễ hiểu nhất và cũng là điểm khởi đầu là hệ số SH bậc 0, gọi là thành phần DC — tương tự thành phần một chiều trong biến đổi Fourier — được khởi tạo trực tiếp từ màu RGB quan sát được thông qua công thức chuyển đổi $f_{dc} = \text{RGB2SH}(\text{color})$, như minh họa ở hình bên. Slide sau sẽ giới thiệu cơ sở SH đầy đủ hơn và cách tăng dần bậc trong quá trình huấn luyện.

---

## Slide 11: Cơ sở SH và lịch tăng bậc
**Nội dung chính trên slide:** Hình các hàm cơ sở SH bậc thấp trên mặt cầu; lịch tăng dần bậc SH khi huấn luyện; mỗi vài nghìn iteration tăng 1 bậc, tối đa bậc 3 (16 hệ số/kênh màu).

**Lời thuyết trình:**
Hình bên trái minh họa các hàm cơ sở Spherical Harmonics ở những bậc thấp, được vẽ trực quan trên mặt cầu — mỗi hàm cơ sở có một hình dạng "thùy" đặc trưng riêng, giống như cách các hàm sin, cos ở các tần số khác nhau tạo thành cơ sở của chuỗi Fourier. Càng nhiều bậc SH được sử dụng, khả năng biểu diễn các biến thiên màu sắc phức tạp theo góc nhìn càng cao, nhưng đổi lại chi phí bộ nhớ và tính toán cũng tăng lên. Vì lý do đó, trong thực tế huấn luyện, người ta không dùng bậc SH tối đa ngay từ đầu, mà áp dụng một lịch tăng dần: bắt đầu chỉ với bậc 0, sau mỗi vài nghìn iteration lại tăng thêm một bậc, cho đến khi đạt bậc tối đa là bậc 3 — tương ứng với 16 hệ số cho mỗi kênh màu. Chiến lược này giúp quá trình tối ưu ổn định hơn ở giai đoạn đầu, tránh việc mô hình phải học quá nhiều tham số phức tạp khi hình dạng và vị trí Gaussian còn chưa hội tụ. Hình bên phải minh họa rõ lịch trình tăng bậc này theo số iteration. Tiếp theo, ta sẽ xem xét cách khởi tạo tập Gaussian ban đầu.

---

## Slide 12: Khởi tạo từ SfM/COLMAP
**Nội dung chính trên slide:** Gaussian khởi tạo tại point-cloud thưa từ COLMAP; scene extent = đường chéo bounding box camera; scene extent dùng làm tham chiếu ngưỡng phân tách Gaussian ở ADC.

**Lời thuyết trình:**
Trước khi bắt đầu huấn luyện, ta cần một điểm khởi đầu hợp lý cho tập Gaussian, thay vì đặt ngẫu nhiên trong không gian. Cách làm phổ biến là sử dụng kết quả từ COLMAP — một công cụ Structure-from-Motion, viết tắt SfM — để tái tạo một point cloud thưa từ tập ảnh đầu vào. Mỗi điểm trong point cloud thưa này sẽ trở thành tâm khởi tạo của một Gaussian, như minh họa ở hình bên. Ngoài point cloud, COLMAP còn cho ta vị trí các camera đã chụp cảnh, từ đó ta tính được một đại lượng gọi là scene extent — chính là độ dài đường chéo của bounding box bao quanh toàn bộ vị trí camera trong cảnh. Đại lượng này không chỉ mang tính mô tả mà còn có vai trò kỹ thuật quan trọng: nó được dùng làm giá trị tham chiếu để xác định ngưỡng phân tách Gaussian, thông qua tham số dense, trong cơ chế Adaptive Density Control mà chúng em sẽ trình bày chi tiết ở phần sau. Đến đây, chúng ta đã hoàn tất phần nền tảng cơ bản về 3D Gaussian. Tiếp theo, em xin chuyển sang phần toán học đầy đủ của Spherical Harmonics — công cụ đã được nhắc tới ở các slide trước.

---

## Slide 13: Xuất phát điểm: Phương trình Laplace
**Nội dung chính trên slide:** Hàm điều hòa $\Phi(x,y,z)$ là nghiệm phương trình Laplace $\nabla^2\Phi=0$; chuyển sang tọa độ cầu; kiểm chứng số học với $\Phi=x^2-y^2$ (điều hòa) và $\Phi=x^2+y^2$ (không điều hòa).

**Lời thuyết trình:**
Ở phần trước, chúng ta đã dùng hệ số Spherical Harmonics như một công cụ có sẵn để biểu diễn màu. Bây giờ, nhóm em muốn đi sâu hơn, giải thích SH thực sự xuất phát từ đâu về mặt toán học, vì đây là nội dung đồ án yêu cầu nắm vững nền tảng đầy đủ. Điểm xuất phát là phương trình Laplace — một phương trình đạo hàm riêng rất cơ bản trong vật lý toán, phát biểu rằng tổng các đạo hàm riêng cấp hai theo x, y, z của một hàm $\Phi$ phải bằng 0. Hàm nào thỏa mãn điều kiện này được gọi là hàm điều hòa. Khi chuyển phương trình này sang hệ tọa độ cầu, toán tử Laplace có dạng phức tạp hơn nhưng tách được thành phần bán kính r và phần góc $(\theta,\phi)$. Điều quan trọng là: chính phần góc của nghiệm phương trình Laplace sẽ dẫn tới hàm điều hòa cầu $Y_l^m$ — lý do SH được xem là cơ sở tự nhiên trên mặt cầu, tương tự như chuỗi Fourier là cơ sở tự nhiên trên đường tròn. Hình bên phải minh họa kiểm chứng số học: hàm $x^2-y^2$ thỏa điều hòa còn $x^2+y^2$ thì không. Đây chính là điểm khởi đầu cho cả chuỗi suy diễn toán học mà các slide tiếp theo sẽ trình bày.

---

## Slide 14: Hệ toạ độ cầu $(\theta,\phi)$
**Nội dung chính trên slide:** Định nghĩa góc thiên đỉnh $\theta$, góc phương vị $\phi$; quan hệ Descartes-cầu; đo $d\Omega=\sin\theta\,d\theta\,d\phi$; lưới tọa độ cầu minh họa bằng meshgrid.

**Lời thuyết trình:**
Trước khi tiếp tục suy diễn toán học, chúng ta cần thống nhất hệ tọa độ sẽ sử dụng xuyên suốt: hệ tọa độ cầu, gồm góc thiên đỉnh $\theta$ nhận giá trị từ 0 đến $\pi$, đo từ trục z xuống, và góc phương vị $\phi$ nhận giá trị từ 0 đến $2\pi$, đo quanh trục z giống như kinh độ trên quả địa cầu. Quan hệ giữa tọa độ cầu và tọa độ Descartes quen thuộc được cho bởi ba công thức lượng giác trên slide: x bằng r sin theta cos phi, y bằng r sin theta sin phi, z bằng r cos theta. Với mặt cầu đơn vị, ta chỉ cần đặt r bằng 1. Một điểm đáng chú ý là các hệ số sin theta, cos theta, sin bình theta xuất hiện ngay trong toán tử Laplace cầu ở slide trước — và đây chính là nguồn gốc của "trọng số đo" $d\Omega = \sin\theta\, d\theta\, d\phi$, một đại lượng sẽ dùng rất nhiều khi tính tích phân trực chuẩn hóa các hàm SH về sau. Hình bên phải là lưới tọa độ cầu được dựng bằng meshgrid, minh họa trực quan cách các đường theta và phi phủ đều mặt cầu đơn vị. Tiếp theo, ta bắt đầu quá trình tách biến để giải phương trình Laplace.

---

## Slide 15: Tách biến: Nghiệm phần bán kính $R(r)$
**Nội dung chính trên slide:** Giả sử $\Phi = R(r)\Theta(\theta)\Phi_m(\phi)$; tách phương trình Laplace thành phần r và phần góc, hằng số phân ly $l(l+1)$; nghiệm Euler-Cauchy $R(r) = Ar^l + B/r^{l+1}$.

**Lời thuyết trình:**
Kỹ thuật kinh điển để giải phương trình Laplace là phương pháp tách biến: ta giả sử nghiệm $\Phi$ có thể viết dưới dạng tích của ba hàm, mỗi hàm chỉ phụ thuộc vào một biến — $R(r)$ chỉ phụ thuộc bán kính, $\Theta(\theta)$ chỉ phụ thuộc góc thiên đỉnh, và $\Phi_m(\phi)$ chỉ phụ thuộc góc phương vị. Sau khi thay vào phương trình Laplace ở tọa độ cầu và nhân chia khéo léo, ta tách được phương trình thành hai nhóm: một nhóm chỉ chứa biến r, một nhóm chỉ chứa biến theta và phi. Vì hai nhóm này phụ thuộc vào các biến độc lập nhau nhưng tổng của chúng luôn bằng 0, cả hai buộc phải bằng một hằng số đối nhau — ta đặt hằng số đó là $l(l+1)$, một cách đặt khéo léo sẽ cho nghiệm đẹp về sau. Riêng phần bán kính trở thành một phương trình Euler-Cauchy, có nghiệm tổng quát gồm hai nhánh: $Ar^l$ và $B/r^{l+1}$. Nhánh đầu hữu hạn tại gốc tọa độ nên phù hợp khi mô tả trường bên trong một miền chứa gốc, còn nhánh sau hữu hạn tại vô cực nên phù hợp cho trường bên ngoài. Việc chọn nhánh nào tùy vào điều kiện biên vật lý cụ thể. Slide tiếp theo sẽ xử lý phần phương vị $\phi$.

---

## Slide 16: Tách biến: Nghiệm phần phương vị $\Phi_m(\phi)$
**Nội dung chính trên slide:** Đặt hằng số tách $m^2$; phương trình $\Phi_m''=-m^2\Phi_m$; nghiệm $\Phi_m(\phi)=e^{im\phi}$; điều kiện tuần hoàn dẫn đến $m\in\mathbb{Z}$.

**Lời thuyết trình:**
Tiếp tục quá trình tách biến, phần còn lại phụ thuộc theta và phi ở slide trước lại được tách thêm một lần nữa, lần này tách phần phụ thuộc phi ra khỏi phần phụ thuộc theta, với một hằng số phân ly thứ hai mà ta đặt là $m^2$. Phương trình vi phân thu được cho $\Phi_m(\phi)$ rất quen thuộc: đạo hàm bậc hai bằng âm $m^2$ nhân chính nó — đây chính là phương trình dao động điều hòa mà ta thường gặp trong vật lý, ví dụ như con lắc. Nghiệm tổng quát dưới dạng sóng phức là $e^{im\phi}$. Tuy nhiên, vì phi là một góc, hàm nghiệm phải tuần hoàn với chu kỳ $2\pi$, tức là giá trị tại phi phải bằng giá trị tại phi cộng $2\pi$. Điều kiện này buộc $e^{im\cdot2\pi}$ phải bằng 1, và điều đó chỉ xảy ra khi m là một số nguyên — đây chính là nguồn gốc của việc SH chỉ được định nghĩa với m nguyên, một dạng "lượng tử hóa" xuất hiện tự nhiên từ điều kiện vật lý. Hình bên phải minh họa phần thực cos(m phi) và quỹ đạo phức trên vòng tròn đơn vị cho các giá trị m khác nhau — đây chính là phiên bản "trên mặt cầu" của chuỗi Fourier quen thuộc. Slide sau sẽ giải quyết phần còn lại: phương trình theo theta.

---

## Slide 17: Phương trình Legendre liên kết
**Nội dung chính trên slide:** Đặt $x=\cos\theta$, thu được phương trình Legendre liên kết; điều kiện nghiệm hữu hạn tại cực dẫn tới $l=0,1,2,\dots$, $m=-l,\dots,l$; hàm Legendre liên kết $P_l^m(x)$ và công thức Rodrigues.

**Lời thuyết trình:**
Phần góc theta còn lại sau hai bước tách biến trước sẽ được xử lý bằng một phép đổi biến thông minh: đặt $x = \cos\theta$. Sau phép đổi biến này, phương trình vi phân cho theta biến thành một phương trình rất nổi tiếng trong toán lý gọi là phương trình Legendre liên kết, như hiển thị trên slide. Điều thú vị là: phương trình này chỉ có nghiệm hữu hạn — tức không phát tán vô cùng — tại hai điểm cực x bằng cộng trừ 1, tương ứng với cực Bắc và cực Nam của mặt cầu, khi và chỉ khi l nhận giá trị nguyên không âm 0, 1, 2, và m nằm trong khoảng từ âm l đến dương l. Đây chính là nguồn gốc toán học của điều kiện lượng tử hóa bậc l mà chúng ta đã nhắc tới nhưng chưa chứng minh ở các slide trước. Nghiệm của phương trình là hàm Legendre liên kết $P_l^m(x)$, được xây dựng từ đa thức Legendre thường $P_l(x)$ thông qua đạo hàm bậc m, còn bản thân đa thức Legendre lại có công thức tường minh gọi là công thức Rodrigues. Hình bên phải vẽ các đa thức này với các bậc l và m khác nhau. Đến đây, ta đã có đủ ba mảnh ghép: phần bán kính, phần phương vị, và phần Legendre theo theta.

---

## Slide 18: Ghép lại: Định nghĩa đầy đủ $Y_l^m$
**Nội dung chính trên slide:** Công thức đầy đủ $Y_l^m(\theta,\phi)$ ghép Legendre và $e^{im\phi}$ với hằng số chuẩn hóa; $l$ là bậc (degree), $m$ là bậc con (order), có $2l+1$ hàm ứng với mỗi $l$; hình dạng thùy phụ thuộc $(l,m)$.

**Lời thuyết trình:**
Đây là slide quan trọng nhất trong chuỗi suy diễn toán học, vì nó ghép tất cả các mảnh đã xây dựng ở ba slide trước lại thành một công thức hoàn chỉnh. Ta lấy hàm Legendre liên kết $P_l^m(\cos\theta)$ nhân với nghiệm phương vị $e^{im\phi}$, rồi nhân thêm một hằng số chuẩn hóa được chọn cẩn thận để đảm bảo tính trực chuẩn — kết quả chính là công thức định nghĩa đầy đủ của hàm điều hòa cầu $Y_l^m(\theta,\phi)$, được đóng khung trên slide. Trong công thức này, l không âm được gọi là bậc, degree, còn m nằm trong khoảng từ âm l đến dương l được gọi là bậc con, order. Với mỗi giá trị l cố định, có đúng $2l+1$ hàm $Y_l^m$ độc lập tuyến tính — ví dụ ứng với l bằng 1 có 3 hàm, đó chính là 3 hệ số SH bậc 1 mà 3DGS dùng để biểu diễn màu, như đã nhắc ở phần trước. Về mặt hình dạng, số l càng lớn thì hàm càng dao động nhanh trên mặt cầu, tạo ra nhiều "thùy" hơn; còn m quyết định các thùy đó phân bố theo phương vị như thế nào, như minh họa ở hình bên phải. Tiếp theo, ta sẽ chứng minh một tính chất toán học quan trọng của $Y_l^m$.

---

## Slide 19: $Y_l^m$ là hàm riêng của Laplace-Beltrami
**Nội dung chính trên slide:** $Y_l^m$ là eigenfunction của toán tử Laplace-Beltrami trên mặt cầu, trị riêng $-l(l+1)$; công thức $\nabla^2_{S^2}Y_l^m=-l(l+1)Y_l^m$; kiểm chứng số học bằng sai phân hữu hạn.

**Lời thuyết trình:**
Slide này làm rõ một tính chất toán học rất quan trọng của $Y_l^m$: nó là hàm riêng, eigenfunction, của một toán tử gọi là Laplace-Beltrami — chính là phần góc của toán tử Laplace 3D mà chúng ta đã gặp ở slide đầu tiên, nhưng giờ chỉ xét trên mặt cầu đơn vị. Nói theo ngôn ngữ đại số tuyến tính quen thuộc, giống như một vector riêng của ma trận khi nhân với ma trận đó chỉ bị nhân với một hằng số vô hướng, thì $Y_l^m$ khi cho qua toán tử Laplace-Beltrami cũng chỉ bị nhân với một hằng số, gọi là trị riêng eigenvalue, có giá trị đúng bằng âm $l(l+1)$. Đây không phải điều ngẫu nhiên — nó chính là hệ quả trực tiếp của bước tách biến ban đầu, khi chúng ta đặt hằng số phân ly cũng là $l(l+1)$. Điều này cho thấy toàn bộ chuỗi suy diễn toán học từ đầu đến giờ nhất quán với nhau. Hình bên phải minh họa kiểm chứng bằng số: tính toán tử Laplace-Beltrami bằng phương pháp sai phân hữu hạn trên lưới theta-phi, rồi so sánh tỉ số kết quả với giá trị lý thuyết âm $l(l+1)$ — kết quả khớp trong sai số rời rạc hóa, xác nhận đây đúng là nghiệm riêng chứ không phải một xấp xỉ. Tiếp theo là một tính chất nền tảng khác: tính trực chuẩn.

---

## Slide 20: Tính trực chuẩn của hệ $\{Y_l^m\}$
**Nội dung chính trên slide:** Hệ $\{Y_l^m\}$ trực chuẩn trên $S^2$: $\int_{S^2}Y_l^m(Y_{l'}^{m'})^*d\Omega=\delta_{ll'}\delta_{mm'}$; kiểm chứng bằng ma trận Gram xấp xỉ ma trận đơn vị.

**Lời thuyết trình:**
Tính chất toán học cuối cùng và cũng rất quan trọng để SH có thể dùng làm "cơ sở" biểu diễn hàm số trên mặt cầu là tính trực chuẩn, orthonormality. Cụ thể, tích phân của $Y_l^m$ nhân với liên hợp phức của $Y_{l'}^{m'}$ trên toàn mặt cầu, lấy theo đo $d\Omega$ mà chúng ta đã định nghĩa ở slide hệ tọa độ cầu, sẽ bằng 1 nếu hai cặp chỉ số l,m và l',m' trùng nhau, và bằng 0 nếu khác nhau — đây chính là ý nghĩa của ký hiệu delta Kronecker trong công thức. Tính chất này là hệ quả của lý thuyết Sturm-Liouville áp dụng cho phương trình Legendre ở phần theta, kết hợp với tính trực giao quen thuộc của các hàm mũ phức $e^{im\phi}$ trên một chu kỳ đầy đủ ở phần phi. Ý nghĩa thực tiễn rất lớn: nhờ trực chuẩn, ta có thể dùng $\{Y_l^m\}$ như một cơ sở kiểu Fourier trên mặt cầu, và hệ số khai triển của bất kỳ hàm nào trên mặt cầu tính được đơn giản bằng phép chiếu, không có sự chồng lấn giữa các bậc khác nhau — đây chính là lý do vì sao ta có thể tăng dần bậc SH trong quá trình huấn luyện 3DGS mà không phá vỡ các bậc đã học trước đó. Hình bên phải là ma trận Gram tính bằng số, xấp xỉ rất tốt ma trận đơn vị, xác nhận tính trực chuẩn.

---

## Slide 21: Từ SH phức sang SH thực (Real SH)
**Nội dung chính trên slide:** SH gốc chứa $e^{im\phi}$ là số phức; tổ hợp tuyến tính $Y_l^{\pm m}$ để có cơ sở thực trực chuẩn tương đương $Y_{lm}^{\text{real}}$; lý do dùng SH thực trong 3DGS: tránh số phức, giảm nửa biến, khớp trực tiếp RGB.

**Lời thuyết trình:**
Đây là slide cuối cùng của phần toán học SH mà nhóm em phụ trách, khép lại toàn bộ chuỗi suy diễn từ phương trình Laplace đến ứng dụng thực tế trong 3DGS. Vấn đề đặt ra là: định nghĩa $Y_l^m$ mà chúng ta vừa xây dựng chứa thành phần $e^{im\phi}$, vốn là một số phức — trong khi đồ họa máy tính và pipeline huấn luyện của 3DGS chỉ làm việc với số thực. Giải pháp là tổ hợp tuyến tính giữa $Y_l^m$ và liên hợp của nó $Y_l^{-m}$ theo ba trường hợp tùy m âm, bằng 0, hay dương, như công thức trên slide. Với m khác 0, phần ảo sẽ triệt tiêu lẫn nhau trong tổ hợp này, cho ra một hàm hoàn toàn thực nhưng vẫn giữ nguyên tính trực chuẩn đã chứng minh ở slide trước — tức là ta có một cơ sở thực tương đương hoàn toàn về mặt toán học. Việc dùng SH thực trong 3DGS mang lại ba lợi ích cụ thể: tránh phải xử lý số phức xuyên suốt toàn bộ pipeline vốn đã rất phức tạp, giảm một nửa số biến cần lưu trữ so với biểu diễn phức, và quan trọng nhất là khớp trực tiếp với không gian màu RGB thực trong hàm computeColorFromSH của mã nguồn. Đến đây, nhóm em xin kết thúc phần nền tảng toán học Spherical Harmonics, và xin nhường lời cho phần tiếp theo về phần B của Spherical Harmonics.
