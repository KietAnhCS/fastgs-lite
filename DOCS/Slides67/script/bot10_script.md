# Kịch bản thuyết trình — Bot 10 (Slide 174–190) — PHẦN KẾT
## Phạm vi: Phần 6 (FastGS-lite: Kết quả thực nghiệm trên Colab T4), Phần 7 (Quan sát & Kết luận)

---

## Slide 174: FastGS-lite: từ 24GB RTX 4090 xuống Colab T4 miễn phí
**Nội dung chính trên slide:** Giới thiệu fastgs-lite là bản fork đóng gói lại FastGS gốc để chạy trên phần cứng phổ thông; bảng so sánh phần cứng, CUDA extension, entry point, tín hiệu tiến trình giữa FastGS gốc và fastgs-lite.

**Lời thuyết trình:**
Sau khi đã trình bày nền tảng toán học và ba đòn bẩy tăng tốc của FastGS, phần tiếp theo em xin trình bày kết quả thực nghiệm thực tế trên chính bản fork mà nhóm xây dựng: fastgs-lite. Điểm quan trọng cần nhấn mạnh ngay từ đầu: fastgs-lite không thay đổi bất kỳ công thức toán học hay pipeline huấn luyện nào so với FastGS gốc — nó chỉ thay đổi cách đóng gói và vận hành, để bài toán vốn cần GPU 24GB như RTX 4090 có thể chạy được trên Colab T4 miễn phí, chỉ khoảng 15GB VRAM. Cụ thể, CUDA extension được vendor sẵn trong repo thay vì git submodule rời rạc, entry point được bổ sung notebook có instrumentation theo dõi tiến trình, và tín hiệu theo dõi huấn luyện được đổi thành một Score tổng hợp log mỗi 1000 iteration. Đây là nền tảng để chúng ta hiểu các số liệu thực nghiệm ở các slide tiếp theo.

---

## Slide 175: Ràng buộc phần cứng thực tế
**Nội dung chính trên slide:** Colab T4 có 14.56GB VRAM nhưng RAM hệ thống chỉ 12.7GB; VRAM peak thực tế chỉ ~1.16GB (8%), trong khi RAM peak ~9.59GB (91% giới hạn mềm 10.5GB) — RAM mới là nút thắt thật sự, không phải VRAM.

**Lời thuyết trình:**
Một phát hiện khá bất ngờ khi nhóm đo đạc thực tế trên Colab T4. Trực giác thông thường khi làm việc với GPU sẽ nghĩ nghẽn cổ chai nằm ở VRAM — bộ nhớ đồ họa. Nhưng thực tế lại hoàn toàn ngược lại. VRAM peak đo được trong suốt quá trình huấn luyện chỉ khoảng 1.16GB, tức chưa tới 8% trong tổng 14.56GB khả dụng — còn rất nhiều dư địa. Trong khi đó, RAM hệ thống — bộ nhớ CPU của máy chủ Colab — chỉ có 12.7GB, và peak sử dụng thực tế lên tới 9.59GB, tức khoảng 91% giới hạn mềm 10.5GB mà pipeline tự đặt ra. Đây là một phát hiện quan trọng mà nhóm sẽ nhắc lại kỹ hơn ở phần Quan sát cuối bài: ràng buộc thật sự khi chạy fastgs-lite trên Colab T4 chính là RAM hệ thống, chứ không phải VRAM như nhiều người vẫn nghĩ.

---

## Slide 176: Kiến trúc notebook: 7 cell ↔ 7 module
**Nội dung chính trên slide:** Bảng ánh xạ 7 cell notebook Colab với 7 module Python tương ứng (env, config, data, run, trainer+score, report, submission+deliver), giúp dễ debug và tái sử dụng ngoài Colab.

**Lời thuyết trình:**
Để notebook Colab vừa dễ chạy vừa dễ bảo trì, nhóm thiết kế kiến trúc ánh xạ một-một giữa 7 cell trong notebook và 7 module Python trong thư mục pipeline. Cell 1 gọi env.py để cài dependency và kiểm tra GPU. Cell 2 gọi config.py, tạo một object Config duy nhất dùng chung cho toàn pipeline, tránh tham số rải rác khó theo dõi. Cell 3 là data.py để tải dữ liệu và liệt kê scene. Cell 4 chạy run.py — một smoke test vài trăm iteration để phát hiện lỗi sớm trước khi train toàn bộ. Cell 5 là phần nặng nhất: trainer.py kết hợp score.py để train toàn bộ scene và log Score cùng phần trăm tiến trình. Cell 6 dùng report.py để tạo bảng và biểu đồ phân tích. Cell 7 đóng gói submission.zip, kiểm tra định dạng và tự động tải về. Thiết kế này giúp việc debug từng bước rất rõ ràng, và các module này còn dùng được cả ngoài môi trường Colab.

---

## Slide 177: Thiết lập thực nghiệm
**Nội dung chính trên slide:** 4 scene từ 2 dataset (Deep Blending: drjohnson, playroom; Tanks&Temples: train, truck), split eval với llffhold=8, cấu hình chung iterations=7000, sh_degree=3, resolution=2 khi train nhưng eval ở độ phân giải gốc.

**Lời thuyết trình:**
Bây giờ đi vào chi tiết thiết lập thực nghiệm. Nhóm chọn 4 scene từ 2 bộ dữ liệu chuẩn: drjohnson và playroom thuộc Deep Blending, train và truck thuộc Tanks and Temples. Việc chia tập train/test dùng tham số eval với llffhold bằng 8, nghĩa là cứ mỗi 8 ảnh thì lấy 1 ảnh làm test, còn lại 7 ảnh dùng để train. Cấu hình tối ưu được giữ thống nhất cho mọi scene: 7000 iteration, bậc spherical harmonics là 3, và độ phân giải giảm một nửa — resolution bằng 2 — trong lúc train để tiết kiệm tài nguyên. Tuy nhiên có một điểm cần lưu ý đặc biệt: khi đánh giá cuối cùng, nhóm lại dùng độ phân giải gốc, submission_resolution bằng 1. Sự lệch giữa train và eval này không phải là sơ suất của nhóm, mà là do protocol của cuộc thi quy định như vậy — điều này sẽ ảnh hưởng đến cách chúng ta đọc số liệu ở các slide sau.

---

## Slide 178: Bảng 1: Chất lượng cuối cùng theo scene
**Nội dung chính trên slide:** Bảng Score, PSNR, số Gaussian của 4 scene; playroom cao nhất (0.8198), train thấp nhất (0.6867); Mean Score 0.7643, PSNR trung bình 24.46dB; 2 scene indoor vượt 2 scene outdoor dù ít Gaussian hơn.

**Lời thuyết trình:**
Đây là bảng kết quả tổng hợp quan trọng nhất của phần thực nghiệm. Score trung bình trên 4 scene đạt 0.7643, PSNR trung bình 24.46dB. Nhìn theo từng scene, playroom đạt điểm cao nhất với 0.8198, tiếp theo là drjohnson 0.8027 với PSNR 27.41dB, rồi đến truck 0.7482, và thấp nhất là train chỉ 0.6867 với PSNR 19.75dB. Điều thú vị là hai scene indoor — playroom và drjohnson — đều vượt trội hơn hai scene outdoor — truck và train — mặc dù chỉ sử dụng khoảng 129 nghìn đến 174 nghìn Gaussian, ít hơn hẳn so với 187 nghìn đến 201 nghìn Gaussian của các scene outdoor. Điều này gợi ý rằng chất lượng không đơn thuần tỷ lệ thuận với số lượng Gaussian, mà phụ thuộc nhiều vào đặc tính hình học và ánh sáng của cảnh — một điểm nhóm sẽ phân tích sâu hơn ở các slide định tính tiếp theo.

---

## Slide 179: Bảng 2: Động lực huấn luyện
**Nội dung chính trên slide:** Biểu đồ Score theo iteration; hai điểm sụt rõ rệt tại iteration 3000 và 6000 trùng với opacity_reset_interval=3000 — là hiện tượng đo lường do reset opacity, không phải mô hình phân kỳ, phục hồi trong dưới 1000 iteration.

**Lời thuyết trình:**
Khi theo dõi Score qua từng 1000 iteration trên cả 4 scene, nhóm quan sát thấy hai điểm sụt giảm rất rõ rệt tại iteration 3000 và 6000, Score rơi xuống chỉ còn khoảng 0.22 đến 0.31 — nhìn thoáng qua có thể tưởng mô hình đang gặp vấn đề nghiêm trọng. Nhưng khi đối chiếu với code, nhóm phát hiện hai thời điểm này trùng khớp chính xác với tham số opacity_reset_interval bằng 3000 — tức là cứ mỗi 3000 iteration, thuật toán lại reset độ mờ của toàn bộ Gaussian về gần 0 để loại bỏ các Gaussian không đóng góp, và Score được đo ngay đúng lúc reset xảy ra. Vì vậy đây chỉ là một hiện tượng đo lường — measurement artifact — chứ hoàn toàn không phải mô hình bị phân kỳ. Chỉ trong chưa đầy 1000 iteration sau đó, Score đã phục hồi hoàn toàn. Đây là một bài học quan trọng về việc phải hiểu cơ chế bên dưới trước khi diễn giải một biểu đồ số liệu.

---

## Slide 180: Kết quả định tính: drjohnson (tốt nhất)
**Nội dung chính trên slide:** Score 0.8027, PSNR 27.41dB; hình học/tư thế/màu sắc khớp gần hoàn toàn với ground truth; lỗi còn lại ở chi tiết tần số cao và highlight phản chiếu, do learning rate của SH bậc cao bị chia thêm /20.

**Lời thuyết trình:**
Chuyển sang phân tích định tính, tức là nhìn trực tiếp vào hình ảnh render so với ảnh thật. Với scene drjohnson — scene đạt kết quả tốt nhất — Score 0.8027 và PSNR 27.41dB, hình học, tư thế camera và màu sắc tổng thể gần như khớp hoàn toàn với ground truth ở view đầu tiên. Tuy nhiên vẫn còn một số lỗi tồn tại ở các chi tiết tần số cao và các chi tiết phụ thuộc góc nhìn: những chi tiết mảnh như khe tản nhiệt hay gáy sách bị mờ, và đặc biệt highlight phản chiếu trên bề mặt gỗ mahogany gần như biến mất hoàn toàn. Nguyên nhân được nhóm truy ra trong code: các lobe spherical harmonics bậc cao — vốn chịu trách nhiệm biểu diễn hiệu ứng phụ thuộc góc nhìn như phản chiếu ánh sáng — có learning rate bị chia thêm một hệ số 20 lần, khiến chúng học rất chậm trong ngân sách 7000 iteration hạn chế.

---

## Slide 181: Kết quả định tính: train (yếu nhất — vấn đề bầu trời)
**Nội dung chính trên slide:** Score 0.6867, PSNR 19.75dB thấp nhất; lỗi rõ rệt ở vùng trời (vệt xám-trắng loang lổ thay vì xanh đồng nhất) do thiếu tín hiệu multi-view; giải thích vì sao lỗi rơi vào PSNR chứ không phải LPIPS.

**Lời thuyết trình:**
Ngược lại, scene train là scene yếu nhất với Score chỉ 0.6867 và PSNR 19.75dB — thấp nhất trong 4 scene. Khi nhìn vào ảnh render, thất bại rõ rệt nhất nằm ở vùng trời: ground truth là một màu xanh đồng nhất, nhưng ảnh render lại xuất hiện đầy vệt xám-trắng loang lổ. Trong khi đó, phần đầu máy xe lửa — nơi có texture và hiệu ứng parallax rõ ràng — lại được tái tạo khá tốt. Lý do nằm ở bản chất hình học: vùng trời không có texture, nằm ở khoảng cách gần như vô hạn, nên không cung cấp được bất kỳ tín hiệu multi-view nào để thuật toán loại bỏ các Gaussian dư thừa hoặc điều chỉnh vị trí chính xác. Điều này cũng giải thích vì sao lỗi này phản ánh rõ ở chỉ số PSNR — vốn nhạy với sai khác pixel tuyệt đối — chứ không thể hiện rõ ở LPIPS, chỉ số đo theo cảm nhận thị giác, vốn ít nhạy với vùng đồng nhất như bầu trời.

---

## Slide 182: Bảng 3: Chi phí & lưu trữ
**Nội dung chính trên slide:** Tổng thời gian train 4 scene 327.6 giây (~5.5 phút), tốc độ 85.5 iter/s, VRAM peak trung bình 0.88GB, file .ply trung bình 171.6MB, đúng 248 byte/Gaussian (dữ liệu không nén).

**Lời thuyết trình:**
Về mặt chi phí tính toán và lưu trữ, kết quả khá ấn tượng cho phần cứng miễn phí: tổng thời gian train cả 4 scene chỉ mất 327.6 giây, tức khoảng 5.5 phút, với tốc độ trung bình 85.5 iteration mỗi giây. VRAM đỉnh trung bình chỉ 0.88GB, cao nhất là 1.16GB ở scene drjohnson — rất thấp so với 14.56GB khả dụng trên T4, một lần nữa củng cố quan sát rằng VRAM không phải là nút thắt. Về lưu trữ, file point_cloud.ply trung bình nặng 171.6MB cho mỗi scene, và con số này khớp chính xác với 248 byte trên mỗi Gaussian — đây chính là kích thước của một record 3DGS hoàn toàn chưa nén, gồm 62 trường số thực float32: 3 giá trị vị trí, 3 normal, 3 hệ số DC màu, 45 hệ số bậc cao, 1 opacity, 3 scale và 4 rotation. Con số này sẽ là manh mối quan trọng cho phần Quan sát tiếp theo.

---

## Slide 183: Khả năng tái lập & giới hạn phép đo
**Nội dung chính trên slide:** Cấu hình Config chi tiết; 4 giới hạn: chạy một lần không có seed/error bar, mẫu hẹp 4 scene, ngân sách 7k chỉ bằng 1/3 so với 30k thiết kế, bỏ qua bước final_prune — kết luận Bảng 1 là baseline tái lập, không phải benchmark chính thức.

**Lời thuyết trình:**
Trước khi chuyển sang phần Quan sát và Kết luận, nhóm muốn thẳng thắn nêu rõ giới hạn của phép đo này để tránh hiểu lầm. Thứ nhất, đây chỉ là một lần chạy duy nhất, không cố định seed và không lặp lại nhiều lần, nên không có ước lượng phương sai hay error bar nào cả. Thứ hai, mẫu thử nghiệm khá hẹp, chỉ 4 scene từ 2 dataset. Thứ ba, và có lẽ quan trọng nhất, ngân sách 7000 iteration mà nhóm sử dụng chỉ bằng một phần ba so với lịch trình 30 nghìn iteration mà optimizer gốc được thiết kế để chạy đủ. Thứ tư, quá trình này bỏ qua hoàn toàn bước prune cuối cùng gọi là final_prune_fastgs. Vì vậy, kết luận công bằng nhất là: Bảng 1 nên được xem như một baseline có thể tái lập của riêng repo này, chứ không phải là kết quả benchmark chính thức để so sánh trực tiếp với số liệu công bố của FastGS hay 3DGS gốc. Sau đây, nhóm xin trình bày các quan sát sâu hơn rút ra từ những giới hạn này.

---

## Slide 184: Quan sát 1: final_prune không kịp chạy ở 7k iteration
**Nội dung chính trên slide:** Hàm final_prune_fastgs chỉ chạy trong cửa sổ 15000<iteration<30000; chạy 7000 iteration không bao giờ chạm cửa sổ này, giải thích file lớn (248B/Gaussian) và LPIPS chững ở 0.31; mọi so sánh trực tiếp với FastGS/3DGS 30k iteration đều không hợp lệ.

**Lời thuyết trình:**
Bước sang phần Quan sát, nơi nhóm tổng hợp lại những phát hiện quan trọng nhất rút ra từ thực nghiệm. Quan sát đầu tiên liên quan trực tiếp đến điều đã nhắc ở slide trước: hàm final_prune_fastgs trong code chỉ được phép kích hoạt trong một cửa sổ bảo vệ rất cụ thể — khi iteration nằm giữa 15000 và 30000. Nhưng lần chạy của nhóm chỉ dùng 7000 iteration, nghĩa là không bao giờ chạm tới cửa sổ này. Hệ quả là mô hình giao ra hoàn toàn chưa từng được prune cuối cùng. Đây chính là lời giải thích hợp lý nhất cho hai hiện tượng đã thấy: kích thước file lớn bất thường ở mức 248 byte mỗi Gaussian hoàn toàn chưa nén, và chỉ số LPIPS bị chững lại quanh mức 0.31 thay vì tiếp tục cải thiện. Hệ quả quan trọng cần nhấn mạnh: mọi so sánh trực tiếp giữa số liệu 7k-iteration này với kết quả công bố gốc của FastGS hay 3DGS — vốn chạy đủ 30 nghìn iteration — đều không hợp lệ, vì khác ngân sách huấn luyện và khác điểm dừng pipeline.

---

## Slide 185: Quan sát 2: Nút thắt là RAM hệ thống, không phải VRAM
**Nội dung chính trên slide:** Bảng đo VRAM 8% vs RAM 91%; khuyến nghị cũ "giảm resolution để tiết kiệm VRAM" không cần thiết; resolution=1 hoàn toàn khả thi, loại bỏ luôn độ lệch train/eval resolution.

**Lời thuyết trình:**
Quan sát thứ hai quay lại phát hiện đã giới thiệu từ đầu phần thực nghiệm: nút thắt thật sự trên Colab T4 là RAM hệ thống, không phải VRAM. VRAM chỉ dùng 8% trong khi RAM hệ thống áp sát 91% giới hạn mềm. Điều này có một hệ quả thực tiễn rất đáng chú ý: những khuyến nghị truyền thống kiểu "giảm độ phân giải để tiết kiệm VRAM" thực ra không cần thiết ở ngân sách huấn luyện này, vì VRAM vốn dĩ đã dư thừa rất nhiều. Ngược lại, nếu muốn tối ưu, nên tập trung quản lý RAM hệ thống. Một hệ quả thú vị khác: vì VRAM không phải nút thắt, việc train trực tiếp ở độ phân giải gốc — resolution bằng 1 — hoàn toàn khả thi về mặt tài nguyên. Điều này còn mang lại lợi ích phụ là loại bỏ luôn được độ lệch giữa resolution lúc train và lúc eval mà nhóm đã nhắc ở slide thiết lập thực nghiệm trước đó — một cải tiến khả thi cho các lần chạy tiếp theo.

---

## Slide 186: Quan sát 3: Live-score vs reported-score — khác protocol, không phải suy giảm
**Nội dung chính trên slide:** Bảng so sánh live vs reported: PSNR -1.77, SSIM -0.0533, LPIPS +0.197; nguyên nhân do LPIPS backbone khác nhau (AlexNet vs VGG) và độ phân giải khác nhau; bài học không so sánh chéo hai protocol đo khác nhau.

**Lời thuyết trình:**
Quan sát thứ ba giải thích một hiện tượng dễ gây hoang mang: Score đo được trong lúc train — gọi là live score — cao hơn hẳn so với score báo cáo cuối cùng — reported score. Cụ thể, PSNR giảm 1.77dB, SSIM giảm ít nhất 0.0533, và đặc biệt LPIPS tăng tới 0.197, tức là tệ đi đáng kể theo con số. Nhưng điều quan trọng cần khẳng định: đây không phải vì mô hình bị suy giảm chất lượng trong quá trình lưu và tải lại. Nguyên nhân thực sự đến từ sự khác biệt về protocol đo: thứ nhất, LPIPS dùng backbone mạng khác nhau — AlexNet lúc train, VGG lúc báo cáo cuối — hai backbone này cho ra thang đo khác nhau; thứ hai, độ phân giải cũng khác nhau, nửa độ phân giải lúc train so với độ phân giải gốc lúc báo cáo. Bài học rút ra rất rõ ràng: không nên so sánh chéo giữa hai protocol đo khác nhau khi đánh giá mức độ tiến bộ giữa các lần chạy — đây là lỗi diễn giải số liệu khá phổ biến cần tránh.

---

## Slide 187: Tổng kết pipeline render 3D Gaussian Splatting
**Nội dung chính trên slide:** Sơ đồ khép kín: 3D Gaussian → chiếu phối cảnh (EWA) → rasterizer khả vi theo tile → alpha blending → loss (L1+D-SSIM) → backward → Adaptive Density Control → quay lại Gaussian.

**Lời thuyết trình:**
Sau khi đã đi qua toàn bộ nền tảng toán học, các đòn bẩy tăng tốc của FastGS, và kết quả thực nghiệm chi tiết, nhóm xin tổng kết lại toàn bộ pipeline render 3D Gaussian Splatting bằng một sơ đồ duy nhất. Đây là một chu trình khép kín, lặp đi lặp lại qua từng iteration huấn luyện. Bắt đầu từ tập hợp các Gaussian 3D, được biểu diễn bởi vị trí trung bình, ma trận hiệp phương sai, độ mờ và hệ số spherical harmonics. Các Gaussian này được chiếu lên mặt phẳng ảnh qua phép chiếu phối cảnh xấp xỉ EWA, rồi được rasterize theo từng tile bằng một bộ rasterizer khả vi. Kết quả render được alpha-blend lại để ra ảnh cuối, so sánh với ảnh thật qua hàm loss kết hợp L1 và D-SSIM. Gradient được lan truyền ngược để cập nhật tham số, đồng thời điều khiển Adaptive Density Control — thêm, tách hoặc loại bỏ Gaussian — rồi vòng lặp quay trở lại điểm xuất phát. Toàn bộ những gì FastGS cải tiến chính là tăng tốc các bước trong chu trình này.

---

## Slide 188: Tổng kết cải tiến FastGS & giới hạn còn lại
**Nội dung chính trên slide:** Ba đòn bẩy chính (Multi-view score, Compact box, Sparse Adam) và tác động; 4 giới hạn/hướng phát triển: dữ liệu chưa nén, bỏ lỡ final_prune, thiếu ablation study, chưa tích hợp 4 nhánh mở rộng của FastGS gốc.

**Lời thuyết trình:**
Tổng kết lại, FastGS cải tiến 3D Gaussian Splatting gốc thông qua ba đòn bẩy chính. Multi-view score kết hợp với Compact box giúp giảm số lượng Gaussian dư thừa mà không làm giảm chất lượng đáng kể — đây là lý do vì sao trong thực nghiệm của nhóm, scene playroom và drjohnson đạt điểm cao dù dùng ít Gaussian hơn. Sparse Adam giúp giảm chi phí tính toán mỗi iteration bằng cách chỉ cập nhật những tham số thực sự liên quan, thay vì toàn bộ. Tuy nhiên, qua toàn bộ quá trình thực nghiệm, nhóm cũng nhận diện rõ những giới hạn còn tồn tại: dữ liệu Gaussian đầu ra hiện vẫn ở dạng thô, chưa được nén; ngân sách 7000 iteration của fastgs-lite bỏ lỡ hoàn toàn bước final_prune quan trọng; nhóm chưa thực hiện được ablation study để cô lập và đo đóng góp thực tế của từng đòn bẩy riêng lẻ; và bốn nhánh mở rộng của FastGS gốc — dynamic scenes, sparse view, surface reconstruction, SLAM — vẫn chưa được tích hợp vào bản fork này, đây chính là hướng phát triển tiềm năng cho tương lai.

---

## Slide 189: Bản đồ tài liệu tham khảo
**Nội dung chính trên slide:** Bảng liệt kê các tài liệu tham khảo: README.md, DOCS/fastgs-acceleration-method.md, DOCS/colab-t4-guide.md, DOCS/BOOK/ (15 chương), GitHub repo KietAnhCS/fastgs-lite.

**Lời thuyết trình:**
Trước khi khép lại bài thuyết trình, nhóm xin giới thiệu nhanh bản đồ tài liệu tham khảo, dành cho những ai muốn tìm hiểu sâu hơn sau buổi báo cáo hôm nay. File README.md là điểm vào bằng tiếng Anh, chứa bảng kết quả đầy đủ. File DOCS/fastgs-acceleration-method.md trình bày nền tảng toán học của 3DGS, ba đòn bẩy tăng tốc của FastGS, và các phép đo so sánh chi tiết giữa fastgs-lite và 3DGS. File DOCS/colab-t4-guide.md là playbook thực hành để chạy trên Colab T4, cách chống lỗi hết bộ nhớ OOM, và lộ trình build CUDA extension. Thư mục DOCS/BOOK chứa một cuốn sách hợp nhất gồm 15 chương, đi từ toán nền tảng cho tới mô hình chi phí tính toán. Và toàn bộ mã nguồn được công khai trên GitHub tại địa chỉ KietAnhCS/fastgs-lite. Sau đây là slide kết thúc bài thuyết trình của nhóm.

---

## Slide 190: Cảm ơn đã theo dõi! / Hỏi đáp (Q&A)
**Nội dung chính trên slide:** Slide kết thúc — lời cảm ơn, tiêu đề Hỏi đáp Q&A, thông tin dự án Digital Twin GS — FastGS-lite và link GitHub.

**Lời thuyết trình:**
Như vậy, nhóm em vừa trình bày xong toàn bộ nội dung đồ án: từ nền tảng toán học của 3D Gaussian Splatting, ba đòn bẩy cải tiến tốc độ của FastGS, cho đến quá trình đóng gói thành fastgs-lite để chạy được trên phần cứng miễn phí như Colab T4, cùng những kết quả thực nghiệm và bài học rút ra — trong đó có hai phát hiện đáng chú ý nhất là RAM hệ thống mới là nút thắt thật sự chứ không phải VRAM, và việc thiếu bước prune cuối do giới hạn ngân sách iteration. Em xin chân thành cảm ơn quý thầy cô trong hội đồng cùng toàn thể các bạn đã dành thời gian lắng nghe phần trình bày của nhóm hôm nay. Rất mong nhận được những câu hỏi, góp ý và nhận xét từ thầy cô để nhóm có thể hoàn thiện đồ án tốt hơn. Nhóm em xin phép được lắng nghe và trả lời các câu hỏi ạ. Xin cảm ơn.
