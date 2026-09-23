# Kịch bản thuyết trình — Bot 08 (Slide 136–154)
## Phạm vi: ADC chuyên sâu P7 (So sánh cuối 3DGS vs FastGS), Cơ chế theo dòng thời gian - phần A

---

## Slide 136: Chuỗi thời gian ADC: 3DGS gốc trên toy 2D
**Nội dung chính trên slide:** Nhắc lại luật ADC gốc của 3DGS (Kerbl 2023) — clone/split dựa trên gradient có dấu, prune cứng tức thời — minh hoạ bằng 6 mốc snapshot render và ellipse trên toy 2D.

**Lời thuyết trình:**
Bây giờ chúng ta bước sang phần so sánh cuối cùng giữa ADC gốc của 3DGS và FastGS-lite, chạy trên cùng một toy model để đảm bảo công bằng. Trước tiên nhắc lại luật gốc: 3DGS chỉ dùng gradient vị trí trung bình tích luỹ, có dấu, để quyết định clone khi Gaussian còn nhỏ, hoặc split khi Gaussian đã lớn vượt ngưỡng S_BIG. Về việc xoá, 3DGS prune ngay lập tức, không khoan nhượng, mọi Gaussian có alpha nhỏ hơn epsilon, hoặc kích thước quá lớn sau mốc reset alpha. Quan trọng là luật gốc này không có trần opacity, không có Importance đa view, không có prune kiểu xác suất, và cũng không có bước final prune ở cuối. Hình bên phải cho thấy 6 mốc thời gian từ t=0 đến T_total, với render ở hàng trên và các ellipse 1.5 sigma của từng Gaussian ở hàng dưới, kèm số lượng N và loss L1 tại mỗi mốc — đây sẽ là đường baseline để slide sau đối chiếu.

---
## Slide 137: Chuỗi thời gian ADC: FastGS-lite trên CÙNG toy model
**Nội dung chính trên slide:** FastGS-lite thêm điều kiện Importance đa view (AND), dùng gradient tuyệt đối cho split, prune multinomial có trọng số, trần opacity 0.8, và final prune bổ sung.

**Lời thuyết trình:**
Với cùng seed, cùng layout khởi tạo như slide trước, đây là cách FastGS-lite xử lý chính toy model đó. Khác biệt đầu tiên: clone và split giờ đòi hỏi thêm điều kiện Importance đa view phải vượt ngưỡng IMP_THR, tức là gradient nói "muốn đổi" thôi chưa đủ, còn phải có bằng chứng render thực sự sai ở đó. Khác biệt thứ hai, split dùng gradient tuyệt đối thay vì có dấu, để tránh hiện tượng triệt tiêu mà chúng ta sẽ phân tích kỹ ở phần sau. Về prune, FastGS không xoá ngay toàn bộ ứng viên mà lập tập C, gán trọng số theo xác suất Pruning, rồi rút mẫu multinomial không hoàn lại đúng một nửa tập đó để xoá — giữ lại phần "ít đáng xoá hơn". Sau mỗi vòng ADC, opacity bị chặn trần ở 0.8, và có thêm hai đợt final prune bổ sung tại t=1000 và t=1100. Đây chính là bộ 5 cải tiến mà slide sau sẽ mổ xẻ từng cái một.

---
## Slide 138: Bản đồ lãnh thổ: 3DGS gốc vs FastGS-lite
**Nội dung chính trên slide:** So sánh trực quan "lãnh thổ" alpha-blend chiếm ưu thế của từng Gaussian, tại 6 mốc thời gian, giữa hai phương pháp.

**Lời thuyết trình:**
Để thấy sự khác biệt không chỉ nằm ở con số mà còn ở cấu trúc không gian, chúng ta tô mỗi pixel theo Gaussian có đóng góp T nhân alpha nhân G lớn nhất tại điểm đó — gọi là bản đồ lãnh thổ. Đường đen là biên vật thể mục tiêu, còn màu sắc thể hiện Gaussian nào đang "chiếm" pixel đó trong quá trình alpha-blend. So sánh trực tiếp hai hàng tại cùng 6 mốc thời gian: hàng trên là 3DGS gốc, hàng dưới là FastGS-lite. Vì luật clone/split của FastGS bị chặn thêm bởi điều kiện Importance, và dùng gradient tuyệt đối cho split, nên ranh giới các mảng lãnh thổ, đặc biệt tại vùng biên vật thể, tách rời rõ rệt so với 3DGS gốc. Đây là bằng chứng trực quan rất thuyết phục rằng hai bộ luật ADC khác nhau không chỉ dẫn đến số lượng Gaussian khác nhau hay loss khác nhau, mà thực sự tạo ra cấu trúc phân bố tham số trong không gian hoàn toàn khác nhau. Tiếp theo ta sẽ nhìn con số định lượng cho sự khác biệt này.

---
## Slide 139: Loss, N(t) và các sự kiện clone/split/prune
**Nội dung chính trên slide:** Ba panel liên kết nhân-quả: (a) loss L1 theo thời gian cho 3 kịch bản, (b) số Gaussian N(t), (c) số sự kiện clone/split/prune tại mỗi mốc.

**Lời thuyết trình:**
Slide này ghép ba biểu đồ để thấy mối liên hệ nhân-quả xuyên suốt quá trình huấn luyện. Panel (a) là loss L1 theo thang log cho ba kịch bản: ADC 3DGS gốc, ADC FastGS-lite, và không dùng ADC với N cố định — các vạch mờ đánh dấu mốc ADC, vạch tím đứt là mốc reset alpha, vạch xanh đứt là mốc final prune của FastGS. Panel (b) vẽ số Gaussian N theo thời gian, và điều đáng chú ý nhất là FastGS-lite kết thúc với N ít hơn hẳn 3DGS gốc, nhờ cơ chế prune multinomial dọc đường và hai đợt final prune cuối, thể hiện rõ bằng các mũi tên "final prune trừ n_r" tại từng mốc. Panel (c) đếm số sự kiện clone, split, prune tại mỗi mốc ADC, đặt cạnh nhau để so sánh trực tiếp hai phương pháp. Điểm mấu chốt cần nhấn mạnh: mỗi bước nhảy của N ở panel (b) khớp chính xác với sự kiện ở panel (c), và mỗi lần N thay đổi lại kéo theo một bước ngoặt trên đường loss ở panel (a) — ba biểu đồ này thực chất kể cùng một câu chuyện từ ba góc nhìn.

---
## Slide 140: Kết quả render cuối cùng: 3 kịch bản đối chiếu
**Nội dung chính trên slide:** So sánh trực tiếp render cuối, sai số, và ellipse của ba kịch bản: ADC 3DGS gốc, ADC FastGS-lite, không ADC.

**Lời thuyết trình:**
Đây là slide chốt hạ cho phần so sánh: kết quả cuối cùng của ba kịch bản đặt song song. Hàng một là ADC 3DGS gốc, N tăng từ khởi tạo lên mức N cuối như đã thấy ở slide trước, cùng loss L1 cuối cùng. Hàng hai là ADC FastGS-lite — và đây là kết quả quan trọng nhất của toàn bộ đồ án: loss L1 cuối gần như tương đương với 3DGS gốc, nhưng số lượng Gaussian lại ít hơn đáng kể. Nói cách khác, cùng một chất lượng render nhưng FastGS-lite đạt được với chi phí bộ nhớ và tính toán thấp hơn, chính là nhờ tổng hợp của Importance đa view, prune multinomial, và final prune. Hàng ba là trường hợp không dùng ADC, giữ N bằng N0 cố định suốt quá trình, không clone, không split, không prune — và chất lượng kém hơn rõ rệt, minh chứng cho việc ADC là thành phần không thể thiếu. Mỗi hàng hiển thị render cuối, bản đồ sai số tuyệt đối, và ellipse 1.5 sigma, với viền màu đỏ, xanh, xám tương ứng ba phương pháp. Sau đây chúng ta sẽ đi sâu từng cải tiến một để hiểu vì sao chúng hoạt động.

---
## Slide 141: Cải tiến 1: logic AND với Importance, số liệu thật trên toy
**Nội dung chính trên slide:** Minh hoạ cụ thể bằng số liệu thật cách Importance đa view lọc lại quyết định densify của gradient, chia thành 4 nhóm quyết định.

**Lời thuyết trình:**
Cải tiến đầu tiên chúng ta phân tích kỹ bằng số liệu thật, không chỉ công thức. Nhắc lại: FastGS chỉ densify khi cả gradient và Importance đa view đồng thời vượt ngưỡng. Ở đây minh hoạ bằng một mốc thời gian cụ thể lấy từ log thật, thường chọn t=300 hoặc mốc có nhiều Gaussian bị chặn nhất. Đầu tiên xây mặt nạ lỗi bằng cách chuẩn hoá minmax của sai số trung bình các kênh màu, ngưỡng tại 0.1. Sau đó Importance của mỗi Gaussian được tính là số pixel lỗi rơi vào footprint hữu hình của nó, sau khi đã lọc qua ba cửa: bán kính 3 sigma, trọng số ít nhất 1/255, và transmittance còn ít nhất 10 mũ âm 4. Ngưỡng IMP_THR đặt ở 5. Kết quả chia thành bốn nhóm rõ ràng trên hình: nhóm xanh lá là cả hai phương pháp đều muốn densify; nhóm đỏ với dấu X là trường hợp gradient của 3DGS muốn đổi nhưng FastGS chặn lại vì Importance thấp — tức gradient "kêu" nhưng ảnh không thực sự sai ở đó; nhóm cam chỉ FastGS densify nhờ split theo gradient tuyệt đối; và nhóm xám không ai densify.

---
## Slide 142: Cải tiến 2: split theo tổng trị tuyệt đối gradient thay vì tổng có dấu
**Nội dung chính trên slide:** Chứng minh bằng ví dụ cụ thể hiện tượng gradient có dấu bị triệt tiêu, và tại sao dùng tổng tuyệt đối giải quyết được vấn đề này.

**Lời thuyết trình:**
Cải tiến thứ hai giải quyết một điểm yếu tinh tế của gradient có dấu: khi một Gaussian to bao phủ một vùng phức tạp, các gradient pixel bên trong footprint của nó có thể kéo về nhiều hướng khác nhau trong cùng một khung hình, và khi cộng vector lại thì chúng triệt tiêu lẫn nhau. Slide chọn ra một Gaussian cụ thể i sao để minh hoạ: tại mốc t, chuẩn của tổng có dấu rất nhỏ so với tổng các trị tuyệt đối — tỉ lệ chênh lệch được ghi rõ trên hình, ví dụ gấp k lần. Hệ quả tích luỹ theo thời gian là gradient có dấu chuẩn hoá theo tau không vượt ngưỡng, nên 3DGS bỏ qua Gaussian này hoàn toàn; trong khi gradient tuyệt đối chuẩn hoá theo tau_abs lại vượt ngưỡng, nên FastGS quyết định split nó. Thí nghiệm chạy tiếp 80 bước Adam sau đó cho thấy rõ ràng: kịch bản không split giống 3DGS chỉ cải thiện loss rất ít, còn kịch bản split thành 2 Gaussian con của FastGS giảm loss ngay lập tức sau khi split rồi tiếp tục giảm đều qua 80 bước tiếp theo. Panel (b) là biểu đồ cột so sánh hai loại gradient cho toàn bộ Gaussian to, kèm hai đường ngưỡng tương ứng.

---
## Slide 143: Cải tiến 3: prune cứng vs multinomial, áp dụng trên toy
**Nội dung chính trên slide:** So sánh cách 3DGS xoá toàn bộ ứng viên ngay lập tức với cách FastGS rút mẫu multinomial có trọng số để giữ lại một phần.

**Lời thuyết trình:**
Cải tiến thứ ba nằm ở bước prune. Cả hai phương pháp đều xuất phát từ cùng một tập ứng viên C, gồm những Gaussian có alpha nhỏ hơn epsilon hoặc kích thước vượt ngưỡng S_HUGE. Nhưng cách xử lý khác hẳn nhau: 3DGS gốc xoá ngay lập tức toàn bộ tập C, không cân nhắc gì thêm. FastGS thì thận trọng hơn — gán mỗi ứng viên một trọng số w tỉ lệ nghịch với xác suất Pruning của nó, rồi rút mẫu theo phân phối multinomial đúng một nửa kích thước tập C, không hoàn lại, và chỉ xoá phần giao giữa C và tập được rút. Nói cách khác, những ứng viên "ít đáng xoá hơn" theo mô hình Pruning có cơ hội được giữ lại thêm một nhịp nữa, tránh xoá nhầm những Gaussian có thể còn hữu ích. Sau bước prune này, FastGS còn áp thêm trần opacity 0.8, và trên hình cho thấy sai khác ảnh lớn nhất giữa trước và sau prune là rất nhỏ, tức là việc giữ lại một phần ứng viên không làm tổn hại chất lượng render tức thời.

---
## Slide 144: Cải tiến 4: trần opacity alpha nhỏ hơn hoặc bằng 0.8
**Nội dung chính trên slide:** Giải thích cơ chế "tràn" opacity làm chôn gradient của lớp sau, và tại sao chặn alpha ở 0.8 giữ được tín hiệu gradient.

**Lời thuyết trình:**
Cải tiến thứ tư xử lý một hiện tượng gọi là "tràn" opacity. Khi một Gaussian phía trước có alpha rất cao, gần bằng 1, thì transmittance còn lại cho các lớp phía sau gần như bằng 0, tính theo công thức T bằng 1 trừ alpha front nhân G front. Cụ thể tại tâm Gaussian, nếu alpha front bằng 0.99 thì T chỉ còn 0.01; nhưng nếu chặn alpha ở 0.8 thì T còn tới 0.20 — gấp 20 lần ánh sáng còn lại truyền cho Gaussian phía sau. Điều này quan trọng vì đạo hàm loss theo tham số của lớp sau tỉ lệ thuận với T nhân alpha nhân G của lớp sau đó — nếu T gần 0, gradient của lớp sau gần như bị "chôn" vĩnh viễn, nó không bao giờ được cập nhật nữa dù render vẫn sai ở đó. Vì vậy trần alpha 0.8 đảm bảo luôn giữ lại ít nhất 20 phần trăm tín hiệu gradient cho các lớp phía sau. Trên hình là histogram opacity cuối cùng, đếm số Gaussian vượt 0.8 và vượt 0.99 ở cả hai phương pháp, cùng bản đồ T cuối cho thấy tỉ lệ phần trăm pixel bị "chặn hết" ở mỗi bên.

---
## Slide 145: Cải tiến 5: final prune trên toy — khép lại chương ADC
**Nội dung chính trên slide:** Bước final prune bổ sung sau khi kết thúc densify, xoá theo alpha thấp hoặc xác suất Pruning cao, khép lại toàn bộ chương so sánh 3DGS vs FastGS.

**Lời thuyết trình:**
Cải tiến cuối cùng, cũng là slide khép lại chương so sánh ADC. Sau khi densify kết thúc ở t bằng 900, FastGS-lite chạy thêm bước final prune tại hai mốc t=1000 và t=1100, với điều kiện xoá là alpha nhỏ hơn 0.1, hoặc xác suất Pruning lớn hơn 0.9. Điều đáng chú ý là 3DGS gốc hoàn toàn không có bước này, nên đường N của nó phẳng lì trong khoảng từ 900 đến 1200; trong khi FastGS giảm theo kiểu "hai bậc thang" rõ rệt tại đúng hai mốc đó. Biểu đồ scatter alpha và Pruning cho thấy một vùng hình chữ L bị xoá — tức là alpha thấp đồng thời Pruning cao — được đánh dấu bằng các chữ X màu đỏ. Kết quả cuối cùng: loss L1 của FastGS-lite tương đương với 3DGS gốc nhưng số lượng Gaussian nhỏ hơn hẳn, cho ra một mô hình gọn nhất trong ba kịch bản đã so sánh. Đây chính là tổng kết cho toàn bộ chương ADC chuyên sâu. Bây giờ chúng ta chuyển sang một góc nhìn khác — nhìn toàn bộ cơ chế ADC theo dòng thời gian thực tế của quá trình huấn luyện 30000 vòng.

---
## Slide 146: ADC nhìn theo dòng thời gian: lịch chạy đầy đủ
**Nội dung chính trên slide:** Toàn cảnh 4 giai đoạn chồng lấp của ADC trong vòng đời huấn luyện thật, T=30000 vòng.

**Lời thuyết trình:**
Phần tiếp theo chuyển từ toy model sang nhìn cơ chế ADC theo đúng dòng thời gian của quá trình huấn luyện thật, kéo dài 30000 vòng lặp. Có bốn giai đoạn diễn ra chồng lấp lên nhau, không tách rời tuần tự. Giai đoạn (a) là tích luỹ gradient, diễn ra ở mọi vòng khi t nhỏ hơn 15000 — đây là nền tảng chạy xuyên suốt nửa đầu quá trình huấn luyện. Giai đoạn (b) là densify và prune, chỉ kích hoạt khi t lớn hơn 500 và t chia hết cho 100, tạo ra tổng cộng 144 mốc can thiệp, từ t=600 đến t=14900. Giai đoạn (c) là reset opacity, xảy ra thưa hơn nhiều, chỉ tại 4 mốc cố định 3000, 6000, 9000, 12000, khi t chia hết cho 3000 và vẫn nhỏ hơn 15000. Cuối cùng giai đoạn (d) là "N đóng băng", bắt đầu từ t=15000 trở đi, lúc này số lượng Gaussian không còn thay đổi nữa, chỉ còn tối ưu các tham số liên tục như vị trí, scale, quaternion, alpha và hệ số cầu điều hoà. Slide sau sẽ đi sâu vào cơ chế tích luỹ gradient ở giai đoạn (a).

---
## Slide 147: Tích luỹ gradient: accum, denom và gradient trung bình
**Nội dung chính trên slide:** Cơ chế cộng dồn accum và denom giữa hai lần densify liên tiếp, công thức tính gradient trung bình, và ví dụ mô phỏng khi Gaussian bị che khuất.

**Lời thuyết trình:**
Đi sâu vào giai đoạn tích luỹ gradient. Mỗi Gaussian i duy trì hai bộ đếm cộng dồn giữa hai lần densify liên tiếp, cách nhau đúng chu kỳ 100 vòng. Bộ đếm accum cộng dồn chuẩn của gradient vị trí mỗi khi Gaussian được chiếu vào khung hình, còn denom chỉ đơn giản đếm số lần Gaussian đó thực sự xuất hiện, tức không bị che khuất. Ký hiệu chỉ báo 1 dưới i tại thời điểm t cho biết Gaussian i có được nhìn thấy tại vòng đó hay không — nếu bị che, cả accum lẫn denom đều không tăng. Gradient trung bình cuối cùng đơn giản là accum chia cho denom. Ví dụ mô phỏng trên hình cho thấy rất rõ: khi Gaussian bị che liên tục trong 12 vòng, từ t=40 đến t=52, cả hai đường accum và denom đều đi ngang phẳng lì trong khoảng đó. Tại thời điểm cuối chu kỳ t=100, gradient trung bình đạt khoảng 3 đến 4 nhân 10 mũ âm 4, vượt ngưỡng tau_grad bằng 2 nhân 10 mũ âm 4, nên Gaussian này trở thành ứng viên densify. Ba panel trên hình lần lượt thể hiện chuỗi hiển thị và gradient mỗi vòng, quá trình cộng dồn, và so sánh gradient trung bình với ngưỡng.

---
## Slide 148: Hiện tượng triệt tiêu gradient (gradient cancellation)
**Nội dung chính trên slide:** Phân tích toán học tại sao gradient có dấu bị triệt tiêu khi Gaussian đúng vị trí nhưng sai kích thước, dẫn đến điểm mù trong tiêu chí densify của 3DGS gốc.

**Lời thuyết trình:**
Đây là phân tích chi tiết cho một điểm mù quan trọng của 3DGS gốc mà chúng ta đã nhắc ở phần trước, giờ chứng minh bằng toán học cụ thể. Điểm mấu chốt: Inria chỉ tích luỹ chuẩn của tổng có dấu của gradient, chứ không dùng tổng các chuẩn từng điểm ảnh. Với hàm loss là tổng bình phương sai số, và mô hình cường độ Gaussian dạng mũ, đạo hàm theo vị trí mu là tổng theo x của sai số nhân cường độ chia sigma bình phương nhân vector lệch x trừ mu. Xét hai trường hợp: trường hợp (a) khi Gaussian lệch vị trí, các vector gradient tại từng pixel đều thiên về cùng một hướng, nên tổng có dấu lớn, gần bằng tổng các chuẩn — không có vấn đề gì. Nhưng trường hợp (b), khi Gaussian đúng vị trí nhưng sai kích thước, tức to hơn ground truth và vẫn đồng tâm, các vector gradient lại đối xứng quanh tâm và triệt tiêu lẫn nhau, khiến chuẩn của tổng có dấu rất nhỏ so với tổng các chuẩn, tỉ số chỉ khoảng 10 mũ âm 1 đến 10 mũ âm 2 trong ví dụ minh hoạ. Hệ quả là gradient trung bình nhỏ, Gaussian này không được densify dù ảnh dựng vẫn sai rõ ràng — đây chính là động lực cho cải tiến dùng gradient tuyệt đối mà FastGS áp dụng.

---
## Slide 149: Lịch sử gradient trung bình của một Gaussian qua nhiều vòng
**Nội dung chính trên slide:** Mô phỏng 15000 vòng cho 5 loại Gaussian khác nhau, thể hiện dạng răng cưa của gradient trung bình do bị reset thống kê mỗi 100 vòng.

**Lời thuyết trình:**
Slide này theo dõi lịch sử gradient trung bình của nhiều loại Gaussian khác nhau xuyên suốt 15000 vòng huấn luyện mô phỏng. Có 5 loại được đưa vào so sánh: Gaussian nằm ở viền vật thể, Gaussian gần viền, Gaussian ở vùng nền phẳng, Gaussian thỉnh thoảng bị che khuất, và một Gaussian mới sinh ra từ clone tại t=3000. Điểm quan trọng cần lưu ý: vì accum và denom bị xoá về 0 tại mỗi mốc densify, tức mỗi khi t chia hết cho 100, nên đường gradient trung bình theo thời gian có dạng răng cưa đặc trưng, không mượt. Với Gaussian ở viền vật thể, xu hướng chung là gradient trung bình giảm dần theo hàm mũ với hằng số thời gian khoảng 5500, nhưng có những xung tăng đột biến ngay sau mỗi lần reset opacity, tại các mốc 3000, 6000, 9000, 12000 — vì reset opacity làm ảnh dựng tạm thời sai lệch nhiều hơn. Với Gaussian "thỉnh thoảng bị che", do xác suất nhìn thấy chỉ 0.3, denom nhỏ nên gradient dao động rất mạnh, thậm chí có thể vượt ngưỡng tau_grad một cách giả tạo, không phản ánh đúng nhu cầu densify thực sự.

---
## Slide 150: Mặt phẳng quyết định: giữ nguyên / clone / split
**Nội dung chính trên slide:** Biểu diễn không gian hai chiều gradient và kích thước, chia thành ba vùng quyết định, cùng các điểm ví dụ minh hoạ ranh giới.

**Lời thuyết trình:**
Slide này tổng hợp trực quan toàn bộ logic quyết định densify thành một mặt phẳng hai chiều. Công thức nhắc lại: clone xảy ra khi gradient trung bình vượt tau_grad, đồng thời kích thước lớn nhất của Gaussian nhỏ hơn hoặc bằng delta nhân extent; split xảy ra khi cùng điều kiện gradient nhưng kích thước lại vượt quá ngưỡng đó. Ở đây delta chính là percent_dense, mặc định 0.01. Trục ngang của biểu đồ là gradient trung bình theo thang log, trục dọc là tỉ lệ kích thước lớn nhất trên extent cũng theo thang log. Ngoài ra còn có một ngưỡng prune riêng theo kích thước, khi max scale vượt 0.1 lần extent, được đánh dấu vùng gạch chéo phía trên cùng của biểu đồ. Sáu điểm ví dụ A đến F được đặt trên mặt phẳng để minh hoạ các trường hợp biên: điểm E nằm đúng trên ranh giới của cả hai điều kiện nên rơi vào vùng clone; điểm F thì vượt cả ngưỡng gradient lẫn ngưỡng prune, nghĩa là nó sẽ bị split trước rồi ngay bước tiếp theo lại bị prune do kích thước con sau split vẫn còn quá lớn.

---
## Slide 151: Đo phạm vi cảnh (scene extent) và chuẩn hoá ngưỡng
**Nội dung chính trên slide:** Cách tính cameras_extent từ vị trí camera, và tại sao các ngưỡng densify/prune phải chuẩn hoá theo phạm vi thật của cảnh chứ không theo pixel.

**Lời thuyết trình:**
Một chi tiết kỹ thuật quan trọng thường bị bỏ qua: các ngưỡng delta mà chúng ta vừa nói không phải là số tuyệt đối cố định, mà được chuẩn hoá theo phạm vi thật của cảnh 3D. Inria tính cameras_extent bằng 1.1 lần khoảng cách xa nhất từ một camera bất kỳ tới tâm của toàn bộ tập camera. Đây là phép chuẩn hoá theo không gian thực, không phải theo pixel — nghĩa là cùng một hệ số delta bằng 0.01, nhưng ngưỡng tuyệt đối delta nhân extent sẽ thay đổi tuỳ theo cảnh đó lớn hay nhỏ trong thực tế. Có ba mức ngưỡng được thiết lập theo tỉ lệ cố định 1 trên 10 trên 100: mức 0.01 lần extent là ranh giới giữa clone và split, mức 0.1 lần extent là ranh giới prune theo kích thước, và bản thân extent là kích thước toàn cảnh. Hình minh hoạ bên trái vẽ các camera bao quanh đám mây điểm với bán kính bằng đúng extent, còn bên phải là một cây thước trực quan thể hiện ba mức tỉ lệ 0.01, 0.1 và 1 lần extent, cùng hai Gaussian ví dụ được vẽ đúng theo hai ngưỡng đó để thấy chênh lệch kích thước tuyệt đối rõ ràng.

---
## Slide 152: Quét ngưỡng: đánh đổi thêm nhiều vs thêm ít
**Nội dung chính trên slide:** Thí nghiệm quét tau_grad và delta trên 10000 Gaussian mô phỏng, thể hiện đánh đổi giữa số lượng Gaussian được densify và chi phí tính toán.

**Lời thuyết trình:**
Slide này trả lời câu hỏi thực tế: nếu thay đổi các ngưỡng thì điều gì xảy ra. Thí nghiệm mô phỏng 10000 Gaussian, với gradient trung bình tuân theo phân phối log-chuẩn có trung vị khoảng 4 nhân 10 mũ âm 5, và tỉ lệ kích thước lớn nhất trên extent cũng log-chuẩn với trung vị khoảng 0.004. Với ngưỡng mặc định tau_grad bằng 2 nhân 10 mũ âm 4, chỉ một phần nhỏ Gaussian, nằm ở đuôi dài của phân phối, vượt ngưỡng và được đưa vào densify. Khi quét tau_grad trong khoảng từ 5 nhân 10 mũ âm 5 đến 10 mũ âm 3, ta thấy rõ đánh đổi cốt lõi: ngưỡng càng thấp thì tỉ lệ phần trăm Gaussian được densify tăng rất mạnh, đường cong dốc đứng quanh vùng giá trị mặc định — nghĩa là ngưỡng thấp cho thêm nhiều chi tiết nhưng tốn bộ nhớ và tính toán hơn, còn ngưỡng cao thì tiết kiệm nhưng dễ bỏ sót chi tiết. Tương tự, khi quét delta trong khoảng 0.002 đến 0.05 và giữ tau_grad cố định, delta càng lớn thì tỉ lệ clone tăng còn tỉ lệ split giảm, hai đường phần trăm cắt nhau gần như đối xứng quanh giá trị mặc định 0.01.

---
## Slide 153: Clone: sao chép nguyên trạng để tăng chi tiết
**Nội dung chính trên slide:** Cơ chế clone tạo bản sao y nguyên tham số cho Gaussian nhỏ nhưng đang dưới tái tạo, kèm ví dụ số liệu cụ thể.

**Lời thuyết trình:**
Bây giờ đi vào chi tiết cơ chế clone. Điều kiện kích hoạt: gradient trung bình vượt tau_grad, và đồng thời kích thước lớn nhất vẫn nhỏ hơn hoặc bằng delta nhân extent — tức là một Gaussian còn nhỏ nhưng đang trong tình trạng dưới tái tạo, chưa đủ khả năng biểu diễn chi tiết vùng nó phụ trách. Ví dụ script cụ thể trên hình: Gaussian có vị trí mu bằng 0.1, 0.05, kích thước s bằng 0.006, 0.003, góc quay 30 độ, và alpha bằng 0.6. Phép clone rất đơn giản: theta mới bằng chính theta cũ, sao chép y nguyên toàn bộ các trường bao gồm vị trí, scale, quaternion, alpha, và cả hệ số cầu điều hoà — không thay đổi tỉ lệ, không thay đổi gì cả. Kết quả là ngay sau clone, hai Gaussian con hoàn toàn trùng khớp vị trí với nhau, và chỉ tách rời dần dần nhờ gradient khác nhau tại các vòng tối ưu tiếp theo, do mỗi bản sao "nhìn thấy" một phần khác nhau của sai số. Về mặt kế toán, mỗi lần clone làm tổng số Gaussian N tăng đúng 1, từ 1 thành 2. Hình bên phải minh hoạ rõ trạng thái trước và sau clone trên một cạnh chi tiết của vật thể.

---
## Slide 154: Split: hình học phân tách Gaussian quá to
**Nội dung chính trên slide:** Cơ chế split sinh hai Gaussian con nhỏ hơn từ một Gaussian gốc quá to, với công thức xoay hệ toạ độ và ví dụ số liệu cụ thể.

**Lời thuyết trình:**
Slide cuối cùng của phần này giải thích cơ chế split — trường hợp đối lập với clone. Điều kiện kích hoạt: gradient trung bình vẫn vượt tau_grad, nhưng lần này kích thước lớn nhất đã vượt quá delta nhân extent, tức Gaussian đang quá to để gánh một vùng có nhiều chi tiết. Cách làm: sinh hai mẫu cục bộ ngẫu nhiên theo phân phối chuẩn với hiệp phương sai là đường chéo bình phương của scale gốc, rồi xoay các mẫu này về hệ toạ độ thế giới bằng ma trận xoay theo góc quaternion của Gaussian gốc, cộng thêm vào vị trí tâm mu. Scale của mỗi Gaussian con bằng scale gốc chia cho hệ số SPLIT_FACTOR bằng 1.6, còn góc quay giữ nguyên. Ví dụ script cụ thể: scale gốc 0.06 và 0.02, góc 40 độ, cho ra scale con khoảng 0.0375 và 0.0125 — diện tích mỗi Gaussian con chỉ còn xấp xỉ 0.39 lần diện tích gốc. Gaussian gốc bị xoá hoàn toàn sau khi split, trong khi alpha và hệ số cầu điều hoà được giữ nguyên cho cả hai con. Về kế toán, split tạo thêm 2 Gaussian con và xoá đi 1 Gaussian gốc, nên tổng N tăng ròng đúng 1 mỗi lần split diễn ra — cùng mức tăng như clone, nhưng cách phân bố không gian hoàn toàn khác. Đến đây, chúng ta đã đi hết toàn bộ cơ chế theo dòng thời gian của ADC, xin chuyển lời cho phần tiếp theo.

