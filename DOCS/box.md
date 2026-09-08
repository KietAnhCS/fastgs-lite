# Phần IV — Bài tập end-to-end: compact box và số tile chạm

> Bài tập tự luyện cho §4.1–§4.5 của [`fastgs-acceleration-method.md`](fastgs-acceleration-method.md).
> Cấu trúc: **Đề bài → Câu hỏi → Bảng công thức → Đáp án chi tiết**.
> Toàn bộ số liệu được thiết kế để tính tay được (eigenvalue chẵn, tâm splat nằm giữa tile).
> Khuyến nghị: làm hết phần câu hỏi rồi mới cuộn xuống đáp án.

---

## ĐỀ BÀI

Một Gaussian $G$ đã được chiếu xuống mặt phẳng ảnh của một camera. Cho:

### A. Covariance 2D đã chiếu (kết quả của công thức (5), §1.2)

$$\Sigma'=\begin{pmatrix}117 & 54\\[2pt] 54 & 36\end{pmatrix}\quad[\text{px}^2]$$

### B. Các đại lượng khác

| Đại lượng | Ký hiệu | Giá trị |
|---|---|---|
| Tâm splat trên ảnh | $p=(p_x,p_y)$ | $(120.0,\ 88.0)$ px |
| Opacity (sau sigmoid) | $o$ | $1.0$ |
| Hệ số compact box | `mult` | $0.5$ (mặc định, `arguments/__init__.py:96`) |
| Kích thước tile | $B_U=B_V$ | $16$ px |
| Kích thước ảnh | | đủ lớn, **bỏ qua clamp biên** |

### C. Quy ước đánh số tile

Tile $(\text{col},\text{row})$ phủ vùng $x\in[16\,\text{col},\,16(\text{col}+1))$, $y\in[16\,\text{row},\,16(\text{row}+1))$.
Ví dụ tâm splat $(120,88)$ nằm trong tile $(7,5)$ vì $120/16=7.5$ và $88/16=5.5$.

---

## CÂU HỎI

### Câu 1 — Hình học của splat (§4.1)

a) Tính $\det\Sigma'$, $\text{mid}=\tfrac12\text{tr}\,\Sigma'$, rồi $\lambda_{\max},\lambda_{\min}$ theo `forward.cu:239-240`.

b) Suy ra $\sigma_{\max},\sigma_{\min}$ và tỉ lệ trục $\rho=\sigma_{\max}/\sigma_{\min}$.

c) Theo §4.1, hộp vuông của 3DGS lãng phí bao nhiêu lần so với ellipse $3\sigma$? Kiểm tra công thức bằng phép thử $\rho=1$.

### Câu 2 — Conic (§4.2)

Tính $M=\Sigma'^{-1}$, tức bộ ba $(A,B,C)=$ `con_o.x, con_o.y, con_o.z`.

Sau đó kiểm tra hai điều:
- $\text{disc}=B^2-AC$ có thoả $\text{disc}<0$ không (điều kiện không suy biến, `auxiliary.h:305`)?
- Chứng minh quan hệ $\text{disc}=-1/\det\Sigma'$ và xác nhận bằng số.

### Câu 3 — Ngưỡng level-set $t$ (§4.2–§4.3)

a) Tính $t=\texttt{mult}\cdot 2\ln(255\,o)$ cho cấu hình đề bài.

b) Lập bảng $t$ cho các trường hợp: $(\texttt{mult},o)\in\{(1.0,\,1.0),\ (0.5,\,1.0),\ (0.5,\,0.1),\ (0.5,\,0.02)\}$.

c) Giải thích tại sao `mult` $=0.5$ làm **diện tích** hộp giảm 50% chứ không phải 75%.

### Câu 4 — Bounding box của compact box (§4.2)

a) Tính nửa cạnh $\text{half}_x=\sqrt{t\,\Sigma'_{11}}$ và $\text{half}_y=\sqrt{t\,\Sigma'_{22}}$.

b) Suy ra $\text{bbox}_{\min}$, $\text{bbox}_{\max}$ theo pixel.

c) Hộp này chạm những cột tile nào, hàng tile nào? Tổng cộng bao nhiêu tile nếu **chỉ dùng hộp** (chưa lọc ellipse)?

### Câu 5 — Điểm tiếp tuyến `x_term` / `y_term` (§4.2)

a) Tính $x_{\text{term}}$ và $y_{\text{term}}$ theo `auxiliary.h:316-319` (nhớ phép lật dấu theo $B$).

b) **Câu hỏi hiểu bản chất:** $x_{\text{term}}$ là toạ độ $x$ của điểm nào trên ellipse? Chứng minh bằng cách đạo hàm điều kiện cực trị, rồi đối chiếu số với kết quả câu 4a.

### Câu 6 — Giao ellipse với một đường thẳng (§4.2)

Dùng công thức nghiệm bậc 2 của `computeEllipseIntersection`:

$$v_\pm(u)=\frac{-B\,h\ \pm\ \sqrt{\text{disc}\cdot h^2+t\,k}}{k}+p_v,\qquad h=u-p_u$$

a) Cắt bằng đường **thẳng đứng** $x=128$ (biên giữa tile cột 7 và 8). Tính khoảng $y$ mà ellipse còn tồn tại. Khoảng đó chạm những hàng tile nào?

b) Trên đường $x=128$, giá trị dưới căn là bao nhiêu? Nếu thay $x=150$ thì sao — kết luận gì?

### Câu 7 — Đếm tile thật bằng `processTiles` (§4.4)

`processTiles` duyệt theo **trục có span ngắn hơn** (`isY = y_span < x_span`), mỗi lát lấy giao chính xác với ellipse.

a) Với kết quả câu 4c, trục nào ngắn hơn? Vậy code duyệt theo hàng hay theo cột?

b) Với **mỗi lát**, tính khoảng toạ độ theo trục còn lại, rồi quy ra số tile. Nhớ quy tắc: nếu điểm cực trị rơi **vào trong** lát thì lấy thẳng biên bbox; nếu không thì lấy $\min/\max$ tại **hai biên của lát**.

c) Tổng $K_{\text{fastgs}}$ là bao nhiêu? So với câu 4c, bước lọc ellipse loại bỏ bao nhiêu tile?

### Câu 8 — Đường so sánh 3DGS gốc (§4.1)

a) Tính `my_radius` $=\lceil 3\sqrt{\lambda_{\max}}\rceil$ (`forward.cu:241`).

b) Dùng công thức rời rạc của `getRect` bên 3DGS gốc:

$$\text{rect}_{\min}=\Bigl\lfloor\frac{p-r}{16}\Bigr\rfloor,\qquad \text{rect}_{\max}=\Bigl\lfloor\frac{p+r+15}{16}\Bigr\rfloor$$

(biên trên **loại trừ**) để tính $K_{\text{3dgs}}$.

c) Tính $R_{\text{tile}}=K_{\text{fastgs}}/K_{\text{3dgs}}$.

### Câu 9 — Đối chiếu với công thức tiệm cận (§4.5)

a) Tính $R_{\text{tile}}$ theo công thức liên tục của §4.5:

$$R_{\text{tile}}\approx\frac{\pi}{4}\cdot\frac{4t\sqrt{\Sigma'_{11}\Sigma'_{22}}}{36\,\lambda_{\max}}$$

b) So với kết quả đo ở câu 8c. Chênh lệch theo hướng nào? **Giải thích tại sao** — nêu đúng cơ chế gây ra chênh lệch.

c) Bước lọc ellipse ở câu 7c loại bỏ bao nhiêu phần trăm? So với con số tiệm cận $1-\pi/4\approx21.5\%$. Vì sao khác?

### Câu 10 — Quét tham số (§4.3)

Lặp lại câu 4 + câu 7 (đếm tile đầy đủ) cho hai cấu hình:

| | `mult` | $o$ |
|---|---|---|
| (i) | $1.0$ | $1.0$ |
| (ii) | $0.5$ | $0.1$ |

a) Điền bảng: $t$, $\text{half}_x$, $\text{half}_y$, số tile của hộp, $K_{\text{fastgs}}$, $R_{\text{tile}}$.

b) Từ cấu hình gốc ($\texttt{mult}=0.5$) sang (i) ($\texttt{mult}=1.0$): diện tích hộp tăng gấp đôi, nhưng $K$ tăng bao nhiêu phần trăm? Giải thích.

c) Từ $o=1.0$ xuống $o=0.1$: $t$ giảm bao nhiêu %, $K$ giảm bao nhiêu %? Khớp với nhận định "**lợi ích từ opacity thấp là thật nhưng khiêm tốn**" ở §4.3 không?

d) Câu hỏi khái niệm: ở cấu hình (i), $\texttt{mult}=1.0$ có đưa hộp về đúng hành vi 3DGS gốc không? Vì sao?

---

## BẢNG CÔNG THỨC (được phép dùng)

| Đại lượng | Công thức | Vị trí code |
|---|---|---|
| Eigenvalue | $\lambda_{1,2}=\text{mid}\pm\sqrt{\text{mid}^2-\det}$, $\text{mid}=\tfrac12\text{tr}\Sigma'$ | `forward.cu:239-240` |
| Bán kính 3DGS | $r=\lceil 3\sqrt{\lambda_{\max}}\rceil$ | `forward.cu:241` |
| Conic | $M=\Sigma'^{-1}=\dfrac{1}{\det}\begin{pmatrix}\Sigma'_{22}&-\Sigma'_{12}\\-\Sigma'_{12}&\Sigma'_{11}\end{pmatrix}$ | `forward.cu` preprocess |
| Level-set | $t=\texttt{mult}\cdot 2\ln(255\,o)$ | `auxiliary.h:312-314` |
| Nửa cạnh | $\text{half}_x=\sqrt{t\,\Sigma'_{11}}$, $\text{half}_y=\sqrt{t\,\Sigma'_{22}}$ | (tương đương toán học) |
| Tiếp điểm | $x_{\text{term}}=-\operatorname{sgn}^*(B)\sqrt{\dfrac{-B^2t}{\text{disc}\cdot A}}$, $y_{\text{term}}$ thay $A\to C$ | `auxiliary.h:316-319, 341, 343` |
| Giao đường thẳng | $v_\pm=\dfrac{-Bh\pm\sqrt{\text{disc}\cdot h^2+tk}}{k}+p_v$ | `auxiliary.h:159-174` |
| | ($k=C$ khi cắt theo $x$; $k=A$ khi cắt theo $y$) | |
| Biệt thức | $\text{disc}=B^2-AC$, cần $<0$ | `auxiliary.h:305` |

Hằng số: $\ln 255=5.541264$, $\ln 25.5=3.238679$, $\ln 5.1=1.629241$, $\pi/4=0.785398$.

---
---

# ĐÁP ÁN

## Câu 1 — Hình học của splat

**a)**

$$\det\Sigma'=117\cdot36-54^2=4212-2916=\mathbf{1296}$$
$$\text{mid}=\tfrac12(117+36)=\mathbf{76.5}$$
$$\sqrt{\text{mid}^2-\det}=\sqrt{5852.25-1296}=\sqrt{4556.25}=67.5$$
$$\lambda_{\max}=76.5+67.5=\mathbf{144},\qquad \lambda_{\min}=76.5-67.5=\mathbf{9}$$

**b)** $\sigma_{\max}=\sqrt{144}=\mathbf{12}$ px, $\sigma_{\min}=\sqrt{9}=\mathbf{3}$ px, $\rho=12/3=\mathbf{4}$.

Splat này dẹt tỉ lệ **4:1** — nằm đúng vùng "bình thường sau khi train" mà §4.1 mô tả.

**c)**

$$\frac{\text{hộp vuông}}{\text{ellipse }3\sigma}=\frac{36\,\sigma_{\max}^2}{9\pi\,\sigma_{\max}\sigma_{\min}}=\frac{4}{\pi}\rho=\frac{4}{\pi}\cdot 4=\mathbf{5.093\times}$$

Phép thử tỉnh táo: đặt $\rho=1$ → $4/\pi\approx1.273$, đúng bằng tỉ lệ hình vuông trên hình tròn nội tiếp. ✅

---

## Câu 2 — Conic

$$M=\Sigma'^{-1}=\frac{1}{1296}\begin{pmatrix}36 & -54\\ -54 & 117\end{pmatrix}
=\begin{pmatrix}1/36 & -1/24\\ -1/24 & 13/144\end{pmatrix}$$

$$A=\tfrac{1}{36}=0.0277778,\qquad B=-\tfrac{1}{24}=-0.0416667,\qquad C=\tfrac{13}{144}=0.0902778$$

**Biệt thức:**

$$\text{disc}=B^2-AC=\frac{1}{576}-\frac{1}{36}\cdot\frac{13}{144}=\frac{9}{5184}-\frac{13}{5184}=-\frac{4}{5184}=-\frac{1}{1296}$$

$$\text{disc}=-7.716\times10^{-4}<0\quad ✅$$

**Chứng minh $\text{disc}=-1/\det\Sigma'$:** với $M$ đối xứng $2\times2$,

$$B^2-AC=-(AC-B^2)=-\det M=-\det(\Sigma'^{-1})=-\frac{1}{\det\Sigma'}$$

Xác nhận: $-1/1296$ ✓. Đây là cách kiểm tra $(A,B,C)$ rẻ nhất — sai một dấu là lộ ngay.

---

## Câu 3 — Ngưỡng level-set

**a)** $t=0.5\cdot 2\ln(255\cdot 1.0)=\ln 255=\mathbf{5.5413}$

**b)**

| `mult` | $o$ | $255\,o$ | $t=\texttt{mult}\cdot2\ln(255o)$ | Bán kính hiệu dụng $\sqrt{t}\,\sigma'$ |
|---|---|---|---|---|
| 1.0 | 1.00 | 255 | **11.0825** | $3.33\,\sigma'$ |
| 0.5 | 1.00 | 255 | **5.5413** | $2.35\,\sigma'$ |
| 0.5 | 0.10 | 25.5 | **3.2387** | $1.80\,\sigma'$ |
| 0.5 | 0.02 | 5.1 | **1.6292** | $1.28\,\sigma'$ |

**c)** Vì `mult` nhân vào $t$, mà $t$ đứng **dưới dấu căn** ở nửa cạnh:

$$\text{half}_x=\sqrt{t\,\Sigma'_{11}}\ \Rightarrow\ \text{cạnh}\propto\sqrt{\texttt{mult}},\qquad \text{diện tích}=4\,\text{half}_x\text{half}_y\propto \texttt{mult}$$

Cạnh co theo $\sqrt{0.5}=0.707$, diện tích co theo $0.5$. Cách hiểu sai ("nhân 0.5 thẳng vào cạnh") sẽ cho diện tích $0.25$ — sai lệch gấp đôi.

---

## Câu 4 — Bounding box

**a)**

$$\text{half}_x=\sqrt{5.5413\times117}=\sqrt{648.33}=\mathbf{25.462}\ \text{px}$$
$$\text{half}_y=\sqrt{5.5413\times36}=\sqrt{199.49}=\mathbf{14.124}\ \text{px}$$

Chú ý ngay: hộp **dị hướng** $25.46\times14.12$ — hoàn toàn khác hộp vuông của 3DGS. Đây là chỗ hệ số $\rho$ của §4.1 biến mất.

**b)**

$$x\in[120-25.462,\ 120+25.462]=[94.538,\ 145.462]$$
$$y\in[88-14.124,\ 88+14.124]=[73.876,\ 102.124]$$

**c)**

| Trục | Khoảng | Tile đầu | Tile cuối | Số tile |
|---|---|---|---|---|
| $x$ | $[94.538,145.462]$ | $\lfloor94.538/16\rfloor=5$ | $\lfloor145.462/16\rfloor=9$ | **5 cột** (5–9) |
| $y$ | $[73.876,102.124]$ | $\lfloor73.876/16\rfloor=4$ | $\lfloor102.124/16\rfloor=6$ | **3 hàng** (4–6) |

$$K_{\text{box}}=5\times3=\mathbf{15}\ \text{tile}$$

---

## Câu 5 — Điểm tiếp tuyến

**a)** $B=-0.0416667<0\ \Rightarrow\ \operatorname{sgn}^*(B)=-1\ \Rightarrow\ -\operatorname{sgn}^*(B)=+1$.

$$\frac{-B^2t}{\text{disc}\cdot A}=\frac{-(1/576)(5.5413)}{(-1/1296)(1/36)}=(1/576)(5.5413)(46656)=5.5413\times81=448.85$$

$$x_{\text{term}}=+\sqrt{448.85}=\mathbf{21.186}$$

$$\frac{-B^2t}{\text{disc}\cdot C}=\frac{-(1/576)(5.5413)}{(-1/1296)(13/144)}=5.5413\times\frac{186624}{7488}=5.5413\times24.923=138.11$$

$$y_{\text{term}}=+\sqrt{138.11}=\mathbf{11.752}$$

**b)** $x_{\text{term}}$ là **độ lệch $x$ của điểm có $y$ cực đại** trên ellipse (không phải nửa cạnh!).

Chứng minh: trên $A\,dx^2+2B\,dx\,dy+C\,dy^2=t$, lấy vi phân toàn phần và đặt $d(dy)=0$ để tìm cực trị của $dy$… thực ra dễ hơn: cực đại $dy$ đạt khi $\partial/\partial(dx)=0$:

$$2A\,dx+2B\,dy=0\ \Longrightarrow\ dx=-\frac{B}{A}dy$$

Thay $dy=\text{half}_y=14.124$:

$$dx=-\frac{-0.0416667}{0.0277778}\times14.124=1.5\times14.124=\mathbf{21.186}\quad ✅\ =x_{\text{term}}$$

Đối xứng, cực đại $dx$ đạt khi $2B\,dx+2C\,dy=0\Rightarrow dy=-\tfrac{B}{C}dx=\tfrac{6}{13}\times25.462=\mathbf{11.752}=y_{\text{term}}$ ✅

**Kết luận:** hai tiếp điểm là $(p_x+x_{\text{term}},\,p_y+\text{half}_y)$ và $(p_x+\text{half}_x,\,p_y+y_{\text{term}})$. Kernel dùng chúng làm đầu vào cho `computeEllipseIntersection` để lấy biên chính xác — nó **không bao giờ** nghịch đảo $M$ ngược về $\Sigma'$.

---

## Câu 6 — Giao với đường thẳng đứng

**a)** Cắt theo $x$ ⟹ $k=C$, $h=128-120=8$.

$$\text{disc}\cdot h^2+tC=(-7.716\times10^{-4})(64)+5.5413(0.0902778)=-0.04938+0.50026=0.45087$$
$$\sqrt{\cdot}=0.67147$$
$$-Bh=+0.0416667\times8=0.33333$$

$$dy_+=\frac{0.33333+0.67147}{0.0902778}=\mathbf{11.130},\qquad dy_-=\frac{0.33333-0.67147}{0.0902778}=\mathbf{-3.746}$$

$$y\in[88-3.746,\ 88+11.130]=[\mathbf{84.254},\ \mathbf{99.130}]$$

Hàng tile: $\lfloor84.254/16\rfloor=5$ đến $\lfloor99.130/16\rfloor=6$ ⟹ **hàng 5 và 6** (2 tile).

Lưu ý khoảng này **lệch tâm** ($-3.7$ so với $+11.1$): đó là hệ quả của $B\ne0$ — ellipse nghiêng, nên đường $x=128$ cắt nó không đối xứng quanh $p_y$.

**b)** Dưới căn $=0.45087>0$ ⟹ đường thẳng còn cắt ellipse.

Với $x=150$: $h=30$, $\text{disc}\cdot h^2+tC=(-7.716\times10^{-4})(900)+0.50026=-0.6944+0.5003=\mathbf{-0.194}<0$.

Căn âm ⟹ **đường thẳng không cắt ellipse** ⟹ tile ở đó bị loại. Hợp lý: $x=150$ nằm ngoài $\text{bbox}_{x,\max}=145.46$. Chính $\text{disc}<0$ (câu 2) là thứ bảo đảm biểu thức dưới căn dương khi và chỉ khi $|h|$ đủ nhỏ.

---

## Câu 7 — Đếm tile thật

**a)** $x\_span=5$ cột, $y\_span=3$ hàng ⟹ `isY = (3 < 5) = true` ⟹ **duyệt theo hàng**, mỗi hàng giải ra khoảng $x$.

Cắt theo $y$ ⟹ $k=A=0.0277778$, $tA=5.5413/36=0.153925$.

$$dx_\pm(h)=\frac{-Bh\pm\sqrt{\text{disc}\cdot h^2+tA}}{A},\qquad h=y-88$$

Hai tiếp điểm cực trị theo $x$ nằm ở $dy=\pm y_{\text{term}}=\pm11.752$, tức $y=76.248$ và $y=99.752$.

**b) Hàng 4** ($y\in[64,80]$, phần hữu hiệu $[73.876,\,80]$):

Tiếp điểm $y=76.248$ **nằm trong** hàng ⟹ lấy thẳng $dx_{\min}=-\text{half}_x=-25.462$ ⟹ $x_{\min}=94.538$.

$dx_{\max}$ lấy tại hai biên lát:
- $y=80$ ($h=-8$): $\sqrt{-0.04938+0.153925}=\sqrt{0.104542}=0.32333$; $-Bh=-0.33333$
  $$dx_+=\frac{-0.33333+0.32333}{0.0277778}=-0.360$$
- $y=73.876$ (tiếp tuyến đáy): $dx=-Bh/A=-21.186$

$dx_{\max}=\max(-0.360,\,-21.186)=-0.360$ ⟹ $x_{\max}=119.640$

$$x\in[94.538,\,119.640]\ \Rightarrow\ \text{cột }\lfloor5.909\rfloor=5\ \text{đến}\ \lfloor7.478\rfloor=7\ \Rightarrow\ \mathbf{3\ tile}$$

**Hàng 5** ($y\in[80,96]$, $h\in[-8,+8]$): không chứa tiếp điểm nào ⟹ lấy cả hai biên.
- $h=-8$: $dx\in[-23.640,\,-0.360]$
- $h=+8$: $-Bh=+0.33333$ ⟹ $dx_+=\frac{0.33333+0.32333}{0.0277778}=23.640$, $dx_-=0.360$

$dx\in[-23.640,\,23.640]$ ⟹ $x\in[96.360,\,143.640]$

$$\text{cột }\lfloor6.023\rfloor=6\ \text{đến}\ \lfloor8.978\rfloor=8\ \Rightarrow\ \mathbf{3\ tile}$$

**Hàng 6** ($y\in[96,112]$, phần hữu hiệu $[96,\,102.124]$):

Tiếp điểm $y=99.752$ **nằm trong** hàng ⟹ $dx_{\max}=+25.462$ ⟹ $x_{\max}=145.462$.

$dx_{\min}$: tại $y=96$ ($h=8$) là $0.360$; tại $y=102.124$ (tiếp tuyến đỉnh) là $+21.186$. ⟹ $dx_{\min}=0.360$ ⟹ $x_{\min}=120.360$

$$x\in[120.360,\,145.462]\ \Rightarrow\ \text{cột }\lfloor7.523\rfloor=7\ \text{đến}\ \lfloor9.091\rfloor=9\ \Rightarrow\ \mathbf{3\ tile}$$

**c)**

$$K_{\text{fastgs}}=3+3+3=\mathbf{9}\ \text{tile}$$

So với $K_{\text{box}}=15$: bước lọc ellipse loại **6 tile (40%)**.

Bản đồ tile (`██` giữ, `··` loại):

```
        cột 5   6   7   8   9
hàng 4:  ██  ██  ██  ··  ··
hàng 5:  ··  ██  ██  ██  ··
hàng 6:  ··  ··  ██  ██  ██
```

Vệt chéo đi lên đúng như kỳ vọng: $\Sigma'_{12}=54>0$ ⟹ tương quan dương ⟹ ellipse nghiêng theo hướng $x$ tăng thì $y$ tăng. Đây là kiểm tra hình học miễn phí cho toàn bộ phép tính.

---

## Câu 8 — Đường so sánh 3DGS

**a)** $r=\lceil 3\sqrt{144}\rceil=\lceil 36\rceil=\mathbf{36}$ px (hộp vuông cạnh 72).

**b)** Trục $x$: $\text{rect}_{\min}=\lfloor(120-36)/16\rfloor=\lfloor5.25\rfloor=5$; $\text{rect}_{\max}=\lfloor(120+36+15)/16\rfloor=\lfloor10.6875\rfloor=10$ ⟹ cột 5–9 = **5 cột**.

Trục $y$: $\text{rect}_{\min}=\lfloor(88-36)/16\rfloor=\lfloor3.25\rfloor=3$; $\text{rect}_{\max}=\lfloor(88+36+15)/16\rfloor=\lfloor8.6875\rfloor=8$ ⟹ hàng 3–7 = **5 hàng**.

$$K_{\text{3dgs}}=5\times5=\mathbf{25}\ \text{tile}$$

**c)**

$$R_{\text{tile}}=\frac{9}{25}=\mathbf{0.360}$$

Compact box tiết kiệm **64%** số tile cho splat này.

Chuỗi ba con số đáng nhớ: $25 \to 15 \to 9$.
- $25\to15$: **hộp theo trục** thay hộp vuông (§4.2) — đóng góp lớn nhất, $-40\%$
- $15\to9$: **lọc ellipse** trong `processTiles` (§4.4) — $-40\%$ nữa

---

## Câu 9 — Đối chiếu công thức tiệm cận

**a)**

$$\sqrt{\Sigma'_{11}\Sigma'_{22}}=\sqrt{117\times36}=\sqrt{4212}=64.900$$
$$4t\sqrt{\Sigma'_{11}\Sigma'_{22}}=4\times5.5413\times64.900=1438.6\quad[\text{diện tích hộp, px}^2]$$
$$36\,\lambda_{\max}=36\times144=5184\quad[\text{diện tích hộp vuông, px}^2]$$

$$R_{\text{tile}}\approx\frac{\pi}{4}\times\frac{1438.6}{5184}=0.785398\times0.27753=\mathbf{0.2180}$$

**b)** Đo được $0.360$, tiệm cận cho $0.218$ — **giá trị đo cao hơn 65%**.

Nguyên nhân: **lượng tử hoá lưới tile**, cụ thể là số hạng "$+1$". Một splat phủ $W$ px theo trục $x$ chạm khoảng $W/16+1$ tile, chứ không phải $W/16$. Hằng số $+1$ đó **không co lại** khi splat nhỏ đi, nên nó chiếm tỉ trọng lớn hơn ở hộp nhỏ:

| | Bề rộng | $W/16$ | $+1$ | Tile thật |
|---|---|---|---|---|
| 3DGS | 72 px | 4.5 | | 5 |
| fastgs | 50.9 px | 3.18 | | 5 |
| fastgs (trục $y$) | 28.2 px | 1.77 | | 3 |

Theo trục $x$, hộp fastgs hẹp hơn 29% nhưng **vẫn chạm đúng 5 cột như 3DGS** — toàn bộ lợi ích trục $x$ bị lượng tử hoá nuốt mất. Đó chính là lý do §4.3 cảnh báo "$K$ giảm chậm hơn nhiều so với bán kính".

Công thức tiệm cận chỉ đúng trong **giới hạn splat lớn** (nhiều tile), khi $+1$ trở nên không đáng kể. Splat trong bài ($5\times3$ tile) còn rất xa giới hạn đó.

**c)** Đo được: loại $6/15=40\%$. Tiệm cận: $1-\pi/4=21.5\%$.

Cùng một nguyên nhân, nhưng chiều tác động ngược: tỉ số $\pi/4$ giả định ellipse nội tiếp hộp **liên tục**. Ở quy mô $5\times3$ tile, mỗi tile là một khối 16px thô — bốn góc hộp bị cắt trọn vẹn thành từng tile nguyên, nên tỉ lệ loại nhảy vọt. Với splat 20×20 tile, con số sẽ hội tụ về gần $21.5\%$.

> 📌 Điều cần rút ra từ câu 9: **cả hai con số $0.218$ và $0.360$ đều không sai** — chúng đo hai thứ khác nhau. $0.218$ là tỉ lệ **diện tích**, $0.360$ là tỉ lệ **số tile**. Kernel trả tiền theo tile, nên $0.360$ mới là con số đi vào mô hình chi phí ở Phần II.

---

## Câu 10 — Quét tham số

**a)**

| Cấu hình | $t$ | $\text{half}_x$ | $\text{half}_y$ | Hộp (tile) | $K_{\text{fastgs}}$ | $R_{\text{tile}}$ |
|---|---|---|---|---|---|---|
| gốc: $\texttt{mult}=0.5,\ o=1$ | 5.5413 | 25.462 | 14.124 | $5\times3=15$ | **9** | **0.360** |
| (i): $\texttt{mult}=1.0,\ o=1$ | 11.0825 | 36.009 | 19.974 | $5\times3=15$ | **11** | **0.440** |
| (ii): $\texttt{mult}=0.5,\ o=0.1$ | 3.2387 | 19.466 | 10.798 | $3\times3=9$ | **7** | **0.280** |

<details>
<summary>Chi tiết đếm tile cho (i) — <code>mult</code> = 1.0</summary>

bbox: $x\in[83.99,156.01]$ → cột 5–9; $y\in[68.03,107.97]$ → hàng 4–6. Hộp $=15$ tile.
$y\_span=3<x\_span=5$ ⟹ duyệt theo hàng. $tA=11.0825/36=0.307847$. Tiếp điểm tại $dy=\pm\tfrac{6}{13}(36.009)=\pm16.619$, tức $y=71.38$ và $y=104.62$.

- **Hàng 4** ($[68.03,80]$): chứa $y=71.38$ ⟹ $x_{\min}=83.99$. Tại $h=-8$: $\sqrt{-0.04938+0.307847}=0.50839$, $dx_+=(-0.33333+0.50839)/0.0277778=6.302$ ⟹ $x_{\max}=126.30$. Cột 5–7 = **3**
- **Hàng 5** ($h\in[-8,8]$): $dx\in[-30.302,30.302]$ ⟹ $x\in[89.70,150.30]$. Cột 5–9 = **5**
- **Hàng 6** ($[96,107.97]$): chứa $y=104.62$ ⟹ $x_{\max}=156.01$; $dx_{\min}=-6.302$ ⟹ $x_{\min}=113.70$. Cột 7–9 = **3**

$K=3+5+3=11$
</details>

<details>
<summary>Chi tiết đếm tile cho (ii) — <code>o</code> = 0.1</summary>

bbox: $x\in[100.53,139.47]$ → cột 6–8; $y\in[77.20,98.80]$ → hàng 4–6. Hộp $=9$ tile.
$y\_span=3$, $x\_span=3$ ⟹ `isY = (3<3) = false` ⟹ **duyệt theo cột**. $tC=3.2387\times0.0902778=0.292395$. Tiếp điểm theo $y$ tại $dx=\pm1.5(10.798)=\pm16.197$, tức $x=103.80$ và $x=136.20$.

- **Cột 6** ($[100.53,112]$): chứa $x=103.80$ ⟹ $y_{\min}=77.20$; tại $h=-8$: $\sqrt{-0.04938+0.292395}=0.49296$, $dy_+=(-0.33333+0.49296)/0.0902778=1.768$ ⟹ $y_{\max}=89.77$. Hàng 4–5 = **2**
- **Cột 7** ($h\in[-8,8]$): $dy\in[-9.153,9.153]$ ⟹ $y\in[78.85,97.15]$. Hàng 4–6 = **3**
- **Cột 8** ($[128,139.47]$): chứa $x=136.20$ ⟹ $y_{\max}=98.80$; $dy_{\min}=-1.768$ ⟹ $y_{\min}=86.23$. Hàng 5–6 = **2**

$K=2+3+2=7$
</details>

**b)** Diện tích hộp: $\texttt{mult}\,1.0$ cho $4(36.009)(19.974)=2877$ px², gấp **đúng 2.00 lần** $1439$ px² của $\texttt{mult}\,0.5$ ✓ (khớp câu 3c).

Nhưng số tile chỉ tăng $9\to11$, tức **$+22\%$**, không phải $+100\%$.

Lý do vẫn là "$+1$": ở cả hai cấu hình, hộp chạm đúng 5 cột × 3 hàng (15 tile bao). Diện tích gấp đôi chỉ làm ellipse "béo" hơn bên trong cùng khung 15 tile đó, nên chỉ giành thêm được 2 tile ở phần lồi ra của hàng 5.

**Hệ quả thực tiễn:** giảm `mult` từ 1.0 xuống 0.5 **không** giảm một nửa chi phí rasterization. Lợi ích thật nhỏ hơn nhiều so với những gì công thức diện tích gợi ý — và đó là lý do §4.5 phải nhấn mạnh bảng đo bằng `demos/fastgs_cost_model.py` thay vì tin công thức liên tục.

**c)** $t$: từ $5.5413$ xuống $3.2387$ ⟹ giảm **41.6%**.
$K$: từ $9$ xuống $7$ ⟹ giảm **22.2%**.

Khớp chính xác nhận định của §4.3: lợi ích từ opacity thấp **là thật nhưng khiêm tốn**, và $K$ giảm chậm hơn nhiều so với $t$ — cùng cơ chế lượng tử hoá tile. (§4.3 đo được $-28\%$ số tile khi $t$ giảm $42\%$; ở đây $-22\%$, cùng bậc.)

Ý nghĩa: những Gaussian mờ — đông đảo nhất ở giai đoạn giữa huấn luyện, ngay trước khi bị prune — được rasterize **rẻ hơn** trong fastgs, trong khi 3DGS trả giá đầy đủ cho chúng vì hộp $3\sigma$ của nó không hề nhìn đến $o$.

**d)** **Không.** $\texttt{mult}=1.0$ đưa hộp về **SnugBox nguyên bản của Speedy-Splat**, không phải 3DGS.

Bằng chứng ngay trong bài: ở $\texttt{mult}=1.0$, $\text{half}_x=36.009$ — **rộng hơn** $3\sigma_{\max}=36$ của 3DGS. SnugBox bao đến mức đóng góp $1/255$, chứ không cắt tại $3\sigma$.

Nhưng $R_{\text{tile}}$ vẫn là $0.440<1$. Cái sinh ra tiết kiệm **không phải** bán kính, mà là:

1. **Hộp theo từng trục** (§4.2): $36.0\times20.0$ thay vì $36\times36$ — chính hệ số $\rho=4$ của câu 1
2. **Phụ thuộc opacity** (§4.3b)
3. **Lọc ellipse** trong `processTiles` (§4.4)

Trong codebase này **không có đường quay lại** cách dựng hộp của 3DGS — `getRect` đã bị xoá khỏi `auxiliary.h`, `my_radius` chỉ còn được ghi vào `radii[]` cho ngưỡng prune 20px.

---

## Bảng tổng kết bài tập

| Đại lượng | Ký hiệu | Kết quả |
|---|---|---|
| Eigenvalue | $\lambda_{\max},\lambda_{\min}$ | $144,\ 9$ |
| Tỉ lệ trục | $\rho$ | $4$ |
| Lãng phí hộp vuông (§4.1, liên tục) | $\tfrac4\pi\rho$ | $5.09\times$ |
| Conic | $(A,B,C)$ | $(\tfrac1{36},\,-\tfrac1{24},\,\tfrac{13}{144})$ |
| Biệt thức | disc | $-1/1296$ |
| Level-set | $t$ | $5.5413$ |
| Nửa cạnh | $\text{half}_x,\text{half}_y$ | $25.46,\ 14.12$ px |
| Tiếp điểm | $x_{\text{term}},y_{\text{term}}$ | $21.19,\ 11.75$ |
| Tile — 3DGS | $K_{\text{3dgs}}$ | $25$ |
| Tile — chỉ hộp | $K_{\text{box}}$ | $15$ |
| Tile — fastgs | $K_{\text{fastgs}}$ | $\mathbf{9}$ |
| **Tỉ số** | $R_{\text{tile}}$ | $\mathbf{0.360}$ |
| Tiệm cận (§4.5) | $R_{\text{tile}}^{\text{lim}}$ | $0.218$ |

Ba bài học chính:

1. **Nguồn tiết kiệm lớn nhất là dị hướng**, không phải việc co hộp. Hộp theo trục một mình đã cho $25\to15$.
2. **Lượng tử hoá tile ăn bớt lợi ích rất nhiều.** Mọi công thức diện tích liên tục đều lạc quan hơn thực tế; splat càng nhỏ, sai lệch càng lớn.
3. **Chỉ có số tile mới đi vào mô hình chi phí** ở Phần II. Tỉ lệ diện tích là công cụ giải thích, không phải đại lượng thanh toán.
