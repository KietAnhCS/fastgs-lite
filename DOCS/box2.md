# Phần IV-b — Ba công thức đếm tile: 3DGS, compact box, ellipse

> Rút gọn từ [`box.md`](box.md) sang dạng **tổng quát** — áp dụng cho mọi splat, không chỉ số liệu của đề.
> Mỗi công thức đều được kiểm chứng lại bằng số liệu bài tập ($\Sigma'=\begin{pmatrix}117&54\\54&36\end{pmatrix}$, $p=(120,88)$, $\texttt{mult}=0.5$, $o=1$).

---

## Ký hiệu dùng chung

$$\Sigma'=\begin{pmatrix}\Sigma_{11}&\Sigma_{12}\\ \Sigma_{12}&\Sigma_{22}\end{pmatrix},\qquad \det=\Sigma_{11}\Sigma_{22}-\Sigma_{12}^2,\qquad \text{mid}=\tfrac12(\Sigma_{11}+\Sigma_{22})$$

$$\lambda_{\max,\min}=\text{mid}\pm\sqrt{\text{mid}^2-\det}$$

$$(A,B,C)=\frac{1}{\det}\bigl(\Sigma_{22},\ -\Sigma_{12},\ \Sigma_{11}\bigr),\qquad \text{disc}=B^2-AC=-\frac{1}{\det}$$

$$t=\texttt{mult}\cdot 2\ln(255\,o),\qquad B_U=B_V=16$$

Với một khoảng pixel $[u_{\min},u_{\max}]$, số tile chạm **luôn** là

$$N[u_{\min},u_{\max}]=\Big\lfloor \tfrac{u_{\max}}{16}\Big\rfloor-\Big\lfloor \tfrac{u_{\min}}{16}\Big\rfloor+1$$

---

## 1. 3DGS gốc — hộp vuông $3\sigma$

$$r=\bigl\lceil 3\sqrt{\lambda_{\max}}\,\bigr\rceil$$

$$K_{\text{3dgs}}=\prod_{u\in\{x,y\}}\left(\Big\lfloor\frac{p_u+r+15}{16}\Big\rfloor-\Big\lfloor\frac{p_u-r}{16}\Big\rfloor\right)$$

Đây chính là `getRect` (biên trên **loại trừ**) — tương đương $N[p_u-r,\ p_u+r]$.

Đặc điểm: **cùng một $r$ cho cả hai trục**, và **không nhìn đến $o$**.

*Kiểm chứng đề:* $r=36$ → $(10-5)\times(8-3)=5\times5=\mathbf{25}$.

---

## 2. Compact box (SnugBox / axis-aligned, chưa lọc ellipse)

$$\text{half}_x=\sqrt{t\,\Sigma_{11}},\qquad \text{half}_y=\sqrt{t\,\Sigma_{22}}$$

$$K_{\text{box}}=N\bigl[p_x-\text{half}_x,\ p_x+\text{half}_x\bigr]\cdot N\bigl[p_y-\text{half}_y,\ p_y+\text{half}_y\bigr]$$

Khác biệt cấu trúc duy nhất so với (1): mỗi trục có nửa cạnh **riêng** — dị hướng, chính là chỗ hệ số $\rho=\sigma_{\max}/\sigma_{\min}$ biến mất — và $t$ phụ thuộc $o$ lẫn `mult`.

*Kiểm chứng đề:* $25.46\times14.12$ px → $5\times3=\mathbf{15}$.

---

## 3. Ellipse chính xác (`processTiles`)

Đây là công thức **tổng theo lát**, không phải tích hai trục.

### Bước 0 — chọn trục quét

$$x\_\text{span}=N_x,\qquad y\_\text{span}=N_y,\qquad \texttt{isY}=\bigl(y\_\text{span}<x\_\text{span}\bigr)$$

- `isY = true` → quét theo **hàng**, mỗi hàng giải ra khoảng $x$ (dùng $k=A$)
- `isY = false` → quét theo **cột**, mỗi cột giải ra khoảng $y$ (dùng $k=C$)

### Bước 1 — nghiệm giao của một đường thẳng $h=$ const

$$d_\pm(h)=\frac{-Bh\pm\sqrt{\text{disc}\cdot h^2+t\,k}}{k},\qquad
k=\begin{cases}A, & h=y-p_y\\[2pt] C, & h=x-p_x\end{cases}$$

Nhánh trên là **quét theo hàng** ($k=A$), nhánh dưới là **quét theo cột** ($k=C$).

Tồn tại nghiệm $\iff \text{disc}\cdot h^2+tk\ge 0 \iff \lvert h\rvert\le\sqrt{\dfrac{tk}{-\text{disc}}}$ — vế phải đúng bằng nửa cạnh của trục kia.

### Bước 2 — tiếp điểm cực trị (biết trước, để khỏi quét mù)

$$x_{\text{term}}=-\frac{B}{A}\,\text{half}_y,\qquad y_{\text{term}}=-\frac{B}{C}\,\text{half}_x$$

Cực trị theo $x$ nằm tại $y=p_y\pm y_{\text{term}}$; cực trị theo $y$ nằm tại $x=p_x\pm x_{\text{term}}$.

### Bước 3 — với mỗi lát $j$

Giả sử quét theo hàng; lát $j$ phủ $[y_j^{lo},y_j^{hi}]$ **đã clip vào bbox**; đặt $h^{lo,hi}=y^{lo,hi}_j-p_y$:

$$d_{\min}^{(j)}=\begin{cases}
-\,\text{half}_x, & (\text{A})\\[4pt]
\min\bigl(d_-(h^{lo}),\ d_-(h^{hi})\bigr), & (\text{B})
\end{cases}
\qquad
d_{\max}^{(j)}=\begin{cases}
+\,\text{half}_x, & (\text{C})\\[4pt]
\max\bigl(d_+(h^{lo}),\ d_+(h^{hi})\bigr), & (\text{D})
\end{cases}$$

Điều kiện chọn nhánh:

| Nhánh | Áp dụng khi | Nghĩa |
|---|---|---|
| **(A)** | $p_y-y_{\text{term}}\in[y_j^{lo},\,y_j^{hi}]$ | lát $j$ **chứa tiếp điểm trái** ⟹ cực trị nằm trong lát, lấy thẳng biên bbox |
| **(B)** | ngược lại | cực trị nằm ở một trong hai đầu lát |
| **(C)** | $p_y+y_{\text{term}}\in[y_j^{lo},\,y_j^{hi}]$ | lát $j$ **chứa tiếp điểm phải** |
| **(D)** | ngược lại | |

> **Lý do phân nhánh:** $d_\pm(h)$ **đơn điệu** trên mỗi lát *trừ khi* lát chứa điểm tiếp tuyến — khi đó cực trị nằm bên trong lát, nên lấy thẳng biên bbox.

### Bước 4 — cộng dồn

$$\boxed{\;K_{\text{fastgs}}=\sum_{j=j_{\min}}^{j_{\max}} N\bigl[p_x+d_{\min}^{(j)},\ p_x+d_{\max}^{(j)}\bigr]\;}$$

Quét theo cột thì hoán vị $x\leftrightarrow y$, $A\leftrightarrow C$, $x_{\text{term}}\leftrightarrow y_{\text{term}}$.

*Kiểm chứng đề:* $3+3+3=\mathbf{9}$.

---

## Bảng đối chiếu

| | Hình bao | Công thức $K$ | Phụ thuộc $o$? | Dị hướng? | Đề bài |
|---|---|---|---|---|---|
| 3DGS | vuông $2r\times2r$ | tích 2 trục | ✗ | ✗ | 25 |
| Compact box | chữ nhật $2\text{half}_x\times2\text{half}_y$ | tích 2 trục | ✓ | ✓ | 15 |
| Ellipse | level-set nghiêng | **tổng theo lát** | ✓ | ✓ + nghiêng | 9 |

---

## Xấp xỉ liên tục (ước lượng nhanh, không đếm lát)

Cho một hình **lồi** diện tích $S$, chu vi $P$, trên lưới bước $B$, kỳ vọng số tile chạm là

$$\mathbb{E}[K]\approx \frac{S}{B^2}+\frac{P}{2B}+1$$

Áp vào từng trường hợp:

| | $S$ | $P$ |
|---|---|---|
| 3DGS | $36\lambda_{\max}$ | $24\sqrt{\lambda_{\max}}$ |
| Compact box | $4t\sqrt{\Sigma_{11}\Sigma_{22}}$ | $4\bigl(\sqrt{t\Sigma_{11}}+\sqrt{t\Sigma_{22}}\bigr)$ |
| Ellipse | $\pi t\sqrt{\det}$ | Ramanujan với $a=\sqrt{t\lambda_{\max}},\ b=\sqrt{t\lambda_{\min}}$ |

Số hạng $+1$ và $P/2B$ chính là **"thuế lượng tử hoá"** mà câu 9 của [`box.md`](box.md) nói tới: chỉ khi $S/B^2$ áp đảo thì $R_{\text{tile}}$ đo được mới hội tụ về tỉ số diện tích $0.218$.

---

## Lưu ý hiệu chỉnh cho §4.5

Công thức tiệm cận trong bài dùng hệ số $\pi/4$ (ellipse nội tiếp hộp). Chính xác hơn, tỉ lệ **ellipse / bbox** của một ellipse **nghiêng** là

$$\frac{\pi t\sqrt{\det}}{4t\sqrt{\Sigma_{11}\Sigma_{22}}}=\frac{\pi}{4}\sqrt{1-\rho_{xy}^2},\qquad \rho_{xy}=\frac{\Sigma_{12}}{\sqrt{\Sigma_{11}\Sigma_{22}}}$$

Với số liệu đề: $\rho_{xy}=54/64.900=0.83205$ → hệ số $=\mathbf{0.43566}$, không phải $0.785$.

Dạng gọn nhất, không cần đến $\rho_{xy}$:

$$R_{\text{tile}}^{\text{lim}}=\frac{\pi\,t\sqrt{\det}}{36\,\lambda_{\max}}=\frac{626.7}{5184}=\mathbf{0.1209}$$

so với $0.218$ mà hệ số $\pi/4$ cho ra — **sai lệch đúng bằng** $1/\sqrt{1-\rho_{xy}^2}=1.803$ lần.

Hệ quả cho câu 9 của [`box.md`](box.md):

| | Tiệm cận (đã sửa) | Đo được | Hướng lệch |
|---|---|---|---|
| $R_{\text{tile}}$ | $0.1209$ | $0.360$ | đo **cao gấp 2.98×** |
| Lọc ellipse loại | $56.4\%$ | $40\%$ | đo loại **ít hơn** |

Hai dòng giờ **cùng dấu** và cùng một nguyên nhân: `processTiles` lấy bao lồi theo hai biên lát rồi làm tròn ra biên tile — cả hai bước chỉ có thể **giữ thêm** tile. Với hệ số $\pi/4$ cũ ($21.5\%$) thì dòng thứ hai lệch ngược dấu, buộc phải bịa ra một cơ chế thứ hai để giải thích.

**Nguồn lỗi:** [`fastgs-acceleration-method.md:1027`](fastgs-acceleration-method.md) viết diện tích ellipse là $\pi\,\text{half}_x\text{half}_y$ — chỉ đúng khi $\Sigma'_{12}=0$. Các dòng 1056 và 1501 kế thừa cùng hệ số.

---
---

# Phụ lục A — Chú thích ký hiệu

Cột **Ví dụ** dùng số liệu bài tập của [`box.md`](box.md): $\Sigma'=\begin{pmatrix}117&54\\54&36\end{pmatrix}$, $p=(120,88)$, $\texttt{mult}=0.5$, $o=1$, tile $16$ px.

## A.1 — Quy ước đọc chỉ số

Nắm 5 quy ước này thì mọi công thức phía trên đọc trôi:

| Dạng | Đọc là | Ý nghĩa | Ví dụ |
|---|---|---|---|
| $u,v$ | "trục quét / trục giải" | **biến trục tổng quát**, thay cho $x$ hoặc $y$ tuỳ hướng quét | quét theo hàng ⟹ $u=y$, $v=x$ |
| $\pm$ | "cộng trừ" | hai nghiệm của phương trình bậc 2 — $-$ là biên **dưới**, $+$ là biên **trên** | $d_-=-23.6$, $d_+=+23.6$ |
| $^{lo},^{hi}$ | "lô / hai" | *low / high* — hai **biên của một lát**, đã clip vào bbox | hàng 5: $y^{lo}=80$, $y^{hi}=96$ |
| $^{(j)}$ | "của lát j" | đại lượng **riêng cho lát thứ $j$**, đổi theo từng lát | $d_{\max}^{(4)}=-0.360$ |
| $_{\min},_{\max}$ | "min / max" | cực trị **trên toàn ellipse** (không kèm $^{(j)}$) hoặc **trong lát** (có kèm) | $\text{half}_x$ vs $d_{\min}^{(j)}$ |

> ⚠️ **Chỗ dễ nhầm nhất:** $d_-$ và $d_{\min}^{(j)}$ **không phải một thứ**.
> $d_-(h)$ là nghiệm dưới tại **một đường thẳng** $h$; $d_{\min}^{(j)}$ là giá trị nhỏ nhất trên **cả lát $j$** — lấy bằng cách so hai đầu lát, hoặc lấy thẳng $-\text{half}$ nếu lát chứa tiếp điểm.

---

## A.2 — Đầu vào (biết trước khi tính)

| Ký hiệu | Đọc là | Ý nghĩa | Đơn vị | Ví dụ | Code |
|---|---|---|---|---|---|
| $\Sigma'$ | "sigma phẩy" | covariance 2D **đã chiếu** xuống mặt phẳng ảnh | px² | $\begin{pmatrix}117&54\\54&36\end{pmatrix}$ | `forward.cu` preprocess, công thức (5) §1.2 |
| $\Sigma'_{11}$ | "sigma một một" | phương sai theo trục $x$ | px² | $117$ | `cov.x` |
| $\Sigma'_{12}$ | "sigma một hai" | hiệp phương sai $x$–$y$; $\ne0$ ⟹ ellipse **nghiêng** | px² | $54$ | `cov.y` |
| $\Sigma'_{22}$ | "sigma hai hai" | phương sai theo trục $y$ | px² | $36$ | `cov.z` |
| $p=(p_x,p_y)$ | "pê" | **tâm splat** trên ảnh (toạ độ pixel, không phải chỉ số tile) | px | $(120,\,88)$ | `points_xy_image` |
| $o$ | "ô" | **opacity** sau sigmoid, $\in(0,1)$ | — | $1.0$ | `con_o.w` |
| $\texttt{mult}$ | "mun" | hệ số co hộp do người dùng chỉnh | — | $0.5$ | `arguments/__init__.py:96` |
| $B_U,B_V$ | "bê u, bê v" | **cạnh tile** theo hai trục (luôn bằng nhau ở repo này) | px | $16,\ 16$ | `BLOCK_X`, `BLOCK_Y` |

---

## A.3 — Dẫn xuất từ $\Sigma'$ (hình học splat)

| Ký hiệu | Đọc là | Ý nghĩa | Đơn vị | Ví dụ | Ghi chú |
|---|---|---|---|---|---|
| $\det$ | "đê-tê" | $\Sigma'_{11}\Sigma'_{22}-\Sigma'^2_{12}$ — "thể tích" của $\Sigma'$ | px⁴ | $1296$ | $\det>0$ ⟺ ellipse hợp lệ |
| $\text{mid}$ | "mít" | $\tfrac12(\Sigma'_{11}+\Sigma'_{22})$ — nửa vết ma trận | px² | $76.5$ | trung bình 2 eigenvalue |
| $\lambda_{\max}$ | "lam-đa max" | eigenvalue **lớn** = phương sai dọc trục dài | px² | $144$ | `forward.cu:239-240` |
| $\lambda_{\min}$ | "lam-đa min" | eigenvalue **nhỏ** = phương sai dọc trục ngắn | px² | $9$ | $\lambda_{\max}\lambda_{\min}=\det$ |
| $\sigma_{\max},\sigma_{\min}$ | "xích-ma" | $\sqrt{\lambda}$ — **bán trục** ở mức $1\sigma$ | px | $12,\ 3$ | |
| $\rho$ | "rô" | $\sigma_{\max}/\sigma_{\min}$ — **độ dẹt** (tỉ lệ trục) | — | $4$ | biến ảnh hưởng mạnh nhất tới $R_{\text{tile}}$ |
| $\rho_{xy}$ | "rô ích-y" | $\Sigma'_{12}/\sqrt{\Sigma'_{11}\Sigma'_{22}}$ — **tương quan chuẩn hoá**, $\in(-1,1)$ | — | $0.832$ | **khác hẳn** $\rho$; đo độ *nghiêng*, không phải độ *dẹt* |

> $\rho$ và $\rho_{xy}$ là hai thứ khác nhau, chỉ trùng chữ cái. $\rho=4$ nói splat dài gấp 4 lần bề ngang; $\rho_{xy}=0.832$ nói trục dài của nó **nghiêng** so với lưới tile. Một splat có thể rất dẹt ($\rho$ lớn) mà không nghiêng chút nào ($\rho_{xy}=0$) — khi đó nó nằm đúng dọc trục $x$ hoặc $y$.

---

## A.4 — Conic và ngưỡng level-set

Kernel **không bao giờ** nghịch đảo ngược về $\Sigma'$ — nó làm việc trực tiếp với $M=\Sigma'^{-1}$.

| Ký hiệu | Đọc là | Ý nghĩa | Đơn vị | Ví dụ | Code |
|---|---|---|---|---|---|
| $M$ | "em" | $\Sigma'^{-1}$ — ma trận **conic** | px⁻² | $\begin{pmatrix}1/36&-1/24\\-1/24&13/144\end{pmatrix}$ | |
| $A$ | "a" | $M_{11}=\Sigma'_{22}/\det$ — hệ số của $dx^2$ | px⁻² | $1/36=0.027778$ | `con_o.x` |
| $B$ | "bê" | $M_{12}=-\Sigma'_{12}/\det$ — hệ số **chéo**; $B\ne0$ ⟹ nghiêng | px⁻² | $-1/24=-0.041667$ | `con_o.y` |
| $C$ | "xê" | $M_{22}=\Sigma'_{11}/\det$ — hệ số của $dy^2$ | px⁻² | $13/144=0.090278$ | `con_o.z` |
| $\text{disc}$ | "đít-cờ" | $B^2-AC=-1/\det$ — **biệt thức**, phải $<0$ | px⁻⁴ | $-1/1296=-7.716\times10^{-4}$ | `auxiliary.h:305` |
| $\Delta$ | "đen-ta" | vector **lệch so với tâm**: $\Delta=(x-p_x,\ y-p_y)$ | px | | |
| $t$ | "tê" | **ngưỡng level-set**: ellipse là tập $\Delta^\top M\Delta=t$ | — | $5.5413$ | `auxiliary.h:312-314` |

**Cách đọc $t$:** $t=\texttt{mult}\cdot2\ln(255\,o)$. Bán kính hiệu dụng là $\sqrt{t}$ lần độ lệch chuẩn — ở đây $\sqrt{5.5413}=2.35$, tức hộp bao đến mức $2.35\sigma$. Vì $t$ nằm **dưới dấu căn**, giảm `mult` một nửa chỉ co cạnh $\sqrt{0.5}=71\%$ nhưng co **diện tích** đúng $50\%$.

---

## A.5 — Hộp bao

| Ký hiệu | Đọc là | Ý nghĩa | Đơn vị | Ví dụ | Ghi chú |
|---|---|---|---|---|---|
| $r$ | "rờ" | bán kính 3DGS $=\lceil3\sqrt{\lambda_{\max}}\rceil$ | px | $36$ | **một số duy nhất** cho cả 2 trục |
| $\text{half}_x$ | "háp ích" | nửa cạnh hộp theo $x$ $=\sqrt{t\,\Sigma'_{11}}$ | px | $25.462$ | |
| $\text{half}_y$ | "háp y" | nửa cạnh hộp theo $y$ $=\sqrt{t\,\Sigma'_{22}}$ | px | $14.124$ | dị hướng — chỗ $\rho$ được khai thác |
| $x_{\text{term}}$ | "ích-tơm" | độ lệch $x$ của điểm có **$y$ cực đại** trên ellipse $=-\tfrac BA\text{half}_y$ | px | $21.186$ | **không phải** nửa cạnh! `auxiliary.h:316-319` |
| $y_{\text{term}}$ | "y-tơm" | độ lệch $y$ của điểm có **$x$ cực đại** $=-\tfrac BC\text{half}_x$ | px | $11.752$ | `auxiliary.h:341,343` |

**Bốn tiếp điểm** của ellipse với hộp bao — đây là toàn bộ lý do $x_{\text{term}},y_{\text{term}}$ tồn tại:

| Tiếp điểm | Toạ độ | Ví dụ |
|---|---|---|
| chạm cạnh **trên** | $(p_x+x_{\text{term}},\ p_y+\text{half}_y)$ | $(141.19,\ 102.12)$ |
| chạm cạnh **dưới** | $(p_x-x_{\text{term}},\ p_y-\text{half}_y)$ | $(98.81,\ 73.88)$ |
| chạm cạnh **phải** | $(p_x+\text{half}_x,\ p_y+y_{\text{term}})$ | $(145.46,\ 99.75)$ |
| chạm cạnh **trái** | $(p_x-\text{half}_x,\ p_y-y_{\text{term}})$ | $(94.54,\ 76.25)$ |

Nếu $B=0$ (không nghiêng) thì $x_{\text{term}}=y_{\text{term}}=0$ và bốn tiếp điểm rơi đúng vào giữa bốn cạnh — đúng như hình ellipse nội tiếp mà ai cũng vẽ trong đầu.

---

## A.6 — Quét tile (phần rắc rối nhất)

| Ký hiệu | Đọc là | Ý nghĩa | Đơn vị | Ví dụ |
|---|---|---|---|---|
| $N[\cdot,\cdot]$ | "en" | **hàm đếm tile** của một khoảng pixel | tile | $N[94.5,119.6]=3$ |
| $N_x,N_y$ | "en ích / en y" | số cột / số hàng mà **bbox** chạm | tile | $5,\ 3$ |
| $x\_\text{span}$ | "ích-xpan" | $=N_x$, bề rộng bbox tính bằng tile | tile | $5$ |
| $\texttt{isY}$ | "ít-y" | cờ chọn trục quét: $(y\_\text{span}<x\_\text{span})$ | bool | `true` ⟹ quét theo **hàng** |
| $j$ | "giê" | **chỉ số lát** — số thứ tự hàng (hoặc cột) tile đang xét | — | $j=4,5,6$ |
| $j_{\min},j_{\max}$ | | lát đầu / lát cuối, lấy từ bbox | — | $4,\ 6$ |
| $y_j^{lo},y_j^{hi}$ | | hai biên lát $j$, **đã clip vào bbox** | px | hàng 4: $[73.876,\ 80]$ |
| $h$ | "hát" | **độ lệch của đường cắt so với tâm**: $h=y-p_y$ (khi quét hàng) | px | $h^{hi}=80-88=-8$ |
| $k$ | "ca" | hệ số mẫu: $=A$ khi quét hàng, $=C$ khi quét cột | px⁻² | $A=0.027778$ |
| $d_\pm(h)$ | "đê cộng/trừ" | hai **nghiệm giao** của đường $h$ với ellipse, tính từ tâm | px | tại $h=-8$: $-23.640$ và $-0.360$ |
| $d_{\min}^{(j)},d_{\max}^{(j)}$ | | biên trái/phải của ellipse **trên cả lát $j$** | px | hàng 4: $-25.462$ và $-0.360$ |
| $T$ | "tê" | tập tile mà ellipse thật sự chạm | — | $\lvert T\rvert=9$ |

### Vì sao lại có hai nhánh trong $d_{\min}^{(j)}$?

$d_\pm(h)$ là hàm **đơn điệu** theo $h$ trên mỗi nửa ellipse. Nên bình thường cực trị của cả lát nằm ở **một trong hai đầu lát** — chỉ cần so 2 số.

Nhưng nếu **tiếp điểm** rơi *vào giữa* lát thì cực trị nằm bên trong, hai đầu lát đều không bắt được. Lúc đó giá trị đúng chính là nửa cạnh bbox. Đó là toàn bộ nội dung của nhánh `if`:

$$p_y - y_{\text{term}} \in [y_j^{lo},\,y_j^{hi}]\ \Rightarrow\ d_{\min}^{(j)}=-\,\text{half}_x
\qquad
p_y + y_{\text{term}} \in [y_j^{lo},\,y_j^{hi}]\ \Rightarrow\ d_{\max}^{(j)}=+\,\text{half}_x$$

---

## A.7 — Kết quả

| Ký hiệu | Đọc là | Ý nghĩa | Ví dụ |
|---|---|---|---|
| $K_{\text{3dgs}}$ | | số tile của **hộp vuông** 3DGS | $25$ |
| $K_{\text{box}}$ | | số tile của **hộp compact**, chưa lọc ellipse | $15$ |
| $K_{\text{fastgs}}$ | | số tile **thật** sau khi lọc ellipse | $\mathbf{9}$ |
| $R_{\text{tile}}$ | "a-tail" | $K_{\text{fastgs}}/K_{\text{3dgs}}$ — **tỉ số đi vào mô hình chi phí** | $0.360$ |
| $R_{\text{tile}}^{\text{lim}}$ | "a-tail lim" | giá trị tiệm cận theo **diện tích** (giới hạn splat lớn) | $0.1209$ |
| $S$ | "ét" | diện tích hình bao | ellipse: $626.7$ px² |
| $P$ | "pê" | chu vi hình bao — sinh ra "thuế lượng tử hoá" $P/2B$ | ellipse: $\approx121$ px |

---

## A.8 — Đọc thử trọn một lát: hàng $j=4$

Ghép tất cả ký hiệu trên vào một ví dụ chạy tay.

**Bối cảnh:** đã biết bbox $x\in[94.538,145.462]$, $y\in[73.876,102.124]$ ⟹ $N_x=5$, $N_y=3$ ⟹ $\texttt{isY}=(3<5)=$ `true` ⟹ **quét theo hàng**, mỗi hàng giải ra khoảng $x$, dùng $k=A$.

| Bước | Ký hiệu | Tính | Kết quả |
|---|---|---|---|
| 1 | $[y_4^{lo},y_4^{hi}]$ | hàng 4 phủ $[64,80]$, clip vào bbox $y$ | $[73.876,\ 80]$ |
| 2 | $h^{lo},h^{hi}$ | $y-p_y$ | $-14.124,\ -8$ |
| 3 | tiếp điểm dưới | $p_y-y_{\text{term}}=88-11.752$ | $76.248$ |
| 4 | **nằm trong lát?** | $73.876\le76.248\le80$ | **có** ⟹ nhánh 1 |
| 5 | $d_{\min}^{(4)}$ | lấy thẳng $-\text{half}_x$ | $-25.462$ |
| 6 | $d_+(h^{hi})$ | $\dfrac{-0.33333+\sqrt{-0.04938+0.153925}}{0.027778}$ | $-0.360$ |
| 7 | $d_+(h^{lo})$ | tại tiếp điểm, căn $=0$ ⟹ $-Bh/A$ | $-21.186$ |
| 8 | $d_{\max}^{(4)}$ | $\max(-0.360,\,-21.186)$ | $-0.360$ |
| 9 | khoảng $x$ | $p_x+[d_{\min}^{(4)},\,d_{\max}^{(4)}]$ | $[94.538,\ 119.640]$ |
| 10 | $N[\cdot,\cdot]$ | $\lfloor119.640/16\rfloor-\lfloor94.538/16\rfloor+1=7-5+1$ | $\mathbf{3}$ tile |

Đọc lại bước 4 cho kỹ: tiếp điểm **dưới** nằm trong hàng 4, mà tiếp điểm dưới là điểm ellipse **thò ra trái nhất** ⟹ biên trái của lát này chính là biên trái của cả bbox. Không có bước này thì bước 5 sẽ lấy nhầm $\min(d_-(h^{lo}),d_-(h^{hi}))=-23.640$ và **mất tile cột 5**.

Làm tương tự cho $j=5,6$ được $3+3+3=\mathbf{9}$, và bản đồ tile chéo lên (vì $\Sigma'_{12}>0$):

```
        cột 5   6   7   8   9
hàng 4:  ██  ██  ██  ··  ··
hàng 5:  ··  ██  ██  ██  ··
hàng 6:  ··  ··  ██  ██  ██
```
