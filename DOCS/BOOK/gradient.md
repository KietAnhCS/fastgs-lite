Trước tiên, vì bản dán bị lỗi OCR (tử số/mẫu số bị đảo, công thức bị lặp), mình đoán đây là **backward pass của 3D Gaussian Splatting** (alpha-blending + Gaussian dạng conic). Mình viết lại đúng ký hiệu chuẩn rồi chứng minh từng bước.

## Thiết lập

**Blending phía trước (forward):**
$$C=\sum_{n=1}^{N} c_n\,\alpha_n(x)\,T_n,\qquad T_n=\prod_{m=1}^{n-1}\bigl(1-\alpha_m(x)\bigr)$$

với $\alpha_n(x)=o_n\,G_n(x)$ ($o_n$: opacity học được, $G_n(x)$: trọng số Gaussian 2D tại pixel).

**Gaussian dạng conic:**
$$G(\Delta u,\Delta v)=\exp\!\Bigl(-\tfrac12\bigl(A\Delta u^2+2B\Delta u\Delta v+C\Delta v^2\bigr)\Bigr),\qquad \Delta u=u-\mu_u',\;\Delta v=v-\mu_v'$$

trong đó $\begin{pmatrix}A&B\\B&C\end{pmatrix}=\Sigma'^{-1}$ (nghịch đảo hiệp phương sai 2D).

Cho trước gradient ngược $\dfrac{\partial L}{\partial C}$, ta cần lan truyền về $c_n, o_n, \mu_u',\mu_v', A,B,C$.

---

## 1. Gradient theo màu $c_n$

$C$ tuyến tính theo $c_n$ với hệ số $\alpha_n T_n$, các số hạng khác không phụ thuộc $c_n$:

$$\frac{\partial C}{\partial c_n}=\alpha_nT_n \;\Longrightarrow\; \boxed{\frac{\partial L}{\partial c_n}=\alpha_nT_n\,\frac{\partial L}{\partial C}}$$

## 2. Gradient theo alpha tại pixel, $\partial L/\partial\alpha_n(x)$

Đây là phần không tầm thường vì $\alpha_n$ vừa xuất hiện trực tiếp ở số hạng $n$, vừa nằm trong $T_m$ ($m>n$). Viết $C$ tách riêng phần trước và sau $n$:

$$C = \underbrace{\sum_{m<n}c_m\alpha_mT_m}_{\text{không phụ thuộc }\alpha_n} + c_n\alpha_nT_n + T_n(1-\alpha_n)\underbrace{\sum_{m>n}c_m\frac{\alpha_mT_m}{T_n(1-\alpha_n)}}_{=:S_n\text{ (màu tích lũy còn lại)}}$$

Đạo hàm trực tiếp:
$$\frac{\partial C}{\partial \alpha_n}=c_nT_n - T_nS_n = T_n(c_n-S_n)$$

$$\Longrightarrow \frac{\partial L}{\partial \alpha_n(x)}=T_n(c_n-S_n)\,\frac{\partial L}{\partial C}$$

(Đây là "công thức tích lũy ngược" nổi tiếng trong 3DGS — $S_n$ tính đệ quy từ pixel sau ra trước, không cần lưu toàn bộ forward.)

## 3. Tách $\alpha_n(x)=o_n G_n(x)$ bằng quy tắc tích

Vì $\alpha_n(x)$ là **tích** của $o_n$ (hằng theo pixel) và $G_n(x)$:

$$\frac{\partial \alpha_n(x)}{\partial o_n}=G_n(x),\qquad \frac{\partial \alpha_n(x)}{\partial G_n}=o_n$$

Áp dụng chain rule:
$$\boxed{\frac{\partial L}{\partial o_n}=G_n(x)\,\frac{\partial L}{\partial\alpha_n(x)}},\qquad \boxed{\frac{\partial L}{\partial G_n}=o_n\,\frac{\partial L}{\partial\alpha_n(x)}}$$

— đúng hai công thức bạn dán (chỉ là ký hiệu $\alpha_n$ ở đó đóng vai trò $o_n$).

## 4. Đạo hàm $G$ theo $\Delta u,\Delta v$

Đặt $E=-\tfrac12(A\Delta u^2+2B\Delta u\Delta v+C\Delta v^2)$ sao cho $G=e^E$. Vì $\dfrac{\partial G}{\partial \Delta u}=G\cdot\dfrac{\partial E}{\partial \Delta u}$:

$$\frac{\partial E}{\partial \Delta u}=-\tfrac12(2A\Delta u+2B\Delta v)=-(A\Delta u+B\Delta v)$$
$$\frac{\partial E}{\partial \Delta v}=-\tfrac12(2B\Delta u+2C\Delta v)=-(B\Delta u+C\Delta v)$$

$$\boxed{\frac{\partial G}{\partial \Delta u}=-G(A\Delta u+B\Delta v)},\qquad \boxed{\frac{\partial G}{\partial \Delta v}=-G(C\Delta v+B\Delta u)}$$

## 5. Đạo hàm theo tâm chiếu $\mu_u',\mu_v'$

$\mu_u'$ thường được lưu ở toạ độ chuẩn hoá NDC $\in[-1,1]$, còn $\Delta u$ tính ở toạ độ pixel:
$$u_{\text{pixel}}=\frac{(\text{ndc}_u+1)\,W}{2}\;\Longrightarrow\; \frac{\partial \Delta u}{\partial \mu_u'}=-\frac{\partial \mu_{u,\text{pixel}}}{\partial \mu_u'}=-\frac{W}{2}$$

(tương tự $-H/2$ cho trục $v$). Kết hợp chain rule qua $G$:

$$\frac{\partial L}{\partial\mu_u'}=\frac{\partial L}{\partial G}\cdot\frac{\partial G}{\partial \Delta u}\cdot\left(-\frac{W}{2}\right)=\frac{W}{2}\,G(A\Delta u+B\Delta v)\,\frac{\partial L}{\partial G}$$

Đây chính là "$\dfrac{\partial L}{\partial G}\dfrac{\partial G}{\partial \Delta u}\cdot\dfrac{W}{2}$" (dấu trừ đã gộp vào định nghĩa $\partial G/\partial \Delta u$ nếu quy ước $\Delta u=\mu_u'-u$ thay vì $u-\mu_u'$ — hai quy ước lệch nhau đúng một dấu, tuỳ code cụ thể). Tương tự cho $\mu_v'$ với $H/2$.

## 6. Đạo hàm theo $A,B,C$ (các phần tử ma trận conic)

Từ $E=-\tfrac12(A\Delta u^2+2B\Delta u\Delta v+C\Delta v^2)$, coi $\Delta u,\Delta v$ cố định (không phụ thuộc $A,B,C$):

$$\frac{\partial E}{\partial A}=-\tfrac12\Delta u^2,\quad \frac{\partial E}{\partial B}=-\Delta u\Delta v,\quad \frac{\partial E}{\partial C}=-\tfrac12\Delta v^2$$

Nhân với $G=e^E$ để ra $\partial G/\partial(\cdot)$, rồi nhân tiếp $\partial L/\partial G$:

$$\boxed{\frac{\partial L}{\partial A}=-\tfrac12G\Delta u^2\,\frac{\partial L}{\partial G}},\quad \boxed{\frac{\partial L}{\partial B}=-G\Delta u\Delta v\,\frac{\partial L}{\partial G}},\quad \boxed{\frac{\partial L}{\partial C}=-\tfrac12G\Delta v^2\,\frac{\partial L}{\partial G}}$$

(hệ số của $B$ không có $\tfrac12$ vì số hạng chéo $2B\Delta u\Delta v$ đã gấp đôi trong $E$; công thức bạn dán có $\tfrac12$ có lẽ do quy ước $A,B,C$ khác — nếu conic viết là $A\Delta u^2+B\Delta u\Delta v+C\Delta v^2$ thay vì $A\Delta u^2+2B\Delta u\Delta v+C\Delta v^2$ thì cả ba hệ số đều có $-\tfrac12 G(\cdot)$ đối xứng như bạn viết).

---

**Tóm lại chuỗi truyền ngược đầy đủ:**
$$\frac{\partial L}{\partial C}\to \frac{\partial L}{\partial c_n},\ \frac{\partial L}{\partial\alpha_n(x)} \to \frac{\partial L}{\partial o_n},\frac{\partial L}{\partial G_n}\to\frac{\partial L}{\partial \Delta u},\frac{\partial L}{\partial \Delta v}\to \frac{\partial L}{\partial \mu_u'},\frac{\partial L}{\partial \mu_v'},\frac{\partial L}{\partial A},\frac{\partial L}{\partial B},\frac{\partial L}{\partial C_{\text{conic}}}$$

rồi $A,B,C$ tiếp tục lan về scale/rotation của Gaussian 3D qua $\Sigma=RSS^\top R^\top$ (bước tiếp theo nếu bạn cần).
