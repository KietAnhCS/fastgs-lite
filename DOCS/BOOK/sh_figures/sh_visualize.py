"""
Mô phỏng matplotlib cho TOÀN BỘ công thức trong SH.md
(Từ phương trình Laplace đến công thức màu trong 3D Gaussian Splatting)

Mỗi hàm fig_x_y() ứng với đúng một công thức / một mục trong tài liệu,
xuất ra 1 file PNG trong cùng thư mục sh_figures/.

Chạy: python sh_visualize.py
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from scipy.special import sph_harm_y, eval_legendre, assoc_legendre_p
import os

OUT = os.path.dirname(os.path.abspath(__file__))

plt.rcParams["axes.unicode_minus"] = True
plt.rcParams["figure.constrained_layout.use"] = True


def save(fig, name):
    path = os.path.join(OUT, name)
    fig.savefig(path, dpi=140)
    plt.close(fig)
    print("saved:", path)


def sphere_surface(ax, THETA, PHI, values, cmap="RdBu_r", vcenter=None):
    """Ve mat cau ban kinh CO DINH r=1, mau sac the hien gia tri ham (khong
    bien dang ban kinh) — tranh hinh dang meo mo, de doc hon."""
    X = np.sin(THETA) * np.cos(PHI)
    Y = np.sin(THETA) * np.sin(PHI)
    Z = np.cos(THETA)
    vmax = np.max(np.abs(values)) + 1e-12
    norm = (values + vmax) / (2 * vmax) if vcenter is None else (values - values.min()) / (values.max() - values.min() + 1e-12)
    colors = plt.get_cmap(cmap)(norm)
    ax.plot_surface(X, Y, Z, facecolors=colors, rstride=2, cstride=2, linewidth=0, antialiased=True)
    ax.set_box_aspect([1, 1, 1])
    ax.set_axis_off()


def lobe_surface(ax, THETA, PHI, values, cmap="RdBu_r"):
    """Ve dang 'canh hoa' kinh dien cua SH: ban kinh = |gia tri|, mau = dau."""
    Rr = np.abs(values)
    X = Rr * np.sin(THETA) * np.cos(PHI)
    Y = Rr * np.sin(THETA) * np.sin(PHI)
    Z = Rr * np.cos(THETA)
    vmax = np.max(np.abs(values)) + 1e-12
    colors = plt.get_cmap(cmap)((values + vmax) / (2 * vmax))
    ax.plot_surface(X, Y, Z, facecolors=colors, rstride=2, cstride=2, linewidth=0, antialiased=True)
    ax.set_box_aspect([1, 1, 1])
    ax.set_axis_off()


# ---------------------------------------------------------------------------
# PHẦN I
# ---------------------------------------------------------------------------

def fig_1_1_laplace_3d():
    """Eq (1): nabla^2 Phi = 0. Kiểm tra số hóa trên lưới 2D (lát cắt z=const)
    cho hàm điều hòa Phi(x,y,z) = x^2 - y^2 so với Phi = x^2 + y^2 (không điều hòa)."""
    n = 60
    h = 2.0 / n
    x = np.linspace(-1, 1, n)
    y = np.linspace(-1, 1, n)
    z = 0.3
    X, Y = np.meshgrid(x, y)

    def laplacian_xy(f, X, Y, z, h):
        fx2 = (f(X + h, Y, z) - 2 * f(X, Y, z) + f(X - h, Y, z)) / h**2
        fy2 = (f(X, Y + h, z) - 2 * f(X, Y, z) + f(X, Y - h, z)) / h**2
        return fx2 + fy2

    f_harm = lambda X, Y, z: X**2 - Y**2
    f_not = lambda X, Y, z: X**2 + Y**2

    L_harm = laplacian_xy(f_harm, X, Y, z, h)
    L_not = laplacian_xy(f_not, X, Y, z, h)

    fig, axes = plt.subplots(1, 2, figsize=(9, 4.3), constrained_layout=True)
    im0 = axes[0].imshow(L_harm, extent=[-1, 1, -1, 1], cmap="RdBu_r", vmin=-4.2, vmax=4.2)
    axes[0].set_title(r"$\Phi=x^2-y^2$ : $\nabla^2\Phi \approx 0$ (điều hòa)")
    fig.colorbar(im0, ax=axes[0], shrink=0.8)

    im1 = axes[1].imshow(L_not, extent=[-1, 1, -1, 1], cmap="RdBu_r", vmin=-4.2, vmax=4.2)
    axes[1].set_title(r"$\Phi=x^2+y^2$ : $\nabla^2\Phi = 4 \neq 0$")
    fig.colorbar(im1, ax=axes[1], shrink=0.8)
    for ax in axes:
        ax.set_xlabel("x"); ax.set_ylabel("y")
    fig.suptitle(r"1.1 — Phương trình Laplace $\nabla^2\Phi=0$: so sánh số hóa", fontsize=13)
    save(fig, "1_1_laplace_equation.png")


def fig_1_2_spherical_transform():
    """Eq (2)-(3): đổi biến sang tọa độ cầu (r,theta,phi)."""
    theta = np.linspace(0, np.pi, 40)
    phi = np.linspace(0, 2 * np.pi, 80)
    THETA, PHI = np.meshgrid(theta, phi)
    X = np.sin(THETA) * np.cos(PHI)
    Y = np.sin(THETA) * np.sin(PHI)
    Z = np.cos(THETA)

    fig = plt.figure(figsize=(10, 5), constrained_layout=True)
    ax1 = fig.add_subplot(1, 2, 1, projection="3d")
    ax1.plot_wireframe(X, Y, Z, rstride=3, cstride=6, linewidth=0.5, color="steelblue")
    ax1.set_title(r"Lưới $(\theta,\phi)$ trên mặt cầu $r=1$" "\n" r"$x=r\sin\theta\cos\phi,\ y=r\sin\theta\sin\phi,\ z=r\cos\theta$", fontsize=10)
    ax1.set_box_aspect([1, 1, 1])

    ax2 = fig.add_subplot(1, 2, 2)
    ax2.plot(theta, np.sin(theta), label=r"$\sin\theta$ (hệ số Jacobian)")
    ax2.plot(theta, np.cos(theta), label=r"$\cos\theta$")
    ax2.set_xlabel(r"$\theta$")
    ax2.legend()
    ax2.set_title("Các hàm lượng giác trong toán tử Laplace cầu", fontsize=10)
    fig.suptitle("1.2 — Chuyển sang tọa độ cầu", fontsize=13)
    save(fig, "1_2_spherical_coords.png")


def fig_1_2b_separation_and_radial():
    """Eq: tách biến Phi=R(r)Theta(theta)Phi_m(phi) và nghiệm hướng tâm
    R(r) = A r^l + B / r^{l+1}."""
    r = np.linspace(0.2, 2.5, 200)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.3), constrained_layout=True)
    for l in range(4):
        axes[0].plot(r, r**l, label=f"$Ar^{{{l}}}$, l={l}")
    axes[0].set_title(r"Nhánh tăng theo $r$: $R(r)=Ar^l$ (hữu hạn tại gốc)", fontsize=10)
    axes[0].set_xlabel("r"); axes[0].legend(fontsize=8)

    for l in range(4):
        axes[1].plot(r, 1 / r ** (l + 1), label=f"$B/r^{{{l+1}}}$, l={l}")
    axes[1].set_title(r"Nhánh giảm theo $r$: $R(r)=B/r^{l+1}$ (hữu hạn tại $\infty$)", fontsize=10)
    axes[1].set_xlabel("r"); axes[1].set_ylim(0, 5); axes[1].legend(fontsize=8)
    fig.suptitle(r"1.2 — Tách biến: $\Phi=R(r)\Theta(\theta)\Phi_m(\phi)$, nghiệm $R(r)=Ar^l+B/r^{l+1}$", fontsize=12)
    save(fig, "1_2b_radial_solution.png")


def fig_1_3_azimuthal():
    """Eq (6): -1/Phi_m d^2 Phi_m/dphi^2 = m^2  =>  Phi_m(phi) = e^{i m phi}."""
    phi = np.linspace(0, 2 * np.pi, 400)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.3), constrained_layout=True)
    for m in [0, 1, 2, 3]:
        axes[0].plot(phi, np.cos(m * phi), label=f"m={m} (Re)")
    axes[0].set_title(r"Phần thực $\mathrm{Re}(e^{im\phi})=\cos(m\phi)$", fontsize=10)
    axes[0].set_xlabel(r"$\phi$")
    axes[0].legend(fontsize=8)

    ax = axes[1]
    for m in [1, 2, 3]:
        ax.plot(phi, np.unwrap(np.angle(np.exp(1j * m * phi))), label=f"m={m}")
    ax.set_xlabel(r"$\phi$"); ax.set_ylabel(r"pha $\arg(e^{im\phi})$ (rad, đã mở cuộn)")
    ax.set_title(r"Số vòng quấn của $e^{im\phi}$ tỉ lệ với $m$ (chu kỳ $2\pi/m$)", fontsize=10)
    ax.legend(fontsize=8)
    fig.suptitle(r"1.3 — Nghiệm phương vị $\Phi_m(\phi)=e^{im\phi}$, $m\in\mathbb{Z}$", fontsize=13)
    save(fig, "1_3_azimuthal_solution.png")


def fig_1_4_legendre():
    """Eq: P_l(x) (Rodrigues, m=0) và P_l^m(x) (Legendre liên kết)."""
    x = np.linspace(-1, 1, 400)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.6), constrained_layout=True)
    for l in range(6):
        axes[0].plot(x, eval_legendre(l, x), label=f"$P_{l}$")
    axes[0].set_title(r"Đa thức Legendre $P_l(x)=\frac{1}{2^l l!}\frac{d^l}{dx^l}(x^2-1)^l$", fontsize=10)
    axes[0].set_xlabel(r"$x=\cos\theta$")
    axes[0].legend(fontsize=8, ncol=2)

    l_fixed = 3
    for m in range(0, l_fixed + 1):
        y = np.asarray(assoc_legendre_p(l_fixed, m, x)).reshape(-1)
        axes[1].plot(x, y, label=f"$P_{l_fixed}^{{{m}}}$")
    axes[1].set_title(r"Hàm Legendre liên kết $P_l^m(x)=(-1)^m(1-x^2)^{m/2}\frac{d^m}{dx^m}P_l(x)$" f"\n($l={l_fixed}$, $m=0..{l_fixed}$)", fontsize=10)
    axes[1].set_xlabel(r"$x=\cos\theta$")
    axes[1].legend(fontsize=8)
    fig.suptitle("1.4 — Phương trình Legendre liên kết và nghiệm", fontsize=13)
    save(fig, "1_4_legendre_functions.png")


def fig_1_5_ylm_sphere():
    """Eq (boxed) Y_l^m(theta,phi): vẽ phần thực trên mặt cầu cho l=0,1,2,3."""
    theta = np.linspace(0, np.pi, 100)
    phi = np.linspace(0, 2 * np.pi, 200)
    THETA, PHI = np.meshgrid(theta, phi)

    combos = [(0, 0), (1, 0), (1, 1), (2, 1), (2, 2), (3, 2)]
    fig = plt.figure(figsize=(13, 8.5), constrained_layout=True)
    for i, (l, m) in enumerate(combos):
        Yre = sph_harm_y(l, m, THETA, PHI).real
        ax = fig.add_subplot(2, 3, i + 1, projection="3d")
        lobe_surface(ax, THETA, PHI, Yre)
        ax.set_title(f"$Y_{{{l}}}^{{{m}}}$  (phần thực, bán kính = |giá trị|)", fontsize=10)
    fig.suptitle(r"1.5 — $Y_l^m(\theta,\phi)$: nghiệm riêng của Laplace–Beltrami trên $S^2$", fontsize=14)
    save(fig, "1_5_spherical_harmonics_lobes.png")


def fig_1_6_eigenvalue():
    """Eq: nabla^2_{S^2} Y_l^m = -l(l+1) Y_l^m. Kiểm tra số học bằng finite-difference
    trên lưới (theta,phi)."""
    n_theta, n_phi = 120, 240
    theta = np.linspace(1e-3, np.pi - 1e-3, n_theta)
    phi = np.linspace(0, 2 * np.pi, n_phi, endpoint=False)
    dth = theta[1] - theta[0]
    dph = phi[1] - phi[0]
    THETA, PHI = np.meshgrid(theta, phi, indexing="ij")

    results = []
    for l, m in [(1, 0), (2, 0), (2, 1), (3, 1)]:
        Y = sph_harm_y(l, m, THETA, PHI).real
        dY_dth = np.gradient(Y, dth, axis=0)
        term1 = np.gradient(np.sin(THETA) * dY_dth, dth, axis=0) / np.sin(THETA)
        d2Y_dph2 = np.gradient(np.gradient(Y, dph, axis=1), dph, axis=1)
        term2 = d2Y_dph2 / np.sin(THETA) ** 2
        lap = term1 + term2
        mask = slice(10, -10)
        ratio = np.mean(lap[mask, :] / (Y[mask, :] + 1e-9 * np.sign(Y[mask, :]) + 1e-12))
        results.append((l, m, -l * (l + 1), ratio))

    fig, ax = plt.subplots(figsize=(7, 4.6), constrained_layout=True)
    labels = [f"l={l},m={m}" for l, m, _, _ in results]
    exact = [r[2] for r in results]
    numeric = [r[3] for r in results]
    x = np.arange(len(labels))
    w = 0.35
    ax.bar(x - w / 2, exact, w, label=r"lý thuyết: $-l(l+1)$")
    ax.bar(x + w / 2, numeric, w, label="số hóa (finite-difference)")
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_title(r"1.5b — Kiểm chứng trị riêng $\nabla^2_{S^2}Y_l^m=-l(l+1)Y_l^m$", fontsize=12)
    ax.legend()
    save(fig, "1_6_eigenvalue_check.png")


def fig_1_7_orthonormality():
    """Eq: int_{S^2} Y_lm (Y_l'm')^* dOmega = delta_ll' delta_mm'."""
    theta = np.linspace(1e-4, np.pi - 1e-4, 90)
    phi = np.linspace(0, 2 * np.pi, 180, endpoint=False)
    dth = theta[1] - theta[0]
    dph = phi[1] - phi[0]
    THETA, PHI = np.meshgrid(theta, phi, indexing="ij")
    dOmega = np.sin(THETA) * dth * dph

    basis = [(l, m) for l in range(3) for m in range(-l, l + 1)]
    N = len(basis)
    M = np.zeros((N, N), dtype=complex)
    Ys = [sph_harm_y(l, m, THETA, PHI) for l, m in basis]
    for i in range(N):
        for j in range(N):
            M[i, j] = np.sum(Ys[i] * np.conj(Ys[j]) * dOmega)

    fig, ax = plt.subplots(figsize=(6.5, 5.6), constrained_layout=True)
    im = ax.imshow(np.abs(M), cmap="viridis", vmin=0, vmax=1)
    ax.set_xticks(range(N)); ax.set_xticklabels([f"({l},{m})" for l, m in basis], rotation=90, fontsize=7)
    ax.set_yticks(range(N)); ax.set_yticklabels([f"({l},{m})" for l, m in basis], fontsize=7)
    fig.colorbar(im, ax=ax, shrink=0.85, label=r"$|\int Y_{lm}Y_{l'm'}^*\,d\Omega|$")
    ax.set_title("1.5c — Ma trận Gram số hóa\n≈ ma trận đơn vị (trực chuẩn)", fontsize=12)
    save(fig, "1_7_orthonormality_matrix.png")


# ---------------------------------------------------------------------------
# PHẦN II
# ---------------------------------------------------------------------------

def fig_2_1_real_sh_construction():
    """Eq case: tổ hợp Y_l^m phức -> Y_lm^real theo công thức 3 trường hợp
    (m<0, m=0, m>0). Minh họa cho l=1."""
    theta = np.linspace(0, np.pi, 100)
    phi = np.linspace(0, 2 * np.pi, 200)
    THETA, PHI = np.meshgrid(theta, phi)

    def real_sh(l, m):
        if m < 0:
            Yp = sph_harm_y(l, -m, THETA, PHI)
            Ym = sph_harm_y(l, m, THETA, PHI)
            return (1j / np.sqrt(2) * (Ym - (-1) ** m * Yp)).real
        elif m == 0:
            return sph_harm_y(l, 0, THETA, PHI).real
        else:
            Yp = sph_harm_y(l, m, THETA, PHI)
            Ym = sph_harm_y(l, -m, THETA, PHI)
            return (1 / np.sqrt(2) * (Ym + (-1) ** m * Yp)).real

    fig = plt.figure(figsize=(12, 4.6), constrained_layout=True)
    titles = [r"$Y_{1,-1}=\frac{i}{\sqrt{2}}(Y_1^{1}-(-1)^1Y_1^{-1})$",
              r"$Y_{1,0}=Y_1^{0}$",
              r"$Y_{1,1}=\frac{1}{\sqrt{2}}(Y_1^{-1}+(-1)^1Y_1^{1})$"]
    for i, m in enumerate([-1, 0, 1]):
        Yre = real_sh(1, m)
        ax = fig.add_subplot(1, 3, i + 1, projection="3d")
        lobe_surface(ax, THETA, PHI, Yre)
        ax.set_title(titles[i], fontsize=10)
    fig.suptitle(r"2.1 — Xây dựng hàm SH thực từ tổ hợp tuyến tính $Y_l^m$ phức (l=1)", fontsize=13)
    save(fig, "2_1_real_sh_construction.png")


def fig_2_2_low_order_real_sh_xyz():
    """Eq: Y00, Y1,-1=C1 y, Y10=C1 z, Y11=C1 x và hằng số C0,C1."""
    C0 = 0.5 * np.sqrt(1 / np.pi)
    C1 = 0.5 * np.sqrt(3 / np.pi)

    theta = np.linspace(0, np.pi, 100)
    phi = np.linspace(0, 2 * np.pi, 200)
    THETA, PHI = np.meshgrid(theta, phi)
    x = np.sin(THETA) * np.cos(PHI)
    y = np.sin(THETA) * np.sin(PHI)
    z = np.cos(THETA)

    funcs = {
        r"$Y_{00}=C_0$": np.full_like(x, C0),
        r"$Y_{1,-1}=C_1\,y$": C1 * y,
        r"$Y_{1,0}=C_1\,z$": C1 * z,
        r"$Y_{1,1}=C_1\,x$": C1 * x,
    }
    fig = plt.figure(figsize=(13, 4.2), constrained_layout=True)
    for i, (title, F) in enumerate(funcs.items()):
        ax = fig.add_subplot(1, 4, i + 1, projection="3d")
        lobe_surface(ax, THETA, PHI, F)
        ax.set_title(title, fontsize=11)
    fig.suptitle(rf"2.2 — SH thực bậc thấp theo $(x,y,z)$   ($C_0={C0:.4f}$, $C_1={C1:.4f}$)", fontsize=13)
    save(fig, "2_2_low_order_real_sh.png")


# ---------------------------------------------------------------------------
# PHẦN III
# ---------------------------------------------------------------------------

def fig_3_1_fourier_vs_sh():
    """Eq: so sánh chuỗi Fourier 1D f(phi)=sum c_n e^{inphi} với khai triển SH
    trên mặt cầu f(theta,phi) = sum k_lm Y_lm."""
    phi = np.linspace(0, 2 * np.pi, 400)
    target = (phi > np.pi).astype(float) - 0.5  # sóng vuông

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), constrained_layout=True)
    for N in [1, 3, 7, 21]:
        approx = np.zeros_like(phi)
        for n in range(-N, N + 1):
            if n == 0:
                continue
            cn = (1 / (1j * np.pi * n)) * (1 - (-1) ** n)
            approx += (cn * np.exp(1j * n * phi)).real
        axes[0].plot(phi, approx, label=f"N={N}")
    axes[0].plot(phi, target, "k--", lw=1, label="hàm mục tiêu")
    axes[0].set_title(r"Fourier 1D: $f(\phi)=\sum_n c_n e^{in\phi}$", fontsize=10)
    axes[0].legend(fontsize=7)

    theta = np.linspace(1e-4, np.pi - 1e-4, 60)
    phi2 = np.linspace(0, 2 * np.pi, 120)
    THETA, PHI = np.meshgrid(theta, phi2, indexing="ij")
    target_sphere = np.sign(np.cos(THETA))
    dth = theta[1] - theta[0]; dph = phi2[1] - phi2[0]
    dOmega = np.sin(THETA) * dth * dph

    errors = []
    Ds = [0, 1, 2, 3, 5, 8]
    for D in Ds:
        recon = np.zeros_like(THETA, dtype=complex)
        for l in range(D + 1):
            for m in range(-l, l + 1):
                Y = sph_harm_y(l, m, THETA, PHI)
                k_lm = np.sum(target_sphere * np.conj(Y) * dOmega)
                recon += k_lm * Y
        err = np.sqrt(np.sum(np.abs(recon.real - target_sphere) ** 2 * dOmega))
        errors.append(err)
    axes[1].plot(Ds, errors, "o-")
    axes[1].set_xlabel("bậc cắt cụt D")
    axes[1].set_ylabel(r"sai số $L^2$ trên $S^2$")
    axes[1].set_title(r"SH 2D-cầu: $f(\theta,\phi)=\sum_{l,m}k_{lm}Y_{lm}$" "\nsai số giảm khi tăng D", fontsize=10)
    fig.suptitle("3.1 — Chuỗi Fourier (1D) và khai triển SH (2D-cầu)", fontsize=13)
    save(fig, "3_1_fourier_vs_sh.png")


def fig_3_2_truncation_and_coeff_count():
    """Eq: c_i(d) ~ sum_{l=0}^{D} sum_m k_lm Y_lm  và  số hệ số = (D+1)^2."""
    theta = np.linspace(1e-4, np.pi - 1e-4, 60)
    phi = np.linspace(0, 2 * np.pi, 120)
    THETA, PHI = np.meshgrid(theta, phi, indexing="ij")

    true_c = 0.5 + 0.3 * np.cos(THETA) + 0.2 * np.sin(THETA) ** 2 * np.cos(2 * PHI)
    dth = theta[1] - theta[0]; dph = phi[1] - phi[0]
    dOmega = np.sin(THETA) * dth * dph

    fig, axes = plt.subplots(1, 4, figsize=(14, 4), subplot_kw={"projection": "3d"}, constrained_layout=True)
    for ax, D in zip(axes, [0, 1, 2, 3]):
        recon = np.zeros_like(THETA, dtype=complex)
        for l in range(D + 1):
            for m in range(-l, l + 1):
                Y = sph_harm_y(l, m, THETA, PHI)
                k_lm = np.sum(true_c * np.conj(Y) * dOmega)
                recon += k_lm * Y
        sphere_surface(ax, THETA, PHI, recon.real, cmap="viridis", vcenter=True)
        ncoef = (D + 1) ** 2
        ax.set_title(f"D={D}\n(D+1)²={ncoef} hệ số", fontsize=10)
    fig.suptitle(r"3.1/3.2 — Cắt cụt chuỗi SH tại bậc $D$: $c_i(\vec d)\approx\sum_{l=0}^{D}\sum_m k_{i,lm}Y_{lm}$", fontsize=13)
    save(fig, "3_2_truncation_reconstruction.png")

    fig2, ax = plt.subplots(figsize=(5.5, 4.3), constrained_layout=True)
    Ds = np.arange(0, 8)
    ax.bar(Ds, (Ds + 1) ** 2, color="steelblue")
    for D in Ds:
        ax.text(D, (D + 1) ** 2 + 1, str((D + 1) ** 2), ha="center", fontsize=8)
    ax.set_xlabel("D"); ax.set_ylabel("số hệ số / kênh màu")
    ax.set_title(r"Số hệ số cần lưu: $\sum_{l=0}^{D}(2l+1)=(D+1)^2$", fontsize=12)
    save(fig2, "3_2b_coefficient_count.png")


def fig_3_3_compute_color_from_sh():
    """Eq đầy đủ: c_i(d) = max(0, 0.5 + C0 k00 + sum_{l=1}^{3} ...). Mô phỏng
    màu đầy đủ của 1 Gaussian với bộ hệ số k_{lm} ngẫu nhiên, thể hiện cả
    hiệu ứng clamp (max(0,.))."""
    rng = np.random.default_rng(7)
    theta = np.linspace(1e-4, np.pi - 1e-4, 80)
    phi = np.linspace(0, 2 * np.pi, 160)
    THETA, PHI = np.meshgrid(theta, phi, indexing="ij")

    fig = plt.figure(figsize=(13, 4.6), constrained_layout=True)
    channel_names = ["R", "G", "B"]
    for ci, cname in enumerate(channel_names):
        # DC nho + bac cao lon => co vung bi am, de thay hieu ung clamp
        k = {(l, m): rng.normal(0, 0.35) for l in range(4) for m in range(-l, l + 1)}
        k[(0, 0)] = rng.uniform(-0.1, 0.3)

        c_raw = np.zeros_like(THETA)
        for l in range(4):
            for m in range(-l, l + 1):
                Y = sph_harm_y(l, m, THETA, PHI).real
                c_raw += k[(l, m)] * Y
        c_raw = 0.5 + c_raw
        c_clamped = np.clip(c_raw, 0, 1)

        ax = fig.add_subplot(1, 3, ci + 1, projection="3d")
        X = np.sin(THETA) * np.cos(PHI); Y_ = np.sin(THETA) * np.sin(PHI); Z = np.cos(THETA)
        facecolor = np.zeros(THETA.shape + (4,))
        facecolor[..., ci] = c_clamped
        facecolor[..., 3] = 1.0
        ax.plot_surface(X, Y_, Z, facecolors=facecolor, rstride=2, cstride=2, linewidth=0)
        ax.set_box_aspect([1, 1, 1]); ax.set_axis_off()
        frac_clamped = np.mean(c_raw < 0) * 100
        ax.set_title(f"kênh {cname}\n{frac_clamped:.1f}% điểm bị clamp(0,·)", fontsize=10)
    fig.suptitle(r"3.2 — $c_i(\vec d)=\max(0,\ 0.5+C_0k_{00}+\sum_{l=1}^{3}\sum_m C_l^{(m)}P_l^{(m)}(x,y,z)k_{lm})$", fontsize=12)
    save(fig, "3_3_compute_color_from_sh.png")


def fig_3_4_degree_schedule():
    """Eq: D(t) = min(3, floor(t/1000)) — hàm bậc thang theo bước huấn luyện."""
    t = np.linspace(0, 5000, 5001)
    D = np.minimum(3, np.floor(t / 1000))
    fig, ax = plt.subplots(figsize=(7, 4.3), constrained_layout=True)
    ax.step(t, D, where="post", color="darkorange", lw=2)
    ax.set_xlabel("bước huấn luyện t")
    ax.set_ylabel("bậc SH kích hoạt D(t)")
    ax.set_yticks([0, 1, 2, 3])
    ax.set_title(r"3.3 — Lịch tăng bậc $D(t)=\min(3,\lfloor t/1000\rfloor)$", fontsize=12)
    ax.grid(alpha=0.3)
    save(fig, "3_4_degree_schedule.png")


# ---------------------------------------------------------------------------
# PHẦN IV
# ---------------------------------------------------------------------------

def fig_4_1_parseval_energy():
    """Eq: int |c_i|^2 dOmega = sum k_lm^2 (Parseval) và k00 chiếm phần lớn
    năng lượng => tách features_dc / features_rest."""
    rng = np.random.default_rng(1)
    k = {(l, m): rng.normal(0, 0.25 / (l + 1)) for l in range(4) for m in range(-l, l + 1)}
    k[(0, 0)] = 1.2

    theta = np.linspace(1e-4, np.pi - 1e-4, 80)
    phi = np.linspace(0, 2 * np.pi, 160)
    THETA, PHI = np.meshgrid(theta, phi, indexing="ij")
    dth = theta[1] - theta[0]; dph = phi[1] - phi[0]
    dOmega = np.sin(THETA) * dth * dph

    c = np.zeros_like(THETA)
    for (l, m), val in k.items():
        c += val * sph_harm_y(l, m, THETA, PHI).real

    lhs = np.sum(c**2 * dOmega)
    energy_by_l = {}
    for l in range(4):
        energy_by_l[l] = sum(k[(l, m)] ** 2 for m in range(-l, l + 1))
    rhs = sum(energy_by_l.values())

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.6), constrained_layout=True)
    axes[0].bar(["VT\n$\\int|c_i|^2 d\\Omega$", "VP\n$\\sum k_{lm}^2$"], [lhs, rhs], color=["steelblue", "indianred"])
    axes[0].set_title("4.1 — Kiểm chứng Parseval trên $S^2$", fontsize=11)
    for i, v in enumerate([lhs, rhs]):
        axes[0].text(i, v, f"{v:.3f}", ha="center", va="bottom")

    ls = list(energy_by_l.keys())
    vals = list(energy_by_l.values())
    axes[1].bar([str(l) for l in ls], vals, color="seagreen")
    axes[1].set_xlabel("bậc l"); axes[1].set_ylabel(r"năng lượng $\sum_m k_{lm}^2$")
    axes[1].set_title(r"Năng lượng theo bậc: $k_{00}$ ($l=0$) chiếm ưu thế" "\n⇒ tách features_dc / features_rest", fontsize=10)
    for i, v in enumerate(vals):
        axes[1].text(i, v, f"{v:.3f}", ha="center", va="bottom", fontsize=8)
    save(fig, "4_1_parseval_energy.png")


def fig_4_2_mean_color():
    """Eq: bar_c_i = 1/(4pi) int c_i dOmega = C0 * k00 (vì Y_{l>=1} có trung bình 0)."""
    rng = np.random.default_rng(2)
    C0 = 0.5 * np.sqrt(1 / np.pi)
    k00 = 0.8
    k = {(l, m): rng.normal(0, 0.2) for l in range(1, 4) for m in range(-l, l + 1)}
    k[(0, 0)] = k00

    theta = np.linspace(1e-4, np.pi - 1e-4, 80)
    phi = np.linspace(0, 2 * np.pi, 160)
    THETA, PHI = np.meshgrid(theta, phi, indexing="ij")
    dth = theta[1] - theta[0]; dph = phi[1] - phi[0]
    dOmega = np.sin(THETA) * dth * dph

    c = np.zeros_like(THETA)
    for (l, m), val in k.items():
        c += val * sph_harm_y(l, m, THETA, PHI).real

    mean_numeric = np.sum(c * dOmega) / (4 * np.pi)
    mean_theory = C0 * k00

    fig, ax = plt.subplots(figsize=(5.5, 4.6), constrained_layout=True)
    ax.bar(["số hóa\n$\\frac{1}{4\\pi}\\int c_i\\,d\\Omega$", "lý thuyết\n$C_0 k_{00}$"],
           [mean_numeric, mean_theory], color=["slateblue", "goldenrod"])
    for i, v in enumerate([mean_numeric, mean_theory]):
        ax.text(i, v, f"{v:.4f}", ha="center", va="bottom")
    ax.set_title(r"4.2 — Màu trung bình $\bar c_i=\frac{1}{4\pi}\int_{S^2}c_i\,d\Omega=C_0 k_{i,00}$" "\n" r"(các bậc $l\geq1$ có trung bình 0 trên mặt cầu)", fontsize=11)
    save(fig, "4_2_mean_color_dc.png")


ALL_FIGS = [
    fig_1_1_laplace_3d,
    fig_1_2_spherical_transform,
    fig_1_2b_separation_and_radial,
    fig_1_3_azimuthal,
    fig_1_4_legendre,
    fig_1_5_ylm_sphere,
    fig_1_6_eigenvalue,
    fig_1_7_orthonormality,
    fig_2_1_real_sh_construction,
    fig_2_2_low_order_real_sh_xyz,
    fig_3_1_fourier_vs_sh,
    fig_3_2_truncation_and_coeff_count,
    fig_3_3_compute_color_from_sh,
    fig_3_4_degree_schedule,
    fig_4_1_parseval_energy,
    fig_4_2_mean_color,
]

if __name__ == "__main__":
    import sys
    names = sys.argv[1:]
    if names:
        by_name = {f.__name__: f for f in ALL_FIGS}
        for n in names:
            by_name[n]()
    else:
        for f in ALL_FIGS:
            f()
    print("DONE — tất cả hình đã lưu trong", OUT)
