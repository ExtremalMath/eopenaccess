# -*- coding: utf-8 -*-
"""生成课程报告中三张行为分析图（SMO 收敛曲线、α 稀疏性、核映射 3D 视图）。

协议与正文实验一致：鸢尾花单次 7:3 划分、种子 42、RBF 核 gamma=0.5、C=1.0。
输出：smo_convergence.png / alpha_sparsity.png / kernel_3d.png
（与 iris_boundary.png、svm_circles.png、svm_gamma_scan.png、roc_curve.png 同目录）
"""
from __future__ import annotations

import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
C = 1.0
GAMMA = 0.5
SEED = 42


def load_iris_two_features(path):
    X, y = [], []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            c = line.split(",")
            X.append([float(c[0]), float(c[1])])
            y.append(1 if c[4] == "Iris-setosa" else 0)
    return np.asarray(X, dtype=float), np.asarray(y, dtype=int)


def rbf_kernel(gamma):
    def K(a, b):
        sq = (np.sum(a ** 2, axis=1, keepdims=True)
              + np.sum(b ** 2, axis=1, keepdims=True).T - 2.0 * (a @ b.T))
        return np.exp(-gamma * np.clip(sq, 0.0, None))
    return K


def smo_simple(X, y, C, toler, max_iter, K, seed=0, record_obj=False):
    """记录目标值轨迹（当 record_obj=True 时），其余与 svm_demo.py 一致。"""
    m = X.shape[0]
    alphas = np.zeros(m)
    b = 0.0
    rng = np.random.default_rng(seed)
    hard_cap = max_iter * m
    it = 0
    total = 0
    objs = []
    while it < max_iter and total < hard_cap:
        if record_obj:
            q = alphas @ ((y[:, None] * y[None, :] * K) @ alphas)
            objs.append(float(np.sum(alphas) - 0.5 * float(q)))
        alpha_pairs_changed = 0
        for i in range(m):
            f_i = float(alphas @ (y * K[:, i])) + b
            E_i = f_i - y[i]
            if (y[i] * E_i < -toler and alphas[i] < C) or \
               (y[i] * E_i > toler and alphas[i] > 0):
                j = i
                while j == i:
                    j = int(rng.integers(0, m))
                f_j = float(alphas @ (y * K[:, j])) + b
                E_j = f_j - y[j]
                ai_old, aj_old = alphas[i], alphas[j]

                if y[i] != y[j]:
                    L = max(0.0, aj_old - ai_old)
                    H = min(C, C + aj_old - ai_old)
                else:
                    L = max(0.0, aj_old + ai_old - C)
                    H = min(C, aj_old + ai_old)
                if L == H:
                    continue

                eta = 2.0 * K[i, j] - K[i, i] - K[j, j]
                if eta >= 0:
                    continue

                alphas[j] = np.clip(aj_old - y[j] * (E_i - E_j) / eta, L, H)
                if abs(alphas[j] - aj_old) < 1e-5:
                    continue
                alphas[i] = ai_old + y[i] * y[j] * (aj_old - alphas[j])

                b1 = b - E_i - y[i]*(alphas[i]-ai_old)*K[i, i] \
                     - y[j]*(alphas[j]-aj_old)*K[i, j]
                b2 = b - E_j - y[i]*(alphas[i]-ai_old)*K[i, j] \
                     - y[j]*(alphas[j]-aj_old)*K[j, j]
                if 0.0 < alphas[i] < C:
                    b = b1
                elif 0.0 < alphas[j] < C:
                    b = b2
                else:
                    b = (b1 + b2) / 2.0
                alpha_pairs_changed += 1

        total += 1
        it = it + 1 if alpha_pairs_changed == 0 else 0

    return alphas, b, objs


def standardize(X_tr, X_te=None):
    mu = X_tr.mean(axis=0)
    sd = X_tr.std(axis=0)
    sd[sd == 0] = 1.0
    X_tr_s = (X_tr - mu) / sd
    X_te_s = None if X_te is None else (X_te - mu) / sd
    return X_tr_s, X_te_s, mu, sd


def make_svm_predictor(kernel, C):
    def predictor(X_tr, y_tr, X_te):
        X_tr_s, X_te_s, _, _ = standardize(X_tr, X_te)
        from svm_demo import svm_fit, svm_predict
        return svm_predict(svm_fit(X_tr_s, y_tr, C, kernel), X_te_s)
    return predictor


def set_matplotlib_cjk():
    """matplotlib 自带字体不含中文，绘图前须显式指定中文字体。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.sans-serif"] = ["SimHei", "FangSong", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False


def plot_smo_convergence():
    set_matplotlib_cjk()
    import matplotlib.pyplot as plt

    X, y = load_iris_two_features(os.path.join(HERE, "iris.data"))
    rng = np.random.default_rng(SEED)
    idx = rng.permutation(len(y))
    n_tr = len(y) - int(round(0.30 * len(y)))
    tr, te = idx[:n_tr], idx[n_tr:]
    X_tr_s, X_te_s, _, _ = standardize(X[tr], X[te])
    y_tr = np.where(y[tr] == 1, 1.0, -1.0)
    K = rbf_kernel(GAMMA)(X_tr_s, X_tr_s)

    alphas, b, objs = smo_simple(X_tr_s, y_tr, C, 1e-3, 200, K, seed=SEED,
                                 record_obj=True)

    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    ax.plot(range(len(objs)), objs, "-", c="#d62728", lw=1.2)
    ax.set_xlabel("外层迭代轮数（每轮完整扫描一次所有样本）")
    ax.set_ylabel("对偶目标 $W(\\boldsymbol{\\alpha})$")
    ax.set_title("SMO 算法的收敛曲线（鸢尾花训练集，RBF 核 gamma=0.5, C=1）")
    ax.grid(alpha=0.3)
    ax.text(0.02, 0.97, f"收敛值 W = {objs[-1]:.4f}（{len(objs)} 轮）",
            transform=ax.transAxes, va="top", fontsize=8,
            bbox=dict(boxstyle="round", fc="w", ec="gray", lw=0.5))
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "smo_convergence.png"), dpi=150)
    plt.close(fig)
    print(f"[OK] smo_convergence.png  ({len(objs)} iterations)")


def plot_alpha_sparsity():
    set_matplotlib_cjk()
    import matplotlib.pyplot as plt

    X, y = load_iris_two_features(os.path.join(HERE, "iris.data"))
    rng = np.random.default_rng(SEED)
    idx = rng.permutation(len(y))
    n_tr = len(y) - int(round(0.30 * len(y)))
    tr = idx[:n_tr]
    X_tr_s, _, _, _ = standardize(X[tr], None)
    y_tr = np.where(y[tr] == 1, 1.0, -1.0)

    from svm_demo import svm_fit
    m = svm_fit(X_tr_s, y_tr, C, rbf_kernel(GAMMA))
    alphas = m["alphas"]

    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.6))

    is_sv = alphas > 1e-4
    is_free = is_sv & (alphas < C - 1e-6)
    is_bound = is_sv & (alphas >= C - 1e-6)
    n_sv = int(is_sv.sum())
    n_free = int(is_free.sum())
    n_bound = int(is_bound.sum())
    total = len(alphas)

    # 左图：α 值排序散点
    order = np.argsort(-alphas)
    ax1 = axes[0]
    ax1.scatter(np.arange(total)[is_sv[order]], alphas[order][is_sv[order]],
                c="#d62728", s=22, zorder=3, label="支持向量")
    ax1.scatter(np.arange(total)[~is_sv[order]], alphas[order][~is_sv[order]],
                c="#7f7f7f", s=14, marker="x", zorder=2, label="α = 0")
    ax1.axhline(C, c="#2ca02c", ls="--", lw=1, label=f"上界 C = {C}")
    ax1.set_xlabel("样本（按 α 从大到小排列）")
    ax1.set_ylabel(r"$\alpha_i$")
    ax1.set_title(f"α 的稀疏性（{n_sv}/{total} = {n_sv/total:.1%} 为支持向量）")
    ax1.legend(fontsize=7)
    ax1.grid(alpha=0.3)

    # 右图：三类计数
    ax2 = axes[1]
    labels = ["α = 0\n(非 SV)", "自由 SV\n(0<α<C)", "边界 SV\n(α=C)"]
    counts = [total - n_sv, n_free, n_bound]
    cols = ["#7f7f7f", "#d62728", "#2ca02c"]
    bars = ax2.bar(labels, counts, color=cols, width=0.5)
    for bar, c in zip(bars, counts):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                 f"{c}", ha="center", fontsize=9, fontweight="bold")
    ax2.set_ylabel("样本数")
    ax2.set_title("训练集支持向量构成")
    ax2.set_ylim(0, max(counts) * 1.25)
    ax2.grid(alpha=0.2, axis="y")

    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "alpha_sparsity.png"), dpi=150)
    plt.close(fig)
    print(f"[OK] alpha_sparsity.png  (SV {n_sv}/{total}, free {n_free}, bound {n_bound})")


def plot_kernel_3d():
    set_matplotlib_cjk()
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    from svm_demo import make_circles
    X, y = make_circles(200, seed=SEED)
    z = X[:, 0] ** 2 + X[:, 1] ** 2

    fig = plt.figure(figsize=(7.0, 4.8))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(X[y == 0, 0], X[y == 0, 1], z[y == 0],
               c="#1f77b4", s=14, alpha=0.7, label="外环 (y=0)")
    ax.scatter(X[y == 1, 0], X[y == 1, 1], z[y == 1],
               c="#d62728", s=14, alpha=0.9, label="内圆 (y=1)")
    # 分隔平面 z = 0.55（手工特征下两类的天然分界）
    xx, yy = np.meshgrid(np.linspace(-1.3, 1.3, 24), np.linspace(-1.3, 1.3, 24))
    ax.plot_surface(xx, yy, np.full_like(xx, 0.55),
                    color="gray", alpha=0.12, linewidth=0)
    ax.set_xlabel("$x_1$")
    ax.set_ylabel("$x_2$")
    ax.set_zlabel("$z = x_1^2 + x_2^2$")
    ax.set_title("核技巧升维的直观示意（手工特征 z = x1² + x2²）")
    ax.view_init(elev=22, azim=-52)
    ax.legend(fontsize=7, loc="upper left")
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "kernel_3d.png"), dpi=150)
    plt.close(fig)
    print("[OK] kernel_3d.png")


if __name__ == "__main__":
    plot_smo_convergence()
    plot_alpha_sparsity()
    plot_kernel_3d()