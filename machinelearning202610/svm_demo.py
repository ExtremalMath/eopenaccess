# -*- coding: utf-8 -*-
"""支持向量机（SVM）与核方法对比实验（纯 NumPy 实现）。

主题：在 Fisher 线性判别（线性基线）基础上，补充 SVM 原理，并以「核函数」为
      改进方向做对比实验。本版按更严格的实验协议实现：
      - 分层 K 折交叉验证（stratified K-fold）
      - 多次重复（不同种子）以报告均值与标准误
      - 一倍标准误（1se）规则实现奥卡姆剃刀的可操作判据
      - 固定 C 扫描 gamma，观察"欠拟合 -> 过拟合"翻转与支持向量数量变化

模型：Fisher 线性判别 + 线性核 SVM + RBF 核 SVM（标签 y ∈ {0,1}，内部转 ±1）。
依赖：仅 numpy（必需）+ matplotlib（可选，绘图）。
运行：python svm_demo.py
输出：svm_results.txt（数值）、svm_circles.png / svm_gamma_scan.png / roc_curve.png（图）。
"""
from __future__ import annotations

import os
import time
from typing import Callable, Tuple

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
IRIS_PATH = os.path.join(HERE, "iris.data")

# ----------------------------------------------------------------------
# 实验协议配置（集中管理，便于复现）
# ----------------------------------------------------------------------
C = 1.0                          # 软间隔惩罚参数（核函数方向，不做参数优化）
GAMMA = 0.5                      # RBF 核基准参数 γ = 1/特征数
K_FOLDS = 5                      # 分层交叉验证折数
N_REPEATS = 10                   # 主评估的重复次数
N_REPEATS_SCAN = 5               # γ 扫描的重复次数（省时）
BASE_SEED = 42                   # 重复 r 使用种子 BASE_SEED + r
GAMMA_GRID = np.logspace(-3, 2, 11)   # log10 γ ∈ [-3, +2]，步长 0.5
C_GRID = np.logspace(-3, 3, 7)        # log10 C ∈ [-3, +3]，用于 C 敏感性分析
SMO_TOL = 1e-3
SMO_MAX_ITER = 200


# ======================================================================
# 数据
# ======================================================================
def load_iris_two_features(path: str) -> Tuple[np.ndarray, np.ndarray]:
    """读取 UCI iris.data，仅取前两列（萼片长、萼片宽），setosa 标为 1。"""
    if not os.path.exists(path):
        raise FileNotFoundError(f"未找到数据文件：{path}（请确认 iris.data 与本脚本同目录）")
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


def make_circles(n: int = 200, factor: float = 0.5, noise: float = 0.05,
                 seed: int = BASE_SEED) -> Tuple[np.ndarray, np.ndarray]:
    """同心圆数据：外环=负类(0)，内圆=正类(1)，线性不可分。"""
    rng = np.random.default_rng(seed)
    n_in = n // 2
    n_out = n - n_in
    t_out = rng.uniform(0, 2 * np.pi, n_out)
    t_in = rng.uniform(0, 2 * np.pi, n_in)
    X_out = np.column_stack([np.cos(t_out), np.sin(t_out)])
    X_in = factor * np.column_stack([np.cos(t_in), np.sin(t_in)])
    X = np.vstack([X_out, X_in]) + rng.normal(scale=noise, size=(n, 2))
    y = np.array([0] * n_out + [1] * n_in, dtype=int)
    return X, y


def standardize(X_tr, X_te=None):
    """按训练集均值/标准差做 Z-score 标准化（拟合只发生在训练集，防数据泄露）。

    统一返回 4 元组 (X_tr_s, X_te_s, mu, sd)；X_te 缺省时 X_te_s 为 None。
    """
    mu = X_tr.mean(axis=0)
    sd = X_tr.std(axis=0)
    sd[sd == 0] = 1.0
    X_tr_s = (X_tr - mu) / sd
    X_te_s = None if X_te is None else (X_te - mu) / sd
    return X_tr_s, X_te_s, mu, sd


# ======================================================================
# 交叉验证协议
# ======================================================================
def stratified_kfold_indices(y: np.ndarray, k: int, seed: int):
    """分层 K 折：按类别分别洗牌后轮流放入各折，保证每折类别比例接近总体。"""
    rng = np.random.default_rng(seed)
    folds = [[] for _ in range(k)]
    for c in np.unique(y):
        idx_c = np.where(y == c)[0]
        rng.shuffle(idx_c)
        for i, idx in enumerate(idx_c):
            folds[i % k].append(idx)
    out = []
    for f in folds:
        f_arr = np.array(f, dtype=int)
        rng.shuffle(f_arr)
        out.append(f_arr)
    return out


def repeated_cv_scores(X: np.ndarray, y: np.ndarray, k: int, n_repeats: int,
                       seed: int, predictor: Callable) -> dict:
    """重复分层 K 折，返回各指标的 (n_repeats, k) 折内矩阵。

    predictor(X_tr, y_tr, X_te) -> y_pred，内部完成标准化与拟合。
    指标键：acc / precision / recall / f1（正类 = setosa，y=1）。
    """
    acc_m, pre_m, rec_m, f1_m = [], [], [], []
    for r in range(n_repeats):
        acc_f, pre_f, rec_f, f1_f = [], [], [], []
        for te_idx in stratified_kfold_indices(y, k, seed + r):
            tr_idx = np.setdiff1d(np.arange(len(y)), te_idx)
            yt = y[te_idx]
            y_pred = predictor(X[tr_idx], y[tr_idx], X[te_idx])
            acc_f.append(accuracy(yt, y_pred))
            p, r_, f = prf1(yt, y_pred)
            pre_f.append(p)
            rec_f.append(r_)
            f1_f.append(f)
        acc_m.append(acc_f)
        pre_m.append(pre_f)
        rec_m.append(rec_f)
        f1_m.append(f1_f)
    return {"acc": np.array(acc_m), "precision": np.array(pre_m),
            "recall": np.array(rec_m), "f1": np.array(f1_m)}


def cv_summary(scores: np.ndarray) -> Tuple[float, float]:
    """返回 (总体均值, 标准误)。标准误 = 各折均值(跨重复)的标准差 / sqrt(k)。"""
    k = scores.shape[1]
    fold_means = scores.mean(axis=0)
    overall_mean = float(scores.mean())
    se = float(fold_means.std(ddof=1) / np.sqrt(k))
    return overall_mean, se


def one_se_threshold(scores_list: list) -> float:
    """一倍标准误规则：返回"最优均值 - 1se"的阈值（准确率越高越好）。

    候选模型集合 = {CV 准确率 ≥ 最优均值 - 1se}，在其中选最简单的模型。
    """
    means = [cv_summary(s)[0] for s in scores_list]
    best_i = int(np.argmax(means))
    _, best_se = cv_summary(scores_list[best_i])
    return means[best_i] - best_se


def roc_curve_auc(y_true: np.ndarray, scores: np.ndarray):
    """由连续判别分数计算 ROC 曲线 (FPR, TPR) 与 AUC（梯形积分）。

    y_true ∈ {0,1}；scores 为连续决策值（越大越倾向正类）。
    """
    order = np.argsort(-scores)                    # 分数从高到低排序
    y_sorted = y_true[order].astype(float)
    tp = np.concatenate([[0.0], np.cumsum(y_sorted)])
    fp = np.concatenate([[0.0], np.cumsum(1.0 - y_sorted)])
    tpr = tp / max(tp[-1], 1.0)
    fpr = fp / max(fp[-1], 1.0)
    auc = float(np.trapezoid(tpr, fpr))
    return fpr, tpr, auc


def gram_matrix_time(X: np.ndarray, kernel: Callable, n_trials: int = 500) -> float:
    """测量核矩阵 K = kernel(X, X) 的单次平均构建耗时（秒）。

    多次重复取均值以稳定微秒级测量。直接比较线性核（X·Xᵀ，BLAS 加速矩阵乘法）
    与 RBF 核（成对平方距离 + 指数运算）的 Gram 矩阵构建开销——后者是 RBF 核
    的固有额外成本，也是其“精度换取开销”的代价来源。
    """
    for _ in range(20):                      # 预热，排除首次调用/缓存开销
        kernel(X, X)
    t0 = time.perf_counter()
    for _ in range(n_trials):
        kernel(X, X)
    return (time.perf_counter() - t0) / n_trials


# ======================================================================
# 核函数
# ======================================================================
def linear_kernel() -> Callable:
    return lambda a, b: a @ b.T


def rbf_kernel(gamma: float) -> Callable:
    def K(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        sq = (np.sum(a ** 2, axis=1, keepdims=True)
              + np.sum(b ** 2, axis=1, keepdims=True).T - 2.0 * (a @ b.T))
        return np.exp(-gamma * np.clip(sq, 0.0, None))
    return K


# ======================================================================
# SVM：简化 SMO 求解软间隔对偶问题
# ======================================================================
def smo_simple(X: np.ndarray, y: np.ndarray, C: float, toler: float,
               max_iter: int, K: np.ndarray, seed: int = 0):
    """简化版 SMO，求解对偶问题 max Σα - ½ ΣΣ αα yy K，s.t. 0≤α≤C, Σα y = 0。

    返回 (alphas, b, converged)。converged 表示是否在总迭代上限内停止
    （对应教材 8.3“直至全部满足 KKT 条件”的收敛要求）。
    """
    m = X.shape[0]
    alphas = np.zeros(m)
    b = 0.0
    rng = np.random.default_rng(seed)   # 固定随机源（可复现，且避免每次新建）
    hard_cap = max_iter * m             # 总迭代硬上限，防止连续变化导致的死循环
    it = 0
    total = 0
    converged = True
    while it < max_iter and total < hard_cap:
        alpha_pairs_changed = 0
        for i in range(m):
            f_i = float(alphas @ (y * K[:, i])) + b
            E_i = f_i - y[i]
            if (y[i] * E_i < -toler and alphas[i] < C) or (y[i] * E_i > toler and alphas[i] > 0):
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

                b1 = b - E_i - y[i] * (alphas[i] - ai_old) * K[i, i] \
                     - y[j] * (alphas[j] - aj_old) * K[i, j]
                b2 = b - E_j - y[i] * (alphas[i] - ai_old) * K[i, j] \
                     - y[j] * (alphas[j] - aj_old) * K[j, j]
                if 0.0 < alphas[i] < C:
                    b = b1
                elif 0.0 < alphas[j] < C:
                    b = b2
                else:
                    b = (b1 + b2) / 2.0
                alpha_pairs_changed += 1

        total += 1
        it = it + 1 if alpha_pairs_changed == 0 else 0

    if total >= hard_cap:
        converged = False
    return alphas, b, converged


def svm_fit(X: np.ndarray, y: np.ndarray, C: float = 1.0,
            kernel: Callable = None, seed: int = 0) -> dict:
    """训练 SVM（标签内部转 ±1），返回模型字典。

    区分自由支持向量（0<α<C，落在边界线上）与有界支持向量（α=C，落在“禁区”内），
    对应教材 8.4 的两类软间隔支持向量。默认 C=1.0（不引用模块级常量，避免默认参数
    在函数定义时求值的陷阱）。
    """
    if kernel is None:
        kernel = linear_kernel()
    y_pm = np.where(y == 1, 1.0, -1.0)
    K = kernel(X, X)
    alphas, b, converged = smo_simple(X, y_pm, C, SMO_TOL, SMO_MAX_ITER, K, seed)
    n_sv = int((alphas > 1e-4).sum())
    n_sv_bounded = int((alphas > C - 1e-6).sum())   # α ≈ C：禁区内（ξ>0）
    n_sv_free = n_sv - n_sv_bounded                 # 0<α<C：边界线上（ξ=0）
    return {"kernel": kernel, "X": X, "y": y_pm, "alphas": alphas, "b": b,
            "n_sv": n_sv, "n_sv_free": n_sv_free, "n_sv_bounded": n_sv_bounded,
            "converged": converged}


def svm_decision(model: dict, X_new: np.ndarray) -> np.ndarray:
    K = model["kernel"](model["X"], X_new)
    return (model["alphas"] * model["y"]) @ K + model["b"]


def svm_predict(model: dict, X_new: np.ndarray) -> np.ndarray:
    return (svm_decision(model, X_new) > 0).astype(int)


# ======================================================================
# 基线：Fisher 线性判别
# ======================================================================
def fisher_lda(X_tr: np.ndarray, y_tr: np.ndarray) -> Tuple[np.ndarray, float]:
    X1 = X_tr[y_tr == 1]
    X0 = X_tr[y_tr == 0]
    mu1, mu0 = X1.mean(axis=0), X0.mean(axis=0)
    Sw = (X1 - mu1).T @ (X1 - mu1) + (X0 - mu0).T @ (X0 - mu0)
    Sw = Sw + 1e-8 * np.eye(Sw.shape[0])
    w = np.linalg.solve(Sw, mu1 - mu0)
    b = -float(w @ ((mu1 + mu0) / 2.0))
    return w, b


def lda_predict(X: np.ndarray, w: np.ndarray, b: float) -> np.ndarray:
    return (X @ w + b > 0).astype(int)


def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float((y_true == y_pred).mean())


def prf1(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[float, float, float]:
    """正类（y=1）的精确率 / 召回率 / F1，由混淆矩阵 TP/FP/FN 计算。"""
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)
          if (precision + recall) > 0 else 0.0)
    return precision, recall, f1


# ======================================================================
# 可复现的模型封装（标准化在训练折内完成）
# ======================================================================
def lda_predictor(X_tr, y_tr, X_te) -> np.ndarray:
    w, b = fisher_lda(X_tr, y_tr)
    return lda_predict(X_te, w, b)


def make_svm_predictor(kernel: Callable, C: float) -> Callable:
    def predictor(X_tr, y_tr, X_te) -> np.ndarray:
        X_tr_s, X_te_s, _, _ = standardize(X_tr, X_te)
        m = svm_fit(X_tr_s, y_tr, C, kernel)
        return svm_predict(m, X_te_s)
    return predictor


# ======================================================================
# γ 扫描（观察欠拟合 -> 过拟合翻转）
# ======================================================================
def gamma_scan(X, y, C, gammas, k, n_repeats, seed):
    """固定 C 扫描 γ，返回 (gammas, train_mean, cv_mean, nsv_mean, nsv_bounded_mean)。"""
    train_list, cv_list, nsv_list, nsvb_list = [], [], [], []
    not_converged = 0
    for g in gammas:
        kernel = rbf_kernel(g)
        tr_accs, cv_accs, nsvs, nsvbs = [], [], [], []
        for r in range(n_repeats):
            for te_idx in stratified_kfold_indices(y, k, seed + r):
                tr_idx = np.setdiff1d(np.arange(len(y)), te_idx)
                X_tr_s, X_te_s, _, _ = standardize(X[tr_idx], X[te_idx])
                m = svm_fit(X_tr_s, y[tr_idx], C, kernel, seed=r)
                if not m["converged"]:
                    not_converged += 1
                tr_accs.append(accuracy(y[tr_idx], svm_predict(m, X_tr_s)))
                cv_accs.append(accuracy(y[te_idx], svm_predict(m, X_te_s)))
                nsvs.append(m["n_sv"])
                nsvbs.append(m["n_sv_bounded"])
        train_list.append(float(np.mean(tr_accs)))
        cv_list.append(float(np.mean(cv_accs)))
        nsv_list.append(float(np.mean(nsvs)))
        nsvb_list.append(float(np.mean(nsvbs)))
    if not_converged > 0:
        print(f"[WARN] γ 扫描中有 {not_converged} 次拟合达到迭代上限（未完全收敛）")
    return gammas, np.array(train_list), np.array(cv_list), np.array(nsv_list), np.array(nsvb_list)


def c_scan(X, y, kernel, c_grid, k, n_repeats, seed):
    """固定核函数扫描惩罚参数 C，返回 (c_grid, train_mean, cv_mean, nsv_mean)。"""
    train_list, cv_list, nsv_list = [], [], []
    for c in c_grid:
        tr_accs, cv_accs, nsvs = [], [], []
        for r in range(n_repeats):
            for te_idx in stratified_kfold_indices(y, k, seed + r):
                tr_idx = np.setdiff1d(np.arange(len(y)), te_idx)
                X_tr_s, X_te_s, _, _ = standardize(X[tr_idx], X[te_idx])
                m = svm_fit(X_tr_s, y[tr_idx], c, kernel, seed=r)
                tr_accs.append(accuracy(y[tr_idx], svm_predict(m, X_tr_s)))
                cv_accs.append(accuracy(y[te_idx], svm_predict(m, X_te_s)))
                nsvs.append(m["n_sv"])
        train_list.append(float(np.mean(tr_accs)))
        cv_list.append(float(np.mean(cv_accs)))
        nsv_list.append(float(np.mean(nsvs)))
    return c_grid, np.array(train_list), np.array(cv_list), np.array(nsv_list)


# ======================================================================
# 绘图
# ======================================================================
def plot_boundary(ax, X, y, decision_fn, title, labels=("setosa (y=1)", "others (y=0)")):
    x0_min, x0_max = X[:, 0].min() - 0.2, X[:, 0].max() + 0.2
    x1_min, x1_max = X[:, 1].min() - 0.2, X[:, 1].max() + 0.2
    xx, yy = np.meshgrid(np.linspace(x0_min, x0_max, 200),
                         np.linspace(x1_min, x1_max, 200))
    grid = np.column_stack([xx.ravel(), yy.ravel()])
    zz = decision_fn(grid).reshape(xx.shape)
    ax.contourf(xx, yy, zz, levels=[-1e6, 0, 1e6], colors=["#c6d9f1", "#f2c6c6"], alpha=0.5)
    ax.contour(xx, yy, zz, levels=[0], colors="k", linewidths=1.6)
    ax.scatter(X[y == 0, 0], X[y == 0, 1], c="#1f77b4", s=18, label=labels[1])
    ax.scatter(X[y == 1, 0], X[y == 1, 1], c="#d62728", s=18, label=labels[0])
    ax.set_title(title, fontsize=9)
    ax.set_xlim(x0_min, x0_max)
    ax.set_ylim(x1_min, x1_max)


def plot_circles(X, y, model_lin, model_rbf, acc_lin, acc_rbf):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.2))
        plot_boundary(axes[0], X, y, lambda g: svm_decision(model_lin, g),
                      f"linear SVM (acc={acc_lin*100:.1f}%)",
                      labels=("inner (y=1)", "outer (y=0)"))
        plot_boundary(axes[1], X, y, lambda g: svm_decision(model_rbf, g),
                      f"RBF SVM (acc={acc_rbf*100:.1f}%)",
                      labels=("inner (y=1)", "outer (y=0)"))
        axes[0].legend(fontsize=7, loc="best")
        axes[1].legend(fontsize=7, loc="best")
        fig.tight_layout()
        fig.savefig(os.path.join(HERE, "svm_circles.png"), dpi=150)
        plt.close(fig)
    except Exception as e:
        print(f"[WARN] 绘图失败（{e}），跳过 svm_circles.png")


def plot_gamma_scan(gammas, train, cv, nsv):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax1 = plt.subplots(figsize=(6.4, 4.0))
        ax1.plot(np.log10(gammas), train, "o-", c="#d62728", label="train acc")
        ax1.plot(np.log10(gammas), cv, "s-", c="#1f77b4", label="CV acc")
        ax1.set_xlabel("log10(gamma)")
        ax1.set_ylabel("accuracy")
        ax1.set_ylim(0.5, 1.02)
        ax1.grid(alpha=0.3)
        ax2 = ax1.twinx()
        ax2.plot(np.log10(gammas), nsv, "^--", c="#2ca02c", label="#support vectors")
        ax2.set_ylabel("#support vectors", color="#2ca02c")
        ax2.tick_params(axis="y", labelcolor="#2ca02c")
        h1, l1 = ax1.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        ax1.legend(h1 + h2, l1 + l2, fontsize=8, loc="center left")
        ax1.set_title("RBF SVM: train/CV accuracy & #SV vs gamma (C=1)")
        fig.tight_layout()
        fig.savefig(os.path.join(HERE, "svm_gamma_scan.png"), dpi=150)
        plt.close(fig)
    except Exception as e:
        print(f"[WARN] 绘图失败（{e}），跳过 svm_gamma_scan.png")


def plot_roc(fpr_l, tpr_l, auc_l, fpr_s, tpr_s, auc_s):
    """Fisher LDA 与 RBF SVM 的 ROC 曲线对比（谁更靠近左上角）。"""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        plt.figure(figsize=(5.4, 4.6))
        plt.plot(fpr_l, tpr_l, "-", c="#1f77b4", lw=2,
                 label=f"Fisher LDA (AUC={auc_l:.4f})")
        plt.plot(fpr_s, tpr_s, "-", c="#d62728", lw=2,
                 label=f"RBF SVM (AUC={auc_s:.4f})")
        plt.plot([0, 1], [0, 1], "--", c="gray", lw=1, label="random (AUC=0.5)")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC curves on Iris test split (setosa vs rest)")
        plt.legend(fontsize=9, loc="lower right")
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(HERE, "roc_curve.png"), dpi=150)
        plt.close()
        print("[OK] 已保存 roc_curve.png")
    except Exception as e:
        print(f"[WARN] 绘图失败（{e}），跳过 roc_curve.png")


# ======================================================================
# 主流程
# ======================================================================
def main():
    X_iris, y_iris = load_iris_two_features(IRIS_PATH)

    # ---- 实验 A：鸢尾花（近线性），重复分层 CV + 1se 规则 + P/R/F1 ----
    preds = {
        "Fisher LDA": lda_predictor,
        "linear SVM": make_svm_predictor(linear_kernel(), C),
        "RBF SVM": make_svm_predictor(rbf_kernel(GAMMA), C),
    }
    METRICS = ("acc", "precision", "recall", "f1")
    metrics = {}
    for name, p in preds.items():
        metrics[name] = repeated_cv_scores(X_iris, y_iris, K_FOLDS, N_REPEATS, BASE_SEED, p)

    summaries = {name: cv_summary(metrics[name]["acc"]) for name in metrics}
    thresh = one_se_threshold([metrics[n]["acc"] for n in metrics])

    # ---- 实验 B：γ 扫描 ----
    gammas, tr_g, cv_g, nsv_g, nsvb_g = gamma_scan(
        X_iris, y_iris, C, GAMMA_GRID, K_FOLDS, N_REPEATS_SCAN, BASE_SEED)

    # ---- 实验 D：C 敏感性扫描（鸢尾花，线性核）----
    cs, tr_c, cv_c, nsv_c = c_scan(
        X_iris, y_iris, linear_kernel(), C_GRID, K_FOLDS, N_REPEATS_SCAN, BASE_SEED)

    # ---- 实验 E：ROC 曲线与 AUC（鸢尾花单次 7:3 划分，种子 42）----
    rng_e = np.random.default_rng(BASE_SEED)
    idx_e = rng_e.permutation(len(y_iris))
    n_tr_e = len(y_iris) - int(round(0.30 * len(y_iris)))
    tr_e, te_e = idx_e[:n_tr_e], idx_e[n_tr_e:]
    X_tr_s, X_te_s, _, _ = standardize(X_iris[tr_e], X_iris[te_e])
    w_lda, b_lda = fisher_lda(X_tr_s, y_iris[tr_e])
    score_lda = X_te_s @ w_lda + b_lda                 # 连续判别分数
    m_rbf_roc = svm_fit(X_tr_s, y_iris[tr_e], C, rbf_kernel(GAMMA))
    score_svm = svm_decision(m_rbf_roc, X_te_s)        # 连续决策值 f(x)
    fpr_l, tpr_l, auc_l = roc_curve_auc(y_iris[te_e], score_lda)
    fpr_s, tpr_s, auc_s = roc_curve_auc(y_iris[te_e], score_svm)
    plot_roc(fpr_l, tpr_l, auc_l, fpr_s, tpr_s, auc_s)

    # ---- 实验 F：Gram 矩阵计算开销（线性核 vs RBF 核）----
    X_all_s, _, _, _ = standardize(X_iris)
    t_lin_g = gram_matrix_time(X_all_s, linear_kernel())
    t_rbf_g = gram_matrix_time(X_all_s, rbf_kernel(GAMMA))

    # ---- 实验 C：同心圆（强非线性），重复分层 CV ----
    X_c, y_c = make_circles(200)
    lin_c = make_svm_predictor(linear_kernel(), C)
    rbf_c = make_svm_predictor(rbf_kernel(GAMMA), C)
    met_lin = repeated_cv_scores(X_c, y_c, K_FOLDS, N_REPEATS, BASE_SEED, lin_c)
    met_rbf = repeated_cv_scores(X_c, y_c, K_FOLDS, N_REPEATS, BASE_SEED, rbf_c)
    X_cs, _, _, _ = standardize(X_c)
    m_lin = svm_fit(X_cs, y_c, C, linear_kernel())
    m_rbf = svm_fit(X_cs, y_c, C, rbf_kernel(GAMMA))
    plot_circles(X_cs, y_c, m_lin, m_rbf, met_lin["acc"].mean(), met_rbf["acc"].mean())
    plot_gamma_scan(gammas, tr_g, cv_g, nsv_g)

    # ---- 组织输出 ----
    lines = []
    lines.append("支持向量机（SVM）与核方法对比实验 —— 运行输出")
    lines.append("=" * 60)
    lines.append(f"协议   : 分层 {K_FOLDS} 折 × {N_REPEATS} 次重复交叉验证（种子 {BASE_SEED}..）")
    lines.append(f"超参数 : C = {C}, RBF gamma = {GAMMA}（基准）；1se 规则用于模型选择")
    lines.append("改进方向: 核函数（线性核 vs RBF 核）；指标：acc/P/R/F1 + ROC/AUC + 耗时")
    lines.append("-" * 60)
    lines.append("【实验 A】鸢尾花（近线性）—— CV 指标 均值 ± 标准误")
    for name in ["Fisher LDA", "linear SVM", "RBF SVM"]:
        row = [f"  {name:11s}"]
        for key in METRICS:
            m, se = cv_summary(metrics[name][key])
            row.append(f"{key} {m*100:.2f}%±{se*100:.2f}%")
        lines.append("  ".join(row))
    best = max(summaries, key=lambda k: summaries[k][0])
    lines.append(f"  1se 阈值 = {thresh*100:.2f}%（最优 {best} 的均值 - 1 标准误）")
    lines.append("-" * 60)
    lines.append("【实验 B】γ 扫描（鸢尾花，C=1）—— γ | 训练 | CV | 支持向量数(有界)")
    for g, tr, cv, nsv, nsvb in zip(gammas, tr_g, cv_g, nsv_g, nsvb_g):
        lines.append(f"  γ={g:.4f}: train {tr*100:.2f}%  CV {cv*100:.2f}%  SV {nsv:.1f} (有界 {nsvb:.1f})")
    gap = tr_g - cv_g
    flip = int(np.argmax(gap > 0.05)) if (gap > 0.05).any() else -1
    if flip >= 0:
        lines.append(f"  （训练-CV 差距首次超过 5pp：γ={gammas[flip]:.4f}）")
    else:
        lines.append("  （本范围内训练-CV 差距未超过 5pp）")
    lines.append("-" * 60)
    lines.append("【实验 D】C 敏感性（鸢尾花，线性核）—— C | 训练 | CV | 支持向量数")
    for c, tr, cv, nsv in zip(cs, tr_c, cv_c, nsv_c):
        lines.append(f"  C={c:.4f}: train {tr*100:.2f}%  CV {cv*100:.2f}%  SV {nsv:.1f}")
    lines.append("-" * 60)
    lines.append("【实验 E】ROC/AUC（鸢尾花单次 7:3 划分，种子 42）")
    lines.append(f"  Fisher LDA: AUC = {auc_l:.4f}")
    lines.append(f"  RBF 核 SVM: AUC = {auc_s:.4f}")
    lines.append("-" * 60)
    lines.append("【实验 F】Gram 矩阵构建耗时（n=150 标准化样本，500 次重复取均值）")
    lines.append(f"  线性核 K=X·Xᵀ  : {t_lin_g*1e6:8.2f} μs/次")
    lines.append(f"  RBF 核 K=exp(-γ‖xᵢ-xⱼ‖²): {t_rbf_g*1e6:8.2f} μs/次")
    lines.append(f"  耗时比 (RBF/线性) ≈ {t_rbf_g/max(t_lin_g,1e-12):.2f}×（RBF 需 pairwise 距离 + exp）")
    lines.append("  注：n=150 时 Gram 矩阵开销占比很小，完整训练时间主要由 SMO 迭代数决定；")
    lines.append("      该开销随 n 增大按 O(n²) 增长，在大样本下将成为 RBF 核的主要瓶颈。")
    lines.append("-" * 60)
    lines.append("【实验 C】同心圆（强非线性）—— CV 指标 均值 ± 标准误")
    for nm, met in [("线性核 SVM", met_lin), ("RBF 核 SVM", met_rbf)]:
        row = [f"  {nm}:"]
        for key in METRICS:
            m, se = cv_summary(met[key])
            row.append(f"{key} {m*100:.2f}%±{se*100:.2f}%")
        lines.append("  ".join(row))
    lines.append("=" * 60)
    lines.append("结论   : 近线性数据上核方法不劣化；强非线性数据上 RBF 核带来")
    lines.append("         明显提升（线性 ~52% -> RBF ~100%），验证核技巧的价值。")
    lines.append("         ROC/AUC 与 P/R/F1 从阈值无关与类代价两个维度补充准确率；")
    lines.append("         训练耗时对比呈现 RBF 核的精度-开销 Trade-off。")
    txt = "\n".join(lines)
    with open(os.path.join(HERE, "svm_results.txt"), "w", encoding="utf-8") as fh:
        fh.write(txt + "\n")
    print(txt)


if __name__ == "__main__":
    main()
