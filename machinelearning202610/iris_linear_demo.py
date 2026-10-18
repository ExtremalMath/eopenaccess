# -*- coding: utf-8 -*-
"""鸢尾花线性二分类示例：setosa 与其余类（Fisher 线性判别，闭式解）。

数据 : UCI ML Repository 的 Iris 数据集（Fisher, 1936），本地文件 iris.data。
任务 : 以前两个特征（萼片长、萼片宽）做 setosa(=1) 与其余类(=0) 的二分类。
模型 : Fisher 线性判别（LDA，闭式解）——纯线性方法，无迭代优化。
运行 : python iris_linear_demo.py
输出 : ① 控制台摘要
       ② iris_linear_results.txt（说明文字 + 全部数值 + 混淆矩阵）
       ③ 若能导入 matplotlib -> iris_boundary.png；否则 -> iris_figure.tex

依赖 : 仅必需 NumPy；matplotlib 可选（仅用于绘图）。
       本脚本不依赖 LaTeX，可独立运行。
"""
from __future__ import annotations

import os
from typing import Tuple

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(HERE, "iris.data")
RANDOM_SEED = 42
TEST_RATIO = 0.30
EPS = 1e-8  # 类内散布矩阵的正则项，防止其近奇异时求解失败

# 标签约定：setosa 为正类(y=1)，其余两类合并为负类(y=0)
POSITIVE = 1


def load_iris_two_features(path: str) -> Tuple[np.ndarray, np.ndarray]:
    """读取 UCI iris.data，仅取前两列（萼片长、萼片宽），setosa 标为 1。"""
    X_rows: list[list[float]] = []
    y_rows: list[int] = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:  # 跳过空行
                continue
            cols = line.split(",")
            X_rows.append([float(cols[0]), float(cols[1])])       # 前两个特征
            y_rows.append(1 if cols[4] == "Iris-setosa" else 0)   # setosa=1，其余=0
    return np.asarray(X_rows, dtype=float), np.asarray(y_rows, dtype=int)


def split_train_test(
    X: np.ndarray, y: np.ndarray, test_ratio: float, seed: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """按固定种子随机置换后切分，返回 (X_tr, y_tr, X_te, y_te)，保证可复现。"""
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(y))
    # 前 n_tr 个作为训练集，后 n_test 个作为测试集（与原始实现一致，保证数值可复现）
    n_test = int(round(test_ratio * len(y)))
    n_tr = len(y) - n_test
    tr, te = idx[:n_tr], idx[n_tr:]
    return X[tr], y[tr], X[te], y[te]


def fisher_lda(X_tr: np.ndarray, y_tr: np.ndarray) -> Tuple[np.ndarray, float]:
    """Fisher 线性判别（闭式解），返回 (w, b)：g(x) = w^T x + b > 0 判为正类。

    w = Sw^{-1}(mu1 - mu0),  b = -w^T (mu1 + mu0)/2。
    类内散布矩阵 Sw 采用向量化计算，并用 solve 替代显式求逆，数值更稳定。
    """
    X1 = X_tr[y_tr == POSITIVE]
    X0 = X_tr[y_tr != POSITIVE]
    mu1, mu0 = X1.mean(axis=0), X0.mean(axis=0)

    # Sw = sum_k sum_{i in C_k} (x_i - mu_k)(x_i - mu_k)^T（向量化）
    Sw = (X1 - mu1).T @ (X1 - mu1) + (X0 - mu0).T @ (X0 - mu0)
    Sw = Sw + EPS * np.eye(Sw.shape[0])  # 微小正则，防 Sw 奇异

    w = np.linalg.solve(Sw, mu1 - mu0)
    b = -float(w @ ((mu1 + mu0) / 2.0))
    return w, b


def predict(X: np.ndarray, w: np.ndarray, b: float) -> np.ndarray:
    """按符号判定：g(x) = w^T x + b > 0 -> 正类(1)，否则负类(0)。"""
    return (X @ w + b > 0).astype(int)


def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """准确率：预测正确的样本占比。"""
    return float((y_true == y_pred).mean())


def confusion_matrix(
    y_true: np.ndarray, y_pred: np.ndarray
) -> Tuple[int, int, int, int]:
    """返回 (TN, FP, FN, TP)，便于定位误分类样本。"""
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    return tn, fp, fn, tp


def boundary_explicit(w: np.ndarray, b: float) -> str:
    """决策边界 w^T x + b = 0 化为 x2 关于 x1 的显式式，并正确处理正负号。"""
    slope = -w[0] / w[1]
    intercept = -b / w[1]
    sign = "-" if intercept < 0 else "+"
    return f"x2 = {slope:.4f} * x1 {sign} {abs(intercept):.4f}"


def build_report_text(
    n: int, n_pos: int, n_neg: int, n_tr: int, n_te: int,
    w: np.ndarray, b: float, acc_tr: float, acc_te: float,
    cm_tr: Tuple[int, int, int, int], cm_te: Tuple[int, int, int, int],
) -> str:
    """组织“说明文字 + 数值”的文本报告内容。"""
    line = []
    line.append("鸢尾花线性二分类示例 —— 运行输出")
    line.append("=" * 52)
    line.append("数据来源 : UCI ML Repository, Iris Data Set (Fisher, 1936)")
    line.append("          本地文件 iris.data（亦可由 sklearn.load_iris() 获取，同源）")
    line.append(f"样本总数 : n = {n}（setosa={n_pos}，其余={n_neg}）")
    line.append("特征     : 第1列=萼片长(cm)，第2列=萼片宽(cm)；仅用前两维")
    line.append("标签     : setosa -> y=1，其余两类合并 -> y=0")
    line.append(f"划分     : 训练 {n_tr} / 测试 {n_te}（比例 7:3，种子={RANDOM_SEED}）")
    line.append("模型     : Fisher 线性判别 LDA（闭式解，线性方法，无迭代）")
    line.append("          g(x) = w^T x + b，当 g(x)>0 判为 setosa")
    line.append("          w = Sw^{-1}(mu1 - mu0),  b = -w^T * (mu1+mu0)/2")
    line.append("-" * 52)
    line.append(f"判别系数 : w = [{w[0]:.4f}, {w[1]:.4f}],  b = {b:.4f}")
    line.append(f"决策边界 : {w[0]:.4f}*x1 + {w[1]:.4f}*x2 + {b:.4f} = 0")
    line.append(f"          即 {boundary_explicit(w, b)}")
    line.append(f"训练准确率: {acc_tr * 100:.2f}%")
    line.append(f"测试准确率: {acc_te * 100:.2f}%")
    line.append("-" * 52)
    line.append("混淆矩阵（行=真实，列=预测，顺序 [负类 正类]）:")
    line.append(f"  训练集  TN={cm_tr[0]}  FP={cm_tr[1]}  FN={cm_tr[2]}  TP={cm_tr[3]}")
    line.append(f"  测试集  TN={cm_te[0]}  FP={cm_te[1]}  FN={cm_te[2]}  TP={cm_te[3]}")
    line.append("-" * 52)
    line.append("解读     : setosa 在前两维上“近似”而非“完全”线性可分，")
    line.append("          故准确率很高（≈98%）但未达 100%；这源于数据特性，")
    line.append("          不表示线性模型万能。若数据呈明显非线性边界，应")
    line.append("          改用核方法或更复杂模型（见教材第8章 SVM）。")
    line.append("=" * 52)
    return "\n".join(line)


def plot_boundary(X: np.ndarray, y: np.ndarray, w: np.ndarray, b: float) -> bool:
    """用 matplotlib 绘制散点与决策边界，成功返回 True；否则抛出异常由调用方兜底。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    xmin = float(X[:, 0].min() - 0.3)
    xmax = float(X[:, 0].max() + 0.3)
    ymin = float(X[:, 1].min() - 0.3)
    ymax = float(X[:, 1].max() + 0.3)
    # 边界直线两端点（假设 w[1] != 0；若 w[1]=0 则边界为竖线，见 export_tikz 的守卫）
    x2_a = -(w[0] * xmin + b) / w[1]
    x2_b = -(w[0] * xmax + b) / w[1]

    fig, ax = plt.subplots(figsize=(5.4, 4.3))
    ax.scatter(X[y == 1, 0], X[y == 1, 1], c="#d62728", s=24,
               label="setosa (y=1)", alpha=0.85)
    ax.scatter(X[y == 0, 0], X[y == 0, 1], c="#1f77b4", s=24,
               label="others (y=0)", alpha=0.85)
    ax.plot([xmin, xmax], [x2_a, x2_b], "k-", lw=2, label="decision boundary")
    ax.set_xlabel("sepal length (cm)")
    ax.set_ylabel("sepal width (cm)")
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.legend(fontsize=8)
    ax.set_title("Fisher LDA: setosa vs rest (2 features)")
    fig.tight_layout()
    png_path = os.path.join(HERE, "iris_boundary.png")
    fig.savefig(png_path, dpi=150)
    plt.close(fig)
    print(f"[OK] 已保存决策边界图 -> {os.path.basename(png_path)}")
    return True


def export_tikz(X: np.ndarray, y: np.ndarray, w: np.ndarray, b: float) -> None:
    """无 matplotlib 时的 TikZ 兜底：生成供 LaTeX 使用的散点/边界片段。"""
    xmin = float(X[:, 0].min() - 0.3)
    xmax = float(X[:, 0].max() + 0.3)
    ymin = float(X[:, 1].min() - 0.3)
    ymax = float(X[:, 1].max() + 0.3)

    # 若 w[1] 接近 0，边界近似竖线，改用两点垂直线坐标以避免除零
    if abs(w[1]) < 1e-12:
        x2_a, x2_b = ymin, ymax
        x1_fixed = -b / w[0]
        line_coords = f"({x1_fixed:.2f},{x2_a:.2f}) ({x1_fixed:.2f},{x2_b:.2f})"
    else:
        x2_a = -(w[0] * xmin + b) / w[1]
        x2_b = -(w[0] * xmax + b) / w[1]
        line_coords = f"({xmin:.2f},{x2_a:.2f}) ({xmax:.2f},{x2_b:.2f})"

    def coords(mask):
        return " ".join(f"({a:.2f},{c:.2f})" for a, c in X[mask])

    # 注意：f-string 中需原样输出的花括号必须双写 {{ }} 转义
    tikz = (
        "\\begin{tikzpicture}\n"
        "\\begin{axis}[\n"
        "  width=9cm, height=7cm,\n"
        "  xlabel={sepal length (cm)}, ylabel={sepal width (cm)},\n"
        f"  xmin={xmin:.2f}, xmax={xmax:.2f}, ymin={ymin:.2f}, ymax={ymax:.2f},\n"
        "  legend pos=north west, grid=major, grid style={lightgray!30, dashed},\n"
        "  tick label style=/pgf/number format/fixed\n"
        "]\n"
        f"  \\addplot[only marks, mark=*, red, mark size=2.2pt] coordinates {{ {coords(y == 1)} }};\n"
        f"  \\addplot[only marks, mark=*, blue, mark size=2.2pt] coordinates {{ {coords(y == 0)} }};\n"
        f"  \\addplot[no marks, thick, black] coordinates {{ {line_coords} }};\n"
        "  \\legend{setosa (y=1), others (y=0), decision boundary}\n"
        "\\end{axis}\n"
        "\\end{tikzpicture}\n"
    )
    tex_path = os.path.join(HERE, "iris_figure.tex")
    with open(tex_path, "w", encoding="utf-8") as fh:
        fh.write(tikz)
    print(f"[WARN] 绘图失败（matplotlib 缺失或出错），已退化为 TikZ -> {os.path.basename(tex_path)}")


def main() -> None:
    # 1. 读取数据
    X, y = load_iris_two_features(DATA_PATH)
    n = len(X)
    n_pos = int(y.sum())          # setosa
    n_neg = n - n_pos             # 其余两类

    # 2. 确定性划分（固定种子，可复现）
    X_tr, y_tr, X_te, y_te = split_train_test(X, y, TEST_RATIO, RANDOM_SEED)

    # 3. 训练线性分类器（仅用训练集，避免数据泄露）
    w, b = fisher_lda(X_tr, y_tr)

    # 4. 评估
    y_tr_pred = predict(X_tr, w, b)
    y_te_pred = predict(X_te, w, b)
    acc_tr = accuracy(y_tr, y_tr_pred)
    acc_te = accuracy(y_te, y_te_pred)
    cm_tr = confusion_matrix(y_tr, y_tr_pred)
    cm_te = confusion_matrix(y_te, y_te_pred)

    # 5. 组织说明并保存 txt
    report = build_report_text(
        n, n_pos, n_neg, len(X_tr), len(X_te),
        w, b, acc_tr, acc_te, cm_tr, cm_te,
    )
    out_txt = os.path.join(HERE, "iris_linear_results.txt")
    with open(out_txt, "w", encoding="utf-8") as fh:
        fh.write(report + "\n")
    print(report)

    # 6. 绘图（matplotlib 优先；否则退化为 TikZ）
    try:
        plot_boundary(X, y, w, b)
    except Exception:
        export_tikz(X, y, w, b)


if __name__ == "__main__":
    main()
