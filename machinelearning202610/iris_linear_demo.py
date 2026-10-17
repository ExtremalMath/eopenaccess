# -*- coding: utf-8 -*-
"""
鸢尾花线性二分类示例（核心仅依赖 numpy；绘图可选 matplotlib）。
=====================================================================
数据：UCI ML Repository 的 Iris 数据集（Fisher, 1936），本地文件 iris.data。
任务：以前两个特征（萼片长、萼片宽）做 setosa(=1) 与其余类(=0) 的二分类。
模型：Fisher 线性判别（LDA，闭式解）——一种线性分类器。
运行：python iris_linear_demo.py
输出：① 控制台摘要
      ② iris_linear_results.txt（说明文字 + 全部数值，便于查阅/保存）
      ③ 若能导入 matplotlib，另存 iris_boundary.png（决策边界图）；
         否则退而生成 iris_figure.tex（TikZ 片段，供 LaTeX 使用）。
本脚本不依赖 LaTeX 即可独立运行。
"""
import os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(HERE, "iris.data")


def load_iris_two_features(path):
    """读取 UCI iris.data，仅取前两列（萼片长、萼片宽），并将 setosa 标为 1。"""
    X_all, raw = [], []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            p = line.split(",")
            X_all.append([float(p[0]), float(p[1])])      # 仅取前两个特征
            raw.append(0 if p[4] == "Iris-setosa" else 1)  # 临时：setosa=0
    X = np.array(X_all, dtype=float)
    y = np.array([1 if r == 0 else 0 for r in raw], dtype=int)  # 统一：setosa=1
    return X, y


def fisher_lda(X_tr, y_tr):
    """Fisher 线性判别（闭式解），返回 (w, b)：判别函数 g(x)=w^T x + b。"""
    mu1 = X_tr[y_tr == 1].mean(axis=0)   # setosa 均值
    mu0 = X_tr[y_tr == 0].mean(axis=0)   # 其余类均值
    Sw = np.zeros((2, 2))
    for x in X_tr[y_tr == 1]:
        d = (x - mu1).reshape(2, 1)
        Sw += d @ d.T
    for x in X_tr[y_tr == 0]:
        d = (x - mu0).reshape(2, 1)
        Sw += d @ d.T
    w = np.linalg.inv(Sw) @ (mu1 - mu0)
    mid = 0.5 * (mu1 + mu0)
    b = -float(w @ mid)
    return w, b


def predict(X, w, b):
    return (X @ w + b > 0).astype(int)


def main():
    # ---------- 1. 读取数据 ----------
    X, y = load_iris_two_features(DATA_PATH)
    n = len(X)
    n_pos = int(y.sum())          # setosa
    n_neg = n - n_pos             # 其余两类

    # ---------- 2. 确定性划分（固定种子，可复现）----------
    rng = np.random.default_rng(42)
    idx = rng.permutation(n)
    n_tr = int(0.7 * n)
    tr, te = idx[:n_tr], idx[n_tr:]
    X_tr, y_tr = X[tr], y[tr]
    X_te, y_te = X[te], y[te]

    # ---------- 3. 训练线性分类器（LDA）----------
    w, b = fisher_lda(X_tr, y_tr)

    # ---------- 4. 评估 ----------
    acc_tr = float((predict(X_tr, w, b) == y_tr).mean())
    acc_te = float((predict(X_te, w, b) == y_te).mean())

    # ---------- 5. 组织说明并保存 txt ----------
    lines = []
    lines.append("鸢尾花线性二分类示例 —— 运行输出")
    lines.append("=" * 52)
    lines.append("数据来源 : UCI ML Repository, Iris Data Set (Fisher, 1936)")
    lines.append("          本地文件 iris.data（亦可由 sklearn.load_iris() 获取，同源）")
    lines.append(f"样本总数 : n = {n}（setosa={n_pos}，其余={n_neg}）")
    lines.append("特征     : 第1列=萼片长(cm)，第2列=萼片宽(cm)；仅用前两维")
    lines.append("标签     : setosa -> y=1，其余两类合并 -> y=0")
    lines.append(f"划分     : 训练 {len(X_tr)} / 测试 {len(X_te)}（比例 7:3，种子=42）")
    lines.append("模型     : Fisher 线性判别 LDA（闭式解，线性方法，无迭代）")
    lines.append("          g(x) = w^T x + b，当 g(x)>0 判为 setosa")
    lines.append("          w = Sw^{-1}(mu1 - mu0),  b = -w^T * (mu1+mu0)/2")
    lines.append("-" * 52)
    lines.append(f"判别系数 : w = [{w[0]:.4f}, {w[1]:.4f}],  b = {b:.4f}")
    lines.append(f"决策边界 : {w[0]:.4f}*x1 + {w[1]:.4f}*x2 + {b:.4f} = 0")
    lines.append(f"          即 x2 ≈ {(-w[0]/w[1]):.4f}*x1 + {(-b/w[1]):.4f}")
    lines.append(f"训练准确率: {acc_tr*100:.2f}%")
    lines.append(f"测试准确率: {acc_te*100:.2f}%")
    lines.append("-" * 52)
    lines.append("解读     : setosa 在前两维上“近似”而非“完全”线性可分，")
    lines.append("          故准确率很高（≈98%）但未达 100%；这源于数据特性，")
    lines.append("          不表示线性模型万能。若数据呈明显非线性边界，应")
    lines.append("          改用核方法或更复杂模型（见教材第8章 SVM）。")
    lines.append("=" * 52)
    txt = "\n".join(lines)
    out_txt = os.path.join(HERE, "iris_linear_results.txt")
    with open(out_txt, "w", encoding="utf-8") as fh:
        fh.write(txt + "\n")
    print(txt)

    # ---------- 6. 绘图（matplotlib 优先；否则退化为 TikZ）----------
    xmin, xmax = float(X[:, 0].min() - 0.3), float(X[:, 0].max() + 0.3)
    ymin, ymax = float(X[:, 1].min() - 0.3), float(X[:, 1].max() + 0.3)
    x2a = -(w[0] * xmin + b) / w[1]
    x2b = -(w[0] * xmax + b) / w[1]

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(5.4, 4.3))
        ax.scatter(X[y == 1, 0], X[y == 1, 1], c="#d62728", s=24,
                   label="setosa (y=1)", alpha=.85)
        ax.scatter(X[y == 0, 0], X[y == 0, 1], c="#1f77b4", s=24,
                   label="others (y=0)", alpha=.85)
        ax.plot([xmin, xmax], [x2a, x2b], "k-", lw=2, label="decision boundary")
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
    except Exception:
        # 任意绘图异常（含未安装 matplotlib）均不影响核心产出，退化为 TikZ 供 LaTeX 使用
        # 注意：f-string 内需要原样输出的花括号必须写成双花括号 {{ }} 进行转义
        def coords(mask):
            return " ".join(f"({a:.2f},{c:.2f})" for a, c in X[mask])
        tikz = f"""\\begin{{tikzpicture}}
\\begin{{axis}}[
  width=9cm, height=7cm,
  xlabel={{sepal length (cm)}}, ylabel={{sepal width (cm)}},
  xmin={xmin:.2f}, xmax={xmax:.2f}, ymin={ymin:.2f}, ymax={ymax:.2f},
  legend pos=north west, grid=major, grid style={{lightgray!30, dashed}},
  tick label style=/pgf/number format/fixed
]
  \\addplot[only marks, mark=*, red, mark size=2.2pt] coordinates {{ {coords(y == 1)} }};
  \\addplot[only marks, mark=*, blue, mark size=2.2pt] coordinates {{ {coords(y == 0)} }};
  \\addplot[no marks, thick, black] coordinates {{ ({xmin:.2f},{x2a:.2f}) ({xmax:.2f},{x2b:.2f}) }};
  \\legend{{setosa (y=1), others (y=0), decision boundary}}
\\end{{axis}}
\\end{{tikzpicture}}
"""
        tex_path = os.path.join(HERE, "iris_figure.tex")
        with open(tex_path, "w", encoding="utf-8") as fh:
            fh.write(tikz)
        print(f"[WARN] 绘图失败（matplotlib 缺失或出错），已退化为 TikZ -> {os.path.basename(tex_path)}")


if __name__ == "__main__":
    main()
