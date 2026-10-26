"""그림: 예측기간 × 입력 윈도우 히트맵(논문 Table III–IX 형식)과 walk-forward 누적수익률.

히트맵 색은 같은 평가구간 Buy & Hold 대비 차이(파랑 = 더 좋음, 빨강 = 더 나쁨, 회색 = 같음)이고, 칸 안 숫자는
지표 값 그대로다. 굵은 테두리 칸이 논문 결론의 n ≈ s 다. 한글 글꼴이 없는 환경도 있어 그림 글자는 영어로 쓴다.
"""

from __future__ import annotations

from typing import Dict, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402
from matplotlib.ticker import FuncFormatter  # noqa: E402

from .experiment import matched_window  # noqa: E402

SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
SERIES = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#8a5cd6", "#d6457f", "#1a9cb0", "#8c6a2b")
DIVERGING = LinearSegmentedColormap.from_list(
    "vs_buy_hold", ["#a8302f", "#e34948", "#f3b3b2", "#f0efec", "#9ec5f4", "#2a78d6", "#184f95"])

METRIC_LABEL = {"accuracy": ("Accuracy (%)", 100, "{:.1f}"), "win_rate": ("Winning rate (%)", 100, "{:.1f}"),
                "ret_per_trade": ("Return per trade (%)", 100, "{:.2f}"), "sharpe": ("Sharpe ratio", 1, "{:.2f}")}
MODEL_LABEL = {"svm": "SVM", "ann": "ANN", "knn": "kNN", "rf": "Random forest", "xgb": "XGBoost",
               "lgbm": "LightGBM", "logit": "Logit", "arima": "ARIMA"}
HEATMAP_COLS = 4  # 모델이 더 많으면 다음 줄로


def _style(ax):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(INK2)
    ax.tick_params(colors=INK2, labelsize=9)


def grid_heatmaps(metrics: pd.DataFrame, scenario: str, n_classes: int, metric: str, path: str,
                  models: Sequence[str]) -> None:
    label, scale, fmt = METRIC_LABEL[metric]
    sub = metrics[(metrics["scenario"] == scenario) & (metrics["n_classes"] == n_classes)]
    bh = sub[sub["model"] == "buy_hold"].set_index("horizon")[metric]
    tables = {m: sub[sub["model"] == m].pivot(index="horizon", columns="window", values=metric) for m in models}
    tables = {m: t for m, t in tables.items() if not t.empty}
    if not tables:
        return
    diffs = np.concatenate([np.abs(t.sub(bh, axis=0).to_numpy()).ravel() for t in tables.values()])
    diffs = diffs[np.isfinite(diffs)]  # 거래가 없는 칸(승률 · Sharpe 없음)은 빼고 색 범위를 잡는다
    lim = (float(diffs.max()) * scale if len(diffs) else 0.0) or 1.0

    ncols = min(len(tables), HEATMAP_COLS)
    nrows = -(-len(tables) // ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.9 * ncols + 0.8, 4.8 * nrows), squeeze=False,
                             facecolor=SURFACE, constrained_layout=True)
    for ax in axes.ravel()[len(tables):]:
        ax.set_visible(False)
    for ax, (m, t) in zip(axes.ravel(), tables.items()):
        diff = t.sub(bh, axis=0).to_numpy() * scale
        im = ax.imshow(diff, cmap=DIVERGING, vmin=-lim, vmax=lim, aspect="auto")
        for i, s in enumerate(t.index):
            for j, n in enumerate(t.columns):
                v = t.iloc[i, j] * scale
                ink = SURFACE if abs(diff[i, j]) > 0.62 * lim else INK
                ax.text(j, i, fmt.format(v), ha="center", va="center", fontsize=7.5, color=ink)
            j = list(t.columns).index(matched_window(int(s), list(t.columns)))
            ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, edgecolor=INK, linewidth=1.6))
        ax.set_xticks(range(len(t.columns)), [str(int(c)) for c in t.columns])
        ax.set_yticks(range(len(t.index)),
                      [f"{int(s)}  (B&H {fmt.format(bh.get(s, np.nan) * scale)})" for s in t.index])
        ax.set_xlabel("Input window length n (days)", color=INK2, fontsize=9)
        ax.set_ylabel("Forecast horizon s (days)", color=INK2, fontsize=9)
        ax.set_title(MODEL_LABEL.get(m, m), color=INK, fontsize=11, loc="left")
        _style(ax)
    cbar = fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.85, pad=0.01)
    cbar.set_label(f"{label}: strategy minus buy & hold", color=INK2, fontsize=9)
    cbar.ax.tick_params(colors=INK2, labelsize=8)
    fig.suptitle(f"{label} — {scenario}, {n_classes}-class (outlined: n ≈ s)", color=INK, fontsize=12, x=0.01,
                 ha="left")
    fig.savefig(path, dpi=130, facecolor=SURFACE)
    plt.close(fig)


def equity_curves(curves: Dict[str, pd.Series], bench: pd.Series, title: str, path: str) -> None:
    """누적 자산(1 에서 시작). 전략은 범주색 순서대로, Buy & Hold 는 회색 기준선."""
    fig, ax = plt.subplots(figsize=(10, 4.8), facecolor=SURFACE, constrained_layout=True)
    ax.plot(bench.index, (1 + bench).cumprod(), color=INK2, linewidth=1.6, label="Buy & hold")
    for (name, ret), color in zip(curves.items(), SERIES):
        wealth = (1 + ret).cumprod()
        ax.plot(wealth.index, wealth, color=color, linewidth=2.0, label=name)
        ax.annotate(name, (wealth.index[-1], wealth.iloc[-1]), xytext=(4, 0), textcoords="offset points",
                    color=INK, fontsize=8, va="center")
    ax.axhline(1.0, color=GRID, linewidth=1.0, zorder=0)
    ax.set_yscale("log")
    plain = FuncFormatter(lambda v, _: f"{v:g}")
    ax.yaxis.set_major_formatter(plain)
    ax.yaxis.set_minor_formatter(plain)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_ylabel("Wealth (log scale, start = 1)", color=INK2, fontsize=9)
    ax.set_title(title, color=INK, fontsize=11, loc="left")
    ax.legend(frameon=False, fontsize=8, labelcolor=INK, loc="upper left")
    _style(ax)
    fig.savefig(path, dpi=130, facecolor=SURFACE)
    plt.close(fig)
