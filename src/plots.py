"""Shared chart style and helpers, so every figure in the project looks consistent."""
import matplotlib.pyplot as plt
import pandas as pd

from src.config import BLUE, FIGURES_DIR, ORANGE


def setup_style() -> None:
    plt.rcParams.update({
        "figure.figsize": (8, 4.5),
        "figure.dpi": 110,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "axes.titleweight": "bold",
        "axes.titlesize": 12,
        "font.size": 10,
    })


def save(fig, name: str) -> None:
    """Save to reports/figures/<name>.png (used by the README and the business report)."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES_DIR / f"{name}.png", bbox_inches="tight", dpi=130)


def churn_rate_bars(df: pd.DataFrame, column: str, target: str = "churn", ax=None, overall: float | None = None):
    """Horizontal bars of churn rate (%) for each value of `column`, labelled with the rate and group size."""
    ax = ax or plt.gca()
    rates = df.groupby(column)[target].agg(["mean", "size"]).sort_values("mean")
    bars = ax.barh(rates.index.astype(str), rates["mean"] * 100, color=ORANGE, height=0.6)
    for bar, (rate, size) in zip(bars, rates.itertuples(index=False)):
        ax.text(bar.get_width() + 0.8, bar.get_y() + bar.get_height() / 2,
                f"{rate * 100:.1f}%  (n={size:,})", va="center", fontsize=9)
    if overall is not None:
        ax.axvline(overall * 100, color=BLUE, linestyle="--", linewidth=1.2, label=f"overall {overall * 100:.1f}%")
        ax.legend(loc="lower right", frameon=False)
    ax.set_xlabel("Churn rate (%)")
    ax.set_xlim(0, max(rates["mean"] * 100) * 1.45)
    ax.set_title(column.replace("_", " ").capitalize())
    ax.grid(axis="y", visible=False)
    return ax
