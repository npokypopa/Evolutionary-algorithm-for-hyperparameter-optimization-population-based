"""Plot a PBT convergence trace from a ``history.csv`` file.

Reads the per-generation history written by ``save_results`` and renders the
best / mean / worst / global-best scores against generation as labeled lines.

Run with:
    uv run python examples/plot_history.py [history.csv] [out.png]
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # non-interactive backend: never blocks

import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

DEFAULT_CSV = Path("results/history.csv")

_SERIES = [
    ("best_score", "best score"),
    ("mean_score", "mean score"),
    ("worst_score", "worst score"),
    ("global_best_score", "global best (running)"),
]


def plot_history(csv_path: str | Path, out_path: str | Path | None = None) -> Path:
    """Plot a convergence trace and save it as a PNG.

    Reads ``csv_path`` (the columns written by ``save_results``) and draws each
    score series against ``generation``. If ``out_path`` is not given, the PNG is
    written next to the csv with a ``.png`` suffix. Returns the saved path.
    """
    csv_path = Path(csv_path)
    out = Path(out_path) if out_path is not None else csv_path.with_suffix(".png")

    df = pd.read_csv(csv_path)

    fig, ax = plt.subplots(figsize=(8, 5))
    for column, label in _SERIES:
        ax.plot(df["generation"], df[column], marker="o", label=label)

    ax.set_xlabel("generation")
    ax.set_ylabel("score")
    ax.set_title("PBT convergence")
    ax.legend()
    fig.tight_layout()

    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=100)
    plt.close(fig)
    return out


def main(argv: list[str] | None = None) -> None:
    args = sys.argv[1:] if argv is None else argv
    csv_path = Path(args[0]) if len(args) >= 1 else DEFAULT_CSV
    out_path = Path(args[1]) if len(args) >= 2 else None
    saved = plot_history(csv_path, out_path)
    print(f"saved plot to {saved}")


if __name__ == "__main__":
    main()
