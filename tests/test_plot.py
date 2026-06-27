"""Tests for the examples/plot_history.py convergence-plot script.

Uses real csv data and the non-interactive Agg backend; no mocks.
"""

import csv
import importlib.util
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

EXAMPLES = Path(__file__).resolve().parent.parent / "examples"


def _load_plot_module():
    spec = importlib.util.spec_from_file_location("plot_history", EXAMPLES / "plot_history.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_history(path: Path) -> None:
    rows = [
        (0, 0.90, 0.85, 0.80, 0.90),
        (1, 0.92, 0.89, 0.86, 0.92),
        (2, 0.91, 0.90, 0.88, 0.92),
        (3, 0.94, 0.92, 0.90, 0.94),
    ]
    with open(path, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(
            ["generation", "best_score", "mean_score", "worst_score", "global_best_score"]
        )
        writer.writerows(rows)


def test_plot_history_creates_png_next_to_csv(tmp_path):
    mod = _load_plot_module()
    csv_path = tmp_path / "history.csv"
    _write_history(csv_path)

    out = mod.plot_history(csv_path)

    assert out == csv_path.with_suffix(".png")
    assert out.exists()
    assert out.stat().st_size > 0


def test_plot_history_respects_explicit_out_path(tmp_path):
    mod = _load_plot_module()
    csv_path = tmp_path / "history.csv"
    _write_history(csv_path)
    out_path = tmp_path / "nested" / "plot.png"

    out = mod.plot_history(csv_path, out_path)

    assert out == out_path
    assert out.exists()
    assert out.stat().st_size > 0


def test_main_prints_saved_path(tmp_path, capsys):
    mod = _load_plot_module()
    csv_path = tmp_path / "history.csv"
    _write_history(csv_path)
    out_path = tmp_path / "out.png"

    mod.main([str(csv_path), str(out_path)])

    captured = capsys.readouterr()
    assert str(out_path) in captured.out
    assert out_path.exists()
    assert out_path.stat().st_size > 0
