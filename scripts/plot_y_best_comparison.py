#!/usr/bin/env python3
import argparse
import csv
import os
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot y_best comparison curves from local metrics.csv files."
    )
    parser.add_argument("--task", default="earth_124", help="Task name under results/.")
    parser.add_argument("--algorithm", default="RLEA", help="Algorithm name under results/<task>/.")
    parser.add_argument(
        "--runs",
        nargs="*",
        default=None,
        help="Run timestamps or full run directories. If omitted, all runs with metrics.csv are used.",
    )
    parser.add_argument(
        "--labels",
        nargs="*",
        default=None,
        help="Labels for the selected runs. Must match --runs length if provided.",
    )
    parser.add_argument(
        "--x-axis",
        choices=["epoch", "evaluations"],
        default="epoch",
        help="Use epoch or total_evaluations as the x axis.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output image path. Defaults to results/<task>/<algorithm>/comparison_y_best.png.",
    )
    parser.add_argument(
        "--summary-output",
        default=None,
        help="Output summary csv path. Defaults to the output image path with .csv suffix.",
    )
    parser.add_argument("--title", default=None, help="Plot title. Defaults to '<task> / <algorithm>'.")
    return parser.parse_args()


def resolve_run_dirs(base_dir, runs):
    if runs:
        run_dirs = []
        for run in runs:
            path = Path(run)
            if not path.is_absolute() and not path.exists():
                path = base_dir / run
            run_dirs.append(path)
        return run_dirs

    if not base_dir.exists():
        return []
    return sorted(path for path in base_dir.iterdir() if (path / "metrics.csv").exists())


def read_metrics(metrics_path, x_axis):
    x_key = "epoch" if x_axis == "epoch" else "total_evaluations"
    xs = []
    ys = []
    archives = []
    local_evals = []
    with open(metrics_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            y_value = row.get("y_best", "")
            x_value = row.get(x_key, "")
            if y_value == "" or x_value == "":
                continue
            xs.append(float(x_value))
            ys.append(float(y_value))
            archives.append(float(row["archive_size"]) if row.get("archive_size", "") != "" else None)
            local_evals.append(
                float(row["local_search_evaluations"])
                if row.get("local_search_evaluations", "") != ""
                else 0.0
            )
    return xs, ys, archives, local_evals


def write_summary(summary_path, summaries):
    fieldnames = [
        "label",
        "run_dir",
        "points",
        "final_epoch",
        "final_total_evaluations",
        "final_local_search_evaluations",
        "final_archive_size",
        "final_y_best",
        "best_y_best",
    ]
    with open(summary_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summaries)


def main():
    args = parse_args()
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    base_dir = Path("../results") / args.task / f"{args.algorithm}"
    run_dirs = resolve_run_dirs(base_dir, args.runs)
    if not run_dirs:
        raise FileNotFoundError(f"No metrics.csv found under {base_dir}")

    if args.labels is not None and len(args.labels) != len(run_dirs):
        raise ValueError("--labels length must match the number of selected runs")

    output_path = Path(args.output) if args.output else base_dir / "comparison_y_best.png"
    summary_path = Path(args.summary_output) if args.summary_output else output_path.with_suffix(".csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(9, 5))
    summaries = []
    for idx, run_dir in enumerate(run_dirs):
        metrics_path = run_dir / "metrics.csv"
        if not metrics_path.exists():
            raise FileNotFoundError(f"Missing metrics.csv: {metrics_path}")

        label = args.labels[idx] if args.labels else run_dir.name
        xs, ys, archives, local_evals = read_metrics(metrics_path, args.x_axis)
        if not xs:
            continue

        plt.plot(xs, ys, linewidth=1.8, label=label)

        with open(metrics_path, newline="") as f:
            rows = [row for row in csv.DictReader(f) if row.get("y_best", "") != ""]
        last = rows[-1]
        summaries.append(
            {
                "label": label,
                "run_dir": str(run_dir),
                "points": len(rows),
                "final_epoch": last.get("epoch", ""),
                "final_total_evaluations": last.get("total_evaluations", ""),
                "final_local_search_evaluations": last.get("local_search_evaluations", ""),
                "final_archive_size": last.get("archive_size", ""),
                "final_y_best": last.get("y_best", ""),
                "best_y_best": max(float(row["y_best"]) for row in rows),
            }
        )

    plt.xlabel("Epoch" if args.x_axis == "epoch" else "Total evaluations")
    plt.ylabel("Best fitness")
    plt.title(args.title or f"{args.task} / {args.algorithm}")
    plt.grid(True, linestyle="--", alpha=0.35)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()

    write_summary(summary_path, summaries)
    print(f"Saved comparison curve to {output_path}")
    print(f"Saved comparison summary to {summary_path}")


if __name__ == "__main__":
    main()
