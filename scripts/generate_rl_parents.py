#!/usr/bin/env python3
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from algorithms._utils import init_v3_sampler, permutation_sampler
from utils import load_task, seed_everything


def parse_args():
    parser = argparse.ArgumentParser(description="Generate parent solutions for RL EarthBench environments.")
    parser.add_argument("--dims", type=int, default=124)
    parser.add_argument("--num-parents", type=int, default=20)
    parser.add_argument("--seed", type=int, default=2022)
    parser.add_argument("--sampler", choices=["init_v3", "permutation"], default="init_v3")
    parser.add_argument("--output-dir", default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    seed_everything(args.seed)

    output_dir = Path(args.output_dir or f"data/earth_{args.dims}/gen_points_v3")
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.sampler == "init_v3":
        candidates = init_v3_sampler(args.num_parents, args.dims)
    else:
        candidates = permutation_sampler(args.num_parents, args.dims)

    benchmark = load_task({"name": f"earth_{args.dims}", "file_path": None})
    records = []
    for idx, candidate in enumerate(candidates):
        y_best, x_best = benchmark(candidate, is_check=True)
        record = {
            "x_best": x_best.tolist() if isinstance(x_best, np.ndarray) else list(x_best),
            "y_best": float(y_best),
        }
        file_path = output_dir / f"parent_{idx:04d}.json"
        with open(file_path, "w") as f:
            json.dump(record, f)
        records.append(record)

    best = max(records, key=lambda item: item["y_best"])
    print(f"Saved {len(records)} parent solutions to {output_dir}")
    print(f"Best generated parent fitness: {best['y_best']}")


if __name__ == "__main__":
    main()
