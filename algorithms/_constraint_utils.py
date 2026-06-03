import os
from functools import lru_cache

import numpy as np


def _read_int_table(path, min_cols):
    data = np.loadtxt(path, dtype=int)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    return data[:, :min_cols]


@lru_cache(maxsize=None)
def load_event_constraints(dims, bench_root="benchmarks"):
    bench_dir = os.path.join(bench_root, f"CONOPLib_{dims}")
    coex_path = os.path.join(bench_dir, "coex.dat")
    fad_lad_path = os.path.join(bench_dir, "Fb4L.dat")

    constraints = []

    if os.path.exists(coex_path):
        coex = _read_int_table(coex_path, 2)
        a = coex[:, 0]
        b = coex[:, 1]
        constraints.append(np.stack([a * 2, b * 2 + 1], axis=1))
        constraints.append(np.stack([b * 2, a * 2 + 1], axis=1))

    if os.path.exists(fad_lad_path):
        fad_lad = _read_int_table(fad_lad_path, 2)
        constraints.append(
            np.stack([fad_lad[:, 0] * 2, fad_lad[:, 1] * 2 + 1], axis=1)
        )

    if not constraints:
        return np.empty((0, 2), dtype=int)

    constraints = np.vstack(constraints).astype(int)
    constraints = constraints[
        (constraints[:, 0] >= 0)
        & (constraints[:, 0] < dims)
        & (constraints[:, 1] >= 0)
        & (constraints[:, 1] < dims)
        & (constraints[:, 0] != constraints[:, 1])
    ]
    return np.unique(constraints, axis=0)


def positions_of(x):
    positions = np.empty(len(x), dtype=int)
    positions[np.asarray(x, dtype=int)] = np.arange(len(x))
    return positions


def violated_constraints(x, constraints):
    if len(constraints) == 0:
        return constraints
    positions = positions_of(x)
    mask = positions[constraints[:, 0]] > positions[constraints[:, 1]]
    return constraints[mask]


def move_before(x, predecessor, successor):
    values = list(np.asarray(x).tolist())
    predecessor = int(predecessor)
    successor = int(successor)
    values.remove(predecessor)
    successor_index = values.index(successor)
    values.insert(successor_index, predecessor)
    return np.asarray(values, dtype=np.asarray(x).dtype)


def move_after(x, successor, predecessor):
    values = list(np.asarray(x).tolist())
    successor = int(successor)
    predecessor = int(predecessor)
    values.remove(successor)
    predecessor_index = values.index(predecessor)
    values.insert(predecessor_index + 1, successor)
    return np.asarray(values, dtype=np.asarray(x).dtype)


def constraint_insert_mutation(x, constraints, repeats=1):
    next_x = np.asarray(x).copy()
    for _ in range(max(1, repeats)):
        violations = violated_constraints(next_x, constraints)
        if len(violations) == 0:
            i, j = np.random.choice(range(len(next_x)), 2, replace=False)
            if i < j:
                next_x = np.concatenate(
                    (next_x[:i], next_x[i + 1 : j + 1], next_x[i : i + 1], next_x[j + 1 :])
                )
            else:
                next_x = np.concatenate(
                    (next_x[:j], next_x[i : i + 1], next_x[j:i], next_x[i + 1 :])
                )
            continue
        predecessor, successor = violations[np.random.randint(len(violations))]
        if np.random.rand() < 0.5:
            next_x = move_before(next_x, predecessor, successor)
        else:
            next_x = move_after(next_x, successor, predecessor)
    return next_x


def constraint_repair(x, constraints, passes=2):
    next_x = np.asarray(x).copy()
    for _ in range(max(1, passes)):
        violations = violated_constraints(next_x, constraints)
        if len(violations) == 0:
            break
        order = np.random.permutation(len(violations))
        for idx in order:
            predecessor, successor = violations[idx]
            current_violations = violated_constraints(
                next_x, np.asarray([[predecessor, successor]], dtype=int)
            )
            if len(current_violations) > 0:
                next_x = move_before(next_x, predecessor, successor)
    return next_x
