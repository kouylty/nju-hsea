import csv
import json
import logging
import os
import time

import hydra
import omegaconf
import ray
from omegaconf import DictConfig

from utils import load_task, save_solutions, seed_everything

ray.init(
    num_cpus=16,
    num_gpus=1,
    include_dashboard=False,
    logging_level=logging.ERROR,
    _temp_dir=os.path.expanduser('/home/kouyulin/tmp'),
    ignore_reinit_error=True,
)


@ray.remote(num_cpus=1)
class TaskWorker:
    def __init__(self, task_cfg):
        self.task_name = task_cfg['name']
        self.func = load_task(task_cfg)

    def evaluate(self, x):
        if self.task_name.startswith('earth'):
            return self.func(x, is_check=True)
        return self.func(x), x


@hydra.main(version_base=None, config_path='configs', config_name='config')
def main(cfg: DictConfig) -> None:
    import numpy as np
    import wandb
    from algorithms._ea_operator import insert_mutation, swap_mutation

    log = logging.getLogger(__name__)

    if cfg['seed'] is not None:
        seed_everything(cfg['seed'])

    task_cfg = cfg['task']
    alg_cfg = cfg['algorithm']
    alg_name = alg_cfg['name']

    if alg_name != 'map_elites':
        raise ValueError('main_best.py is only intended for the phased map_elites experiment.')

    phase1_end = int(cfg.get('best_phase1_end', 500))
    phase2_end = int(cfg.get('best_phase2_end', 1500))

    alg_cfg['model']['mutation_type'] = 'constraint_insert'
    alg_cfg['model']['constraint_repair_enabled'] = True
    alg_cfg['model']['lns_enabled'] = False

    curr_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
    run_tag = time.strftime('%Y%m%d_%H%M%S', time.localtime())

    use_wandb = True
    if use_wandb:
        wandb.login()
        wandb.init(
            project=cfg['project'],
            name='best1-{}-{}-{}-{}'.format(curr_time, task_cfg['name'], alg_cfg['name'], cfg['seed']),
            config=omegaconf.OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True),
        )

    func = load_task(task_cfg)
    dims = func.dims
    print(f'alg_cfg: {alg_cfg}')
    alg = hydra.utils.instantiate(alg_cfg['model'], dims=dims, lb=np.zeros(dims), ub=np.full(dims, dims - 1))
    log.info(f'func: {func}, alg: {alg}, dims: {dims}')

    com = cfg.get('com', False)
    if not isinstance(com, bool):
        com = str(com).lower() in ['1', 'true', 'yes', 'on']
    result_alg_name = f'{alg_name}1' if com else alg_name
    local_result_dir = os.path.join('results', task_cfg['name'], result_alg_name, run_tag)
    os.makedirs(local_result_dir, exist_ok=True)
    metrics_path = os.path.join(local_result_dir, 'metrics.csv')
    final_summary_path = os.path.join(local_result_dir, 'final_summary.json')
    curve_path = os.path.join(local_result_dir, 'y_best_curve.png')

    cfg_container = omegaconf.OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
    cfg_container['best_phased_strategy'] = {
        'phase1': {
            'epochs': f'1-{phase1_end}',
            'mutation_type': 'constraint_insert',
            'constraint_repair_enabled': True,
            'lns_enabled': False,
            'local_search_enabled': False,
        },
        'phase2': {
            'epochs': f'{phase1_end + 1}-{phase2_end}',
            'mutation_type': 'insert',
            'constraint_repair_enabled': True,
            'lns_enabled': True,
            'local_search_enabled': False,
        },
        'phase3': {
            'epochs': f'{phase2_end + 1}-{cfg["epochs"]}',
            'mutation_type': 'insert',
            'constraint_repair_enabled': True,
            'lns_enabled': True,
            'local_search_enabled': True,
        },
    }
    with open(os.path.join(local_result_dir, 'config.json'), 'w') as f:
        json.dump(cfg_container, f, indent=2)

    metric_fieldnames = [
        'epoch',
        'phase',
        'total_evaluations',
        'local_search_evaluations',
        'y_best',
        'archive_size',
    ]
    with open(metrics_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=metric_fieldnames)
        writer.writeheader()

    n_workers = int(ray.available_resources()['CPU'])
    workers = [TaskWorker.remote(task_cfg) for _ in range(n_workers)]

    local_search_cfg = cfg.get('local_search', {})
    local_search_freq = int(local_search_cfg.get('freq', 50))
    local_search_top_k = int(local_search_cfg.get('top_k', 1))
    local_search_num_neighbors = int(local_search_cfg.get('num_neighbors', 4))
    local_search_mutation_types = list(local_search_cfg.get('mutation_types', ['insert', 'swap']))

    def phase_for_epoch(epoch):
        if epoch <= phase1_end:
            return 'constraint'
        if epoch <= phase2_end:
            return 'lns'
        return 'local_search'

    def apply_phase(epoch):
        phase = phase_for_epoch(epoch)
        alg.constraint_repair_enabled = True
        if phase == 'constraint':
            alg.mutation_type = 'constraint_insert'
            alg.lns_enabled = False
            local_search_active = False
        elif phase == 'lns':
            alg.mutation_type = 'insert'
            alg.lns_enabled = True
            local_search_active = False
        else:
            alg.mutation_type = 'insert'
            alg.lns_enabled = True
            local_search_active = True
        return phase, local_search_active

    def make_local_neighbors(xs):
        neighbors = []
        for x in xs:
            x = np.asarray(x)
            for i in range(local_search_num_neighbors):
                mutation_type = local_search_mutation_types[i % len(local_search_mutation_types)]
                if mutation_type == 'insert':
                    neighbors.append(insert_mutation(x, repeats=1))
                elif mutation_type == 'swap':
                    neighbors.append(swap_mutation(x, repeats=1))
                else:
                    raise ValueError(f'Unsupported local search mutation type: {mutation_type}')
        return neighbors

    def evaluate_candidates(candidates):
        futures = []
        for i, cand in enumerate(candidates):
            worker_idx = i % len(workers)
            futures.append(workers[worker_idx].evaluate.remote(cand))
        eval_results = ray.get(futures)
        return [result[0] for result in eval_results], [result[1] for result in eval_results]

    total_evaluations = 0
    local_search_evaluations = 0
    local_metrics = []
    x_best = None
    y_best = None

    for epoch in range(1, cfg['epochs'] + 1):
        phase, local_search_active = apply_phase(epoch)

        cands, archive = alg.ask(n_repeats=1)
        cands_y, repaired_cands = evaluate_candidates(cands)
        alg.tell(repaired_cands, cands_y)

        total_evaluations += len(cands)
        if y_best is None or np.max(cands_y) > y_best:
            max_idx = np.argmax(cands_y)
            x_best = repaired_cands[max_idx]
            y_best = cands_y[max_idx]

        if (
            local_search_active
            and local_search_freq > 0
            and epoch % local_search_freq == 0
            and len(repaired_cands) > 0
        ):
            top_k = min(local_search_top_k, len(repaired_cands))
            top_indices = np.argsort(cands_y)[-top_k:]
            local_seed_cands = [repaired_cands[idx] for idx in top_indices]
            local_cands = make_local_neighbors(local_seed_cands)
            if local_cands:
                local_y, local_repaired_cands = evaluate_candidates(local_cands)
                alg.tell(local_repaired_cands, local_y)
                total_evaluations += len(local_cands)
                local_search_evaluations += len(local_cands)
                if y_best is None or np.max(local_y) > y_best:
                    max_idx = np.argmax(local_y)
                    x_best = local_repaired_cands[max_idx]
                    y_best = local_y[max_idx]

        if use_wandb:
            wandb.log({
                'epoch': epoch,
                'total evaluations': total_evaluations,
                'local search evaluations': local_search_evaluations,
                'y': y_best,
                'archive size': len(archive),
                'phase': phase,
            })

        metric_row = {
            'epoch': epoch,
            'phase': phase,
            'total_evaluations': total_evaluations,
            'local_search_evaluations': local_search_evaluations,
            'y_best': float(y_best) if y_best is not None else '',
            'archive_size': len(archive),
        }
        local_metrics.append(metric_row)
        with open(metrics_path, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=metric_fieldnames)
            writer.writerow(metric_row)

    ray.shutdown()

    if local_metrics:
        os.environ.setdefault('MPLCONFIGDIR', '/tmp/matplotlib')
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        epochs_for_plot = [row['epoch'] for row in local_metrics]
        y_best_for_plot = [row['y_best'] for row in local_metrics]
        plt.figure(figsize=(8, 4.5))
        plt.plot(epochs_for_plot, y_best_for_plot, linewidth=1.8)
        plt.axvline(phase1_end, color='gray', linestyle='--', alpha=0.45)
        plt.axvline(phase2_end, color='gray', linestyle='--', alpha=0.45)
        plt.xlabel('Epoch')
        plt.ylabel('Best fitness')
        plt.title(f"{task_cfg['name']} / {alg_name} phased best1")
        plt.grid(True, linestyle='--', alpha=0.35)
        plt.tight_layout()
        plt.savefig(curve_path, dpi=200)
        plt.close()

    with open(final_summary_path, 'w') as f:
        json.dump({
            'task': task_cfg['name'],
            'algorithm': alg_name,
            'result_algorithm': result_alg_name,
            'strategy': 'phased_constraint_lns_local_search',
            'seed': cfg['seed'],
            'epochs': cfg['epochs'],
            'phase1_end': phase1_end,
            'phase2_end': phase2_end,
            'total_evaluations': total_evaluations,
            'local_search_evaluations': local_search_evaluations,
            'y_best': float(y_best) if y_best is not None else None,
            'x_best': x_best.tolist() if type(x_best) == np.ndarray else x_best,
            'metrics_path': metrics_path,
            'curve_path': curve_path if local_metrics else None,
        }, f, indent=2)

    print(f'Local results saved to {local_result_dir}')
    if x_best is not None and task_cfg['name'].startswith('earth'):
        save_solutions(x_best, y_best, dims)


if __name__ == '__main__':
    main()
