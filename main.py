import hydra
import omegaconf
from omegaconf import DictConfig
import time
import ray  
import logging
import os
import json
import tqdm
import pickle
import csv
from utils import seed_everything, load_task, save_checkpoint, load_checkpoint, local_search, save_solutions

ray.init(
    num_cpus=16,
    num_gpus=1,
    include_dashboard=False,
    logging_level=logging.ERROR,
    _temp_dir=os.path.expanduser('/home/kouyulin/tmp'),
    ignore_reinit_error=True
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
    import torch
    import numpy as np
    import wandb
    import logging
    import time
    import random
    import warnings
    from algorithms._ea_operator import insert_mutation, swap_mutation
    warnings.simplefilter(action="ignore", category=FutureWarning)

    log = logging.getLogger(__name__)

    # if cfg['seed'] is not None:
    #     seed_everything(cfg['seed'])

    # set parameters
    task_cfg = cfg['task']
    alg_cfg = cfg['algorithm']
    alg_name = alg_cfg['name']
    if cfg.get('mutation') is not None and 'model' in alg_cfg:
        alg_cfg['model']['mutation_type'] = cfg['mutation']
    if cfg.get('adaptive_correlation') is not None and 'model' in alg_cfg:
        adaptive_correlation = cfg['adaptive_correlation']
        if not isinstance(adaptive_correlation, bool):
            adaptive_correlation = str(adaptive_correlation).lower() in ['1', 'true', 'yes', 'on']
        alg_cfg['model']['dynamic_correlation_threshold'] = adaptive_correlation

    checkpoint_dir = os.path.join('checkpoints', task_cfg['name'], alg_name, f"seed_{cfg['seed']}")
    if cfg['use_checkpoint'] and os.path.exists(checkpoint_dir):
        # load checkpoint
        max_epoch = max([int(f.split('.')[0].split('_')[-1]) for f in os.listdir(checkpoint_dir) if f.startswith('epoch_')])
        checkpoint_path = os.path.join(checkpoint_dir, f'epoch_{max_epoch}.pth')
        checkpoint = torch.load(checkpoint_path, weights_only=False)
        ckpt_cfg = checkpoint['cfg']
        alg_cfg = ckpt_cfg['algorithm']
        # restore random states
        torch.set_rng_state(checkpoint['torch_rng'])
        np.random.set_state(checkpoint['numpy_rng'])
        random.setstate(checkpoint['python_rng'])
    elif cfg['use_checkpoint']:
        raise ValueError(f'Checkpoint directory {checkpoint_dir} does not exist.')
    else:
        if cfg['seed'] is not None:
            seed_everything(cfg['seed'])
    
    individual_based_algs = ['sa', 'bo', 'bops', 'mergebo', 'dropout_bops', 'dropout_mergebo']
    if alg_name in individual_based_algs:
        cfg['epochs'] *= alg_cfg['pop_size']
    curr_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
    run_tag = time.strftime('%Y%m%d_%H%M%S', time.localtime())

    use_wandb = True
    if use_wandb:
        # login wandb
        wandb.login()
        wandb.init(
            project=cfg['project'],
            name='5-{}-{}-{}-{}'.format(curr_time, task_cfg['name'], alg_cfg['name'], cfg['seed']),
            config=omegaconf.OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True),
        )

    # device = torch.device(cfg['device'] if torch.cuda.is_available() else 'cpu')
    # log.info(f'device: {device}')
 
    func = load_task(task_cfg)
    dims = func.dims
    print(f"alg_cfg: {alg_cfg}")
    alg = hydra.utils.instantiate(alg_cfg['model'], dims=dims, lb=np.zeros(dims), ub=np.full(dims, dims-1))
    log.info(f'func: {func}, alg: {alg}, dims: {dims}')

    com = cfg.get('com', False)
    if not isinstance(com, bool):
        com = str(com).lower() in ['1', 'true', 'yes', 'on']
    result_alg_name = f"{alg_name}1" if com else alg_name
    local_result_dir = os.path.join('results', task_cfg['name'], result_alg_name, run_tag)
    os.makedirs(local_result_dir, exist_ok=True)
    metrics_path = os.path.join(local_result_dir, 'metrics.csv')
    final_summary_path = os.path.join(local_result_dir, 'final_summary.json')
    curve_path = os.path.join(local_result_dir, 'y_best_curve.png')
    with open(os.path.join(local_result_dir, 'config.json'), 'w') as f:
        json.dump(omegaconf.OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True), f, indent=2)
    metric_fieldnames = ['epoch', 'total_evaluations', 'local_search_evaluations', 'y_best', 'archive_size']
    with open(metrics_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=metric_fieldnames)
        writer.writeheader()
    
    n_workers = int(ray.available_resources()['CPU'])
    workers = [TaskWorker.remote(task_cfg) for _ in range(n_workers)]
    
    # If use checkpoint and checkpoint exists, then load it.
    checkpoint_save_freq = 1000
    checkpoint_dir = 'checkpoints'
    log.info(f"Use checkpoint: {cfg['use_checkpoint']}")
    if cfg['use_checkpoint'] and os.path.exists(checkpoint_dir):
        epochs, x_best, y_best = load_checkpoint(alg, alg_name, cfg, checkpoint_dir)
    else:
        epochs = 0
        x_best = None
        y_best = None

    log.info('------------ config -------------')
    log.info(cfg)
    log.info('--------------------------------')

    # train
    total_evaluations = 0
    local_search_evaluations = 0
    local_metrics = []
    local_search_cfg = cfg.get('local_search', {})
    local_search_enabled = bool(local_search_cfg.get('enabled', False)) and task_cfg['name'].startswith('earth')
    local_search_freq = int(local_search_cfg.get('freq', 50))
    local_search_top_k = int(local_search_cfg.get('top_k', 1))
    local_search_num_neighbors = int(local_search_cfg.get('num_neighbors', 4))
    local_search_mutation_types = list(local_search_cfg.get('mutation_types', ['insert', 'swap']))

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
                    raise ValueError(f"Unsupported local search mutation type: {mutation_type}")
        return neighbors

    def evaluate_candidates(candidates):
        futures = []
        for i, cand in enumerate(candidates):
            worker_idx = i % len(workers)
            futures.append(workers[worker_idx].evaluate.remote(cand))
        eval_results = ray.get(futures)
        return [result[0] for result in eval_results], [result[1] for result in eval_results]

    rl_freq = int(cfg.get('rl_freq', 500))
    for epoch in range(epochs + 1, cfg['epochs'] + 1):
        is_rl_alg = alg_name == 'dqn' or 'RL' in alg_name
        use_rl = is_rl_alg and rl_freq > 0 and epoch % rl_freq == 0

        # alg.ask
        if alg_name == 'map_elites':
            # cands, archive = alg.ask(n_repeats = min(epoch // 1000 + 1, 3))
            cands, archive = alg.ask(n_repeats = 1)
        else:
            if alg_name in ['dqn'] or 'RLEA' in alg_name:
                cands = alg.ask(n_repeats = 1, use_rl=use_rl, cur_x_best=x_best if 'UniRLEA' in alg_name else None)
            elif 'RLQD' in alg_name:
                cands, archive = alg.ask(n_repeats = 1, use_rl=use_rl, cur_x_best=x_best if 'UniRLQD' in alg_name else None)
            else:
                cands = alg.ask(n_repeats = 1)

            # for others
            # cands = alg.ask(n_repeats = min(epoch // 2000 + 1, 3))

        # time1 = time.time()
        cands_y, repaired_cands = evaluate_candidates(cands)

        # alg.tell 
        alg.tell(repaired_cands, cands_y)

        # log
        total_evaluations += len(cands)
        if y_best is None or np.max(cands_y) > y_best:
            max_idx = np.argmax(cands_y)
            x_best = repaired_cands[max_idx]
            y_best = cands_y[max_idx]

        if local_search_enabled and local_search_freq > 0 and epoch % local_search_freq == 0 and len(repaired_cands) > 0:
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

        # save checkpoint
        if cfg['save_checkpoint'] and epoch % checkpoint_save_freq == 0:
            save_checkpoint(epoch, alg, alg_name, cfg)

        # log.info('-------------------------------------')
        # log.info('Epoch: {}, y best: {}'.format(epoch, y_best))
        # log.info('-------------------------------------')
        # log.info('Epoch: {}, total evaluations: {}'.format(epoch, total_evaluations))
        # log.info('x best: {}, y best: {}'.format(x_best, y_best))
        # log.info('cands: {}, cands y: {}'.format(cands, cands_y))
        

        if use_wandb:
            if alg_name not in individual_based_algs or epoch % alg_cfg['pop_size'] == 0:
                wandb.log({
                    'epoch': epoch // alg_cfg['pop_size'] if alg_name in individual_based_algs else epoch,
                    'total evaluations': total_evaluations,
                    'y': y_best,
                    'archive size': len(archive) if ((alg_name in ['map_elites']) or ('QD' in alg_name)) else 0
                })
        metric_row = {
            'epoch': epoch // alg_cfg['pop_size'] if alg_name in individual_based_algs else epoch,
            'total_evaluations': total_evaluations,
            'local_search_evaluations': local_search_evaluations,
            'y_best': float(y_best) if y_best is not None else '',
            'archive_size': len(archive) if ((alg_name in ['map_elites']) or ('QD' in alg_name)) else 0
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
        plt.xlabel('Epoch')
        plt.ylabel('Best fitness')
        plt.title(f"{task_cfg['name']} / {alg_name}")
        plt.grid(True, linestyle='--', alpha=0.35)
        plt.tight_layout()
        plt.savefig(curve_path, dpi=200)
        plt.close()
    with open(final_summary_path, 'w') as f:
        json.dump({
            'task': task_cfg['name'],
            'algorithm': alg_name,
            'result_algorithm': result_alg_name,
            'seed': cfg['seed'],
            'epochs': cfg['epochs'],
            'total_evaluations': total_evaluations,
            'local_search_evaluations': local_search_evaluations,
            'y_best': float(y_best) if y_best is not None else None,
            'x_best': x_best.tolist() if type(x_best) == np.ndarray else x_best,
            'metrics_path': metrics_path,
            'curve_path': curve_path if local_metrics else None,
        }, f, indent=2)
    print(f"Local results saved to {local_result_dir}")
    if x_best is not None and task_cfg['name'].startswith('earth'):
        save_solutions(x_best, y_best, dims)


if __name__ == '__main__':
    main()
