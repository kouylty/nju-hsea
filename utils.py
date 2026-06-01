import torch
import numpy as np
import random
import logging
import json
import os 
import pickle

log = logging.getLogger(__name__)


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    log.info(f'Global seed set to {seed}')
    
    return seed


def load_task(task_cfg):
    if task_cfg['name'] == 'qap':
        from benchmarks import QAPProblem
        return QAPProblem(3)
    elif task_cfg['name'].startswith('tsp'):
        from benchmarks import TSPProblem
        return TSPProblem(task_cfg['file_path'])
    elif task_cfg['name'].startswith('earth'):
        if task_cfg['name'].endswith('124'):
            from benchmarks import EarthBenchmark_124
            return EarthBenchmark_124(task_cfg['file_path'])
        elif task_cfg['name'].endswith('278'):
            from benchmarks import EarthBenchmark_278
            return EarthBenchmark_278(task_cfg['file_path'])
        elif task_cfg['name'].endswith('902'):
            from benchmarks import EarthBenchmark_902
            return EarthBenchmark_902(task_cfg['file_path'])
        elif task_cfg['name'].endswith('904'):
            from benchmarks import EarthBenchmark_904
            return EarthBenchmark_904(task_cfg['file_path'])
        elif task_cfg['name'].endswith('934'):
            from benchmarks import EarthBenchmark_934
            return EarthBenchmark_934(task_cfg['file_path'])
        elif task_cfg['name'].endswith('2538'):
            from benchmarks import EarthBenchmark_2538
            return EarthBenchmark_2538(task_cfg['file_path'])
        elif task_cfg['name'].endswith('2574'):
            from benchmarks import EarthBenchmark_2574
            return EarthBenchmark_2574(task_cfg['file_path'])
        elif task_cfg['name'].endswith('2582'):
            from benchmarks import EarthBenchmark_2582
            return EarthBenchmark_2582(task_cfg['file_path'])
        else:
            raise NotImplementedError
    else:
        raise NotImplementedError
    

def save_checkpoint(epoch, alg, alg_name, cfg, checkpoint_base_dir='checkpoints'):
    task_name = cfg['task']['name']
    seed = cfg['seed']
    checkpoint_dir = os.path.join(checkpoint_base_dir, task_name, alg_name, f'seed_{seed}')

    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    # 保存 checkpoint，包括 cfg 中所有配置 和 算法的状态
    checkpoint_path = os.path.join(checkpoint_dir, f'epoch_{epoch}.pth')
    print(f"Saving checkpoint to {checkpoint_path}")
    torch.save({
        'cfg': cfg,
        'alg_state_dict': alg.get_ckpt_dict(),
        'torch_rng': torch.get_rng_state(),
        'numpy_rng': np.random.get_state(),
        'python_rng': random.getstate(),
    }, checkpoint_path)

def load_checkpoint(alg, alg_name, cfg, checkpoint_base_dir='checkpoints'):
    task_name = cfg['task']['name']
    seed = cfg['seed']
    checkpoint_dir = os.path.join(checkpoint_base_dir, task_name, alg_name, f'seed_{seed}')

    # Load the latest checkpoint
    max_epoch = max([int(f.split('.')[0].split('_')[-1]) for f in os.listdir(checkpoint_dir) if f.startswith('epoch_')])
    checkpoint_path = os.path.join(checkpoint_dir, f'epoch_{max_epoch}.pth')
    checkpoint = torch.load(checkpoint_path, weights_only=False)
    x_best, y_best = alg.load_ckpt_dict(checkpoint['alg_state_dict'])
    return max_epoch, x_best, y_best

def load_solutions(file_path):
    '''
    The data format should be as follows:
    [
        {"x": [1, 2, ..., 3], "y": 10},
        {"x": [4, 5, ..., 6], "y": 20},
        {"x": [7, 8, ..., 9], "y": 30}
    ]
    '''
    with open(file_path, 'r') as f:
        solutions = json.load(f)

    xs = [np.array(sol['x']) for sol in solutions]
    ys = [sol['y'] for sol in solutions]
    return xs, ys

def save_solutions(x_best, y_best, dims):
    output_dir = './data/solutions'
    os.makedirs(output_dir, exist_ok=True)

    file_path = os.path.join(output_dir, f"earth_{dims}.json")
    solution = {
        "x": x_best.tolist() if type(x_best) == np.ndarray else x_best,  # np.ndarray -> List
        "y": y_best
    }

    with open(file_path, 'w') as f:
        json.dump([solution], f)
    
    print(f"Best solution saved to {file_path}")

############ RL 环境相关 ###############
def _get_env(task_name):
    from Environments import EarthBenchEnv
    dims = int(task_name.split('_')[-1])
    return EarthBenchEnv(np.random.permutation(dims), np.random.permutation(dims), dims)
    
def _get_parents(dims):
    parents = []
    dir_path = f"data/earth_{dims}/gen_points_v3"
    # Read all files in the directory
    file_list = os.listdir(dir_path)
    
    for file_name in file_list:
        with open(os.path.join(dir_path, file_name), 'r') as f:
            data = json.load(f)
        x_best = data['x_best']
        y_best = data['y_best']
        parents.append((x_best, y_best))
    return parents


def _get_envs(task_name, num_envs, different_envs=1, env_type='Dummy', need_indices=False, use_local_search=False):
    import itertools
    import tianshou as ts
    from Environments import EarthBenchEnv
    dims = int(task_name.split('_')[-1])

    parents = _get_parents(dims)
    all_pairs = list(itertools.combinations(parents, 2))
    assert num_envs <= len(all_pairs)
    assert different_envs <= num_envs
    # idx = np.random.choice(len(all_pairs), size=num_envs, replace=False)

    if different_envs == 1:
        # 所有的 Env 都一样
        idx = np.argsort([np.mean([p[0][1], p[1][1]]) for p in all_pairs])[800:800+1]
        idx = np.repeat(idx, num_envs)
    else:
        idx = np.random.choice(len(all_pairs), size=different_envs, replace=False)
        idx = np.repeat(idx, num_envs // different_envs)
        # 如果不能整除，把最好的环境重复，拼接到最后
        if num_envs % different_envs != 0:
            print("Warning: num_envs mod different_envs != 0, some envs will be duplicated.")
            extra_idx = np.argsort([np.mean([all_pairs[i][0][1], all_pairs[i][1][1]]) for i in idx])[-(num_envs % different_envs):]
            idx = np.concatenate([idx, extra_idx])

    selected_pairs = [all_pairs[i] for i in idx]
    # print(f"selected pairs idx: {idx[:1]}, selected pairs fitness: {[ (p[0][1], p[1][1]) for p in selected_pairs[:1]] }")
    if use_local_search:
        p1, f1, p2, f2 = selected_pairs[0][0][0], selected_pairs[0][0][1], selected_pairs[0][1][0], selected_pairs[0][1][1]
        p1, f1 = local_search(p1, f1, dims=dims, epochs=200)
        p2, f2 = local_search(p2, f2, dims=dims, epochs=200)
        selected_pairs = [((p1, f1), (p2, f2))] * num_envs

    if need_indices:
        if env_type == 'Dummy':
            return ts.env.DummyVectorEnv([lambda p=pair: EarthBenchEnv(np.array(p[0][0]), np.array(p[1][0]), dims, p[0][1], p[1][1], local_search=use_local_search) for pair in selected_pairs]), idx
        elif env_type == 'Subproc':
            return ts.env.SubprocVectorEnv([lambda p=pair: EarthBenchEnv(np.array(p[0][0]), np.array(p[1][0]), dims, p[0][1], p[1][1], local_search=use_local_search) for pair in selected_pairs]), idx
        else:
            raise ValueError("Invalid env_type. Choose either 'Dummy' or 'Subproc'.")
    else:
        if env_type == 'Dummy':
            return ts.env.DummyVectorEnv([lambda p=pair: EarthBenchEnv(np.array(p[0][0]), np.array(p[1][0]), dims, p[0][1], p[1][1], local_search=use_local_search) for pair in selected_pairs])
        elif env_type == 'Subproc':
            return ts.env.SubprocVectorEnv([lambda p=pair: EarthBenchEnv(np.array(p[0][0]), np.array(p[1][0]), dims, p[0][1], p[1][1], local_search=use_local_search) for pair in selected_pairs])
        else:
            raise ValueError("Invalid env_type. Choose either 'Dummy' or 'Subproc'.")
    

def _get_unified_env(num_segments=None, window_size=1):
    # from Environments import EarthBenchEnvUnified as EarthBenchEnv
    # from Environments import EarthBenchEnvUnifiedNGS as EarthBenchEnv
    from Environments import EarthBenchEnvUnifiedNGSWindow as EarthBenchEnv
    
    dims = 124
    return EarthBenchEnv(np.random.permutation(dims), np.random.permutation(dims), np.random.permutation(dims), dims, num_segments=num_segments, window_size=window_size)   

def _get_unified_envs(num_envs, different_envs_per_dim=1, num_segments=None, window_size=1, env_type='Dummy', need_indices=False, use_local_search=False):
    '''
    从所有维度的 EarthBenchEnv 中，随机选择 num_envs 个环境返回
    '''
    # 每个维度尽量均衡选择
    import itertools
    import tianshou as ts
    # from Environments import EarthBenchEnvUnified as EarthBenchEnv
    # from Environments import EarthBenchEnvUnifiedNGS as EarthBenchEnv
    from Environments import EarthBenchEnvUnifiedNGSWindow as EarthBenchEnv
    # dims_list = [124, 278, 904, 934, 2538, 2582]
    dims_list = [124, 904, 2538]
    # dims_list = [278, 934, 2582]
    num_envs_list = [num_envs // len(dims_list) for _ in dims_list]
    for i in range(num_envs % len(dims_list)):
        num_envs_list[i] += 1
    
    # 需要得到：1. 每个环境的 reference parent，即 y 值最大的 parent
    # 2. 每个环境的 parents 对，从其中随机选择对应的 num_envs 个环境
    # 3. 初始化 EarthBenchEnvUnified 环境，用 p1, p2, ref, dims, f1, f2 初始化
    all_selected_pairs = []
    indices = []
    for i, dims in enumerate(dims_list):
        parents = _get_parents(dims)
        # 找到 reference parent，每个维度的 reference parent 固定
        ref = max(parents, key=lambda x: x[1])
        all_pairs = list(itertools.combinations(parents, 2))

        different_envs_per_dim = min(different_envs_per_dim, num_envs_list[i])
        idx = np.random.choice(len(all_pairs), size=different_envs_per_dim, replace=False)
        idx = np.repeat(idx, num_envs_list[i] // different_envs_per_dim)
        # 如果不能整除，重新随机选择一些环境拼接到最后
        if num_envs_list[i] % different_envs_per_dim != 0:
            extra_idx = np.random.choice(len(all_pairs), size=(num_envs_list[i] % different_envs_per_dim), replace=False)
            idx = np.concatenate([idx, extra_idx])
        indices.extend(idx.tolist())
        selected_pairs = [(all_pairs[i], ref, dims) for i in idx]
        all_selected_pairs.extend(selected_pairs)
    
    assert len(all_selected_pairs) == num_envs, f"Expected {num_envs} envs, but got {len(all_selected_pairs)}"

    # print('='*50)
    # # print(f"{all_selected_pairs[0][0]}\n")
    # # print(f"{all_selected_pairs[0][1]}\n")
    # # print(f"{all_selected_pairs[0][2]}\n")
    # for pair, ref, dims in all_selected_pairs:
    #     print(pair[0][0], '\n')
    #     print(ref, '\n')
    #     print(dims, '\n')
    #     break
    # print('='*50)

    if need_indices:
        if env_type == 'Dummy':
            return ts.env.DummyVectorEnv([lambda p=pair, r=ref, d=dims: EarthBenchEnv(np.array(p[0][0]), np.array(p[1][0]), r[0], d, p[0][1], p[1][1], num_segments=num_segments, window_size=window_size, local_search=use_local_search) for pair, ref, dims in all_selected_pairs]), indices
        elif env_type == 'Subproc':
            return ts.env.SubprocVectorEnv([lambda p=pair, r=ref, d=dims: EarthBenchEnv(np.array(p[0][0]), np.array(p[1][0]), r[0], d, p[0][1], p[1][1], num_segments=num_segments, window_size=window_size, local_search=use_local_search) for pair, ref, dims in all_selected_pairs]), indices
        else:
            raise ValueError("Invalid env_type. Choose either 'Dummy' or 'Subproc'.")
    else:
        if env_type == 'Dummy':
            return ts.env.DummyVectorEnv([lambda p=pair, r=ref, d=dims: EarthBenchEnv(np.array(p[0][0]), np.array(p[1][0]), r[0], d, p[0][1], p[1][1], num_segments=num_segments, window_size=window_size, local_search=use_local_search) for pair, ref, dims in all_selected_pairs])
        elif env_type == 'Subproc':
            return ts.env.SubprocVectorEnv([lambda p=pair, r=ref, d=dims: EarthBenchEnv(np.array(p[0][0]), np.array(p[1][0]), r[0], d, p[0][1], p[1][1], num_segments=num_segments, window_size=window_size, local_search=use_local_search) for pair, ref, dims in all_selected_pairs])
        else:
            raise ValueError("Invalid env_type. Choose either 'Dummy' or 'Subproc'.")


def local_search(x, y=None, dims=None, epochs=200, func=None):
    if type(x) != np.ndarray:
        x = np.array(x)
    from algorithms import SA
    if func is None:
        if dims == 124:
            from benchmarks import EarthBenchmark_124 as EarthBenchmark
        elif dims == 278:
            from benchmarks import EarthBenchmark_278 as EarthBenchmark
        elif dims == 902:
            from benchmarks import EarthBenchmark_902 as EarthBenchmark
        elif dims == 904:
            from benchmarks import EarthBenchmark_904 as EarthBenchmark
        elif dims == 934:
            from benchmarks import EarthBenchmark_934 as EarthBenchmark
        elif dims == 2538:
            from benchmarks import EarthBenchmark_2538 as EarthBenchmark
        elif dims == 2574:
            from benchmarks import EarthBenchmark_2574 as EarthBenchmark
        elif dims == 2582:
            from benchmarks import EarthBenchmark_2582 as EarthBenchmark
        else:
            raise ValueError(f"Unsupported dims: {dims}")
        func = EarthBenchmark()
    if y is None:
        y = func(x)
    sa = SA(dims=len(x), lb=None, ub=None, decay=0.99, T=100, update_freq=100, mutation_type='insert', init_sampler_type='permutation')
    best_x = x
    best_y = y
    sa.tell([x], [y])
    for epoch in range(epochs):
        cand = sa.ask(n_repeats=1)[0]
        cand_y = func(cand)
        sa.tell([cand], [cand_y])
        if cand_y > best_y:
            best_x = cand
            best_y = cand_y
    assert type(best_x) == np.ndarray
    return best_x, best_y

