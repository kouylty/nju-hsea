import torch
from torch.quasirandom import SobolEngine
import numpy as np
import random 
from itertools import combinations

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


# ================== init function =================
def from_unit_cube(points, lb, ub):
    assert np.all(lb < ub) 
    assert lb.ndim == 1 
    assert ub.ndim == 1 
    assert points.ndim  == 2
    new_points = points * (ub - lb) + lb
    return new_points


def sobel_sampler(n, dims) -> np.ndarray:
    seed = np.random.randint(int(5e5))
    sobol = SobolEngine(dims, scramble=True, seed=seed)
    points = sobol.draw(n).to(dtype=torch.float64).cpu().detach().numpy()
    return points


def lhs_sampler(n, dims) -> np.ndarray:
    points = np.zeros((n, dims))
    centers = (1.0 + 2.0 * np.arange(0.0, n)) 
    centers = centers / float(2 * n)
    for i in range(0, dims):
        points[:, i] = centers[np.random.permutation(n)]

    perturbation = np.random.uniform(-1.0, 1.0, (n, dims)) 
    perturbation = perturbation / float(2 * n)
    points += perturbation
    return points


def permutation_sampler(n, dims, choices=None) -> np.ndarray:
    if choices is None:
        choices = range(dims)
    points = [np.random.choice(choices, dims, replace=False) for _ in range(n)]
    points = np.vstack(points)
    return points

def decode_soln(soln):
    # 我们的解 解码为 地科那边的解，如：
    #  我们的        他们的
    #    0     ->   1, 1
    #    1     ->   1, 2
    soln = np.array(soln, dtype=int)
    a_vals = soln // 2 + 1
    b_vals = soln % 2 + 1
    return np.stack((a_vals, b_vals), axis=1)

def encode_soln(data: np.ndarray) -> np.ndarray:
    # 根据decode_soln的逻辑进行编码
    # decode_soln: a_vals = soln // 2 + 1, b_vals = soln % 2 + 1
    # 所以编码应该是: soln = (a_vals - 1) * 2 + (b_vals - 1)
    result = []
    for i in range(data.shape[0]):
        a_val = data[i, 0]
        b_val = data[i, 1]
        encoded_val = (a_val - 1) * 2 + (b_val - 1)
        result.append(encoded_val)
    return np.array(result)

def expert_sampler(n, dims, choices=None) -> np.ndarray:
    dir_path = f'./benchmarks/CONOPLib_{dims}'
    expert_solutions = []
    
    # Try to read expert solutions from *soln.dat files
    try:
        import os
        # soln_files = [f for f in os.listdir(dir_path) if f.endswith('soln.dat')]

        # Edit by gzx, 2025.7.19 18:24. For better crossover.
        soln_files = [f for f in os.listdir(dir_path) if f.startswith('soln') and f.endswith('.dat')]
        
        # Read up to n solutions
        for soln_file in soln_files[:n//2]:
            file_path = os.path.join(dir_path, soln_file)
            data = np.loadtxt(file_path, dtype=int)
            solution = encode_soln(data)
            expert_solutions.append(solution)
    except (FileNotFoundError, OSError):
        pass  # Directory or files not found, proceed with random solutions
    
    # If we have enough expert solutions, return them
    if len(expert_solutions) >= n:
        return np.array(expert_solutions[:n])
    
    # If not enough expert solutions, fill the rest with random permutations
    remaining = n - len(expert_solutions)
    random_solutions = permutation_sampler(remaining, dims, choices)
    
    combined_solutions = np.vstack([np.array(expert_solutions), random_solutions])
    return combined_solutions

def init_v3_sampler(n, dims, choices=None) -> np.ndarray:
    import os
    import ctypes
    
    cwd = os.getcwd()
    base_path = f'./benchmarks/CONOPLib_{dims}'
    work_path = os.path.join(os.getcwd(), base_path)
    os.chdir(work_path)
    so = ctypes.CDLL('./CONOPLib.so')
    COMMOD_HANDLE = ctypes.c_void_p
    so.get_COMMOD_singleton.restype = COMMOD_HANDLE
    so.get_COMMOD_singleton.argtypes = []

    so.get_NSCT.restype = ctypes.c_int
    so.get_NSCT.argtypes = [COMMOD_HANDLE]

    so.get_NEVENT.restype = ctypes.c_int
    so.get_NEVENT.argtypes = [COMMOD_HANDLE]

    so.get_ISTATIC_0.restype = ctypes.POINTER(ctypes.c_short)
    so.get_ISTATIC_0.argtypes = [COMMOD_HANDLE]

    so.get_IROWS_row.restype = ctypes.POINTER(ctypes.c_int)
    so.get_IROWS_row.argtypes = [COMMOD_HANDLE, ctypes.c_int]

    commod = so.get_COMMOD_singleton()
    nsct = so.get_NSCT(commod)
    nevent = so.get_NEVENT(commod)
    ISTATIC_0 = so.get_ISTATIC_0(commod)
    IROWS_0 = so.get_IROWS_row(commod, 0)

    # Convert C arrays to Python lists for easier manipulation
    ISTATIC_0_list = [ISTATIC_0[i] for i in range(nsct * nevent)]
    
    # Create IROWS as a list of lists
    IROWS = []
    for i in range(nevent):
        row_ptr = so.get_IROWS_row(commod, i)
        IROWS.append([row_ptr[j] for j in range(2)])  # Assuming each row has 2 elements

    # 2. Calculate score for each section
    

    permutation_list = []
    for gen_iter in range(n):  
        # Initialize data structures
        sct_event = [[] for _ in range(nsct)]  # List of lists for each section
        size_event = [0] * nsct                # Size of events in each section
        sct_score = [0] * nsct                 # Score for each section
        event_score = [0] * nevent             # Score for each event
        max_event_sct = 0                       # Section with most events
        max_event_size = 0                      # Max events in a section
        max_sct_score = 0                       # Max section score

        # 1. Count total occurrences of each event
        for ii in range(nsct * nevent):
            if ISTATIC_0_list[ii] != -1:
                event_score[ii % nevent] += 1  
        event_count = 0
        sct_count = 0
        n_sct = 0
        for ii in range(nsct * nevent):
            if ISTATIC_0_list[ii] != -1:
                event_count += 1
                sct_count += event_score[ii % nevent]
            
            if (ii + 1) % nevent == 0:
                size_event[n_sct] = event_count
                sct_score[n_sct] = sct_count
                
                if sct_count > max_sct_score:
                    max_sct_score = sct_count
                    max_event_sct = n_sct
                if event_count > max_event_size:
                    max_event_size = event_count
                    
                event_count = 0
                sct_count = 0
                n_sct += 1

        # 引入随机性，保证 0 是真正的最大的 section
        # 其他的解随机选一个剖面作为主导
        if gen_iter != 0:
            max_event_sct = np.random.choice([i for i in range(nsct) if i != max_event_sct])

        # 3. Organize events into sections
        for n_sct in range(nsct):
            event_count = 0
            number = 0  # while level
            
            while event_count != size_event[n_sct]:
                for i in range(n_sct * nevent, nevent * (n_sct + 1)):
                    if ISTATIC_0_list[i] == number:
                        sct_event[n_sct].append(i % nevent)
                        event_count += 1
                number += 1

        # 4. Order sections based on rules
        th = [max_event_sct]  # section order
        hash_have_order = [False] * nsct
        hash_have_order[max_event_sct] = True
        hash_event_add = [False] * nevent
        
        for i in range(size_event[max_event_sct]):
            hash_event_add[sct_event[max_event_sct][i]] = True

        for count_insert in range(1, nsct):
            min_score = float('inf')
            min_sct = -1
            
            for n_sct in range(nsct):
                if hash_have_order[n_sct]:
                    continue
                    
                temp_score = 0
                for i in range(size_event[n_sct]):
                    if not hash_event_add[sct_event[n_sct][i]]:
                        temp_score += event_score[sct_event[n_sct][i]]
                
                if temp_score < min_score:
                    min_score = temp_score
                    min_sct = n_sct
            
            if min_sct == -1:  # All sections have been ordered or no valid section found
                break
                
            hash_have_order[min_sct] = True
            th.append(min_sct)
            
            for i in range(size_event[min_sct]):
                hash_event_add[sct_event[min_sct][i]] = True

        # 5. Initialize permutation with events from the highest-scoring section
        ini_perm_vt = sct_event[max_event_sct].copy()
        hash = [False] * nevent  # Track which events have been added
        
        for i in range(size_event[max_event_sct]):
            hash[sct_event[max_event_sct][i]] = True


        # 6. Insert remaining sections
        for n_sct in range(nsct):
            if n_sct == max_event_sct:
                continue
                
            last_have_add = -1  # Last inserted event position
            # temp_n_sct = n_sct
            count_add = 0
            
            # Get the section in the ordered list
            current_sct = th[n_sct] if n_sct < len(th) else n_sct
            
            for i in range(size_event[current_sct]):
                event = sct_event[current_sct][i]
                
                if not hash[event]:
                    if IROWS[event][1] == 1:  # FAD
                        if last_have_add == -1:
                            ini_perm_vt.insert(0, event)
                            last_have_add = 0
                            hash[event] = True
                        else:
                            ini_perm_vt.insert(last_have_add + 1, event)
                            hash[event] = True
                            last_have_add += 1
                        count_add += 1
                    
                    elif IROWS[event][1] == 2:  # LAD
                        last_FAD = -1
                        for xx in range(len(ini_perm_vt)):
                            if ini_perm_vt[xx] + 1 == event:  # Find corresponding FAD
                                last_FAD = xx
                                break
                        
                        if last_FAD > last_have_add:
                            last_have_add = last_FAD
                        
                        ini_perm_vt.insert(last_have_add + 1, event)
                        last_have_add += 1
                        hash[event] = True
                else:
                    for xx in range(len(ini_perm_vt)):
                        if ini_perm_vt[xx] == event:
                            last_have_add = xx
                            break
                


        # 7. Add any remaining events not yet included
        for n_event in range(nevent):
            if not hash[n_event]:
                if IROWS[n_event][1] == 1:  # FAD
                    ini_perm_vt.insert(0, n_event)
                elif IROWS[n_event][1] == 2:  # LAD
                    ini_perm_vt.append(n_event)

        # 8. Check and correct FAD-LAD ordering
        for xx in range(len(ini_perm_vt)):
            if IROWS[ini_perm_vt[xx]][1] == 1:  # FAD
                for t in range(xx-1, -1, -1):
                    if IROWS[ini_perm_vt[xx]][0] == IROWS[ini_perm_vt[t]][0]:
                        # Swap if FAD appears after its corresponding LAD
                        ini_perm_vt[xx], ini_perm_vt[t] = ini_perm_vt[t], ini_perm_vt[xx]
                        break
        permutation_list.append(np.array(ini_perm_vt, dtype=np.int32))
    os.chdir(cwd)
    return np.array(permutation_list, dtype=np.int32)

def get_init_samples(sampler_type, n, dims, lb, ub):
    if sampler_type == 'sobel':
        points = sobel_sampler(n, dims)
        points = from_unit_cube(points, lb, ub)
    elif sampler_type == 'lhs':
        points = lhs_sampler(n, dims)
        points = from_unit_cube(points, lb, ub)
    elif sampler_type == 'permutation':
        points = permutation_sampler(n, dims)
    elif sampler_type == 'expert':
        points = expert_sampler(n, dims)
    elif sampler_type == 'init_v3':
        points = init_v3_sampler(n, dims)
    else:
        raise NotImplementedError
    return points


# ================== init function =================
def select(dims, active_dims):
    idx = np.random.choice(range(dims), active_dims, replace=False)
    idx = np.sort(idx)
    return idx


def get_subset(train_X, idx):
    # return the position of idx in train_X
    if isinstance(train_X, np.ndarray):
        zeros_fn = np.zeros
        where_fn = np.where
    elif isinstance(train_X, torch.Tensor):
        zeros_fn = torch.zeros
        where_fn = torch.where
    subset_X = zeros_fn((len(train_X), len(idx)))
    for i, j in enumerate(idx):
        pos = where_fn(train_X == j)
        subset_X[:, i] = pos[1]
    return subset_X


def featurize(x, ret_type='torch'):
    assert ret_type in ['torch', 'numpy'], "ret_type must be 'torch' or 'numpy'."

    if isinstance(x, torch.Tensor):
        if x.dim() == 1:
            x = x.unsqueeze(0)
        assert x.dim() == 2, "Input x must be 1D or 2D tensor."

        batch_size, feature_len = x.size()
        
        x_expanded = x.unsqueeze(2)
        comparison_matrix = torch.sign(x_expanded - x_expanded.transpose(1, 2))
        upper_triangle = torch.triu(comparison_matrix, diagonal=1)
        featurize_x = upper_triangle[upper_triangle != 0].reshape(batch_size, -1)

        normalizer = torch.sqrt(torch.tensor(feature_len * (feature_len - 1) / 2.0, dtype=featurize_x.dtype))
        return featurize_x / normalizer

    elif isinstance(x, np.ndarray):
        if x.ndim == 1:
            x = np.expand_dims(x, axis=0)
        assert x.ndim == 2, "Input x must be 1D or 2D array."

        batch_size, feature_len = x.shape
        
        x_expanded = np.expand_dims(x, axis=2)
        comparison_matrix = np.sign(x_expanded - np.transpose(x_expanded, (0, 2, 1)))
        upper_triangle = np.triu(comparison_matrix, k=1)
        featurize_x = upper_triangle[upper_triangle != 0].reshape(batch_size, -1)

        normalizer = np.sqrt(feature_len * (feature_len - 1) / 2.0)
        return featurize_x / normalizer

    else:
        raise ValueError("Input x must be either a torch.Tensor or a np.ndarray.")


# def kendall_ranking_correlation(supports, queries, device='cuda'):
#     def to_torch(x, device):
#         if not isinstance(x, torch.Tensor):
#             return torch.tensor(x, device=device)
#         return x.to(device)
    
#     supports = to_torch(supports, device)
#     queries = to_torch(queries, device)
    
#     # 确保输入为二维 (batch_size, c)
#     if supports.dim() == 1:
#         supports = supports.unsqueeze(0)
#     if queries.dim() == 1:
#         queries = queries.unsqueeze(0)
#     c = supports.shape[1]
#     assert queries.shape[1] == c, "特征维度c不匹配"
    
#     # 生成所有两两组合索引
#     pair_num = c * (c - 1) // 2
#     c_pair = torch.tensor(list(combinations(range(c), 2)), device=device)
    
#     # 提取所有配对并计算符号差
#     def get_prank(tensor):
#         pairs = tensor[:, c_pair]  # (batch, pair_num, 2)
#         diff = pairs[:, :, 1] - pairs[:, :, 0]  # 计算差值
#         return torch.sign(diff).float()  # 符号化并转换为float
    
#     support_prank = get_prank(supports)  # (m, pair_num)
#     query_prank = get_prank(queries)     # (n, pair_num)
    
#     # 批量矩阵乘法计算得分
#     scores = torch.mm(query_prank, support_prank.T)  # (n, m)
    
#     return scores / pair_num  # 归一化

def kendall_ranking_correlation(supports, queries, device='cuda'):
    def to_torch(x, device):
        if not isinstance(x, torch.Tensor):
            return torch.tensor(x, device=device)
        return x.to(device)
    
    supports = to_torch(supports, device)
    queries = to_torch(queries, device)
    
    # 确保输入为二维 (batch_size, c)
    if supports.dim() == 1:
        supports = supports.unsqueeze(0)
    if queries.dim() == 1:
        queries = queries.unsqueeze(0)
    
    batch_size_support, c = supports.shape
    batch_size_query, c_query = queries.shape
    assert c == c_query, "特征维度c不匹配"
    
    pair_num = c * (c - 1) // 2
    
    # 方法1: 使用广播和矩阵运算 (推荐)
    def get_prank_fast(tensor):
        # 使用广播创建所有配对比较
        # tensor: (batch_size, c)
        # 扩展为 (batch_size, c, c) 然后取上三角部分
        expanded1 = tensor.unsqueeze(2)  # (batch_size, c, 1)
        expanded2 = tensor.unsqueeze(1)  # (batch_size, 1, c)
        diff_matrix = expanded2 - expanded1  # (batch_size, c, c)
        
        # 提取上三角部分 (不包括对角线)
        triu_mask = torch.triu(torch.ones(c, c, device=device), diagonal=1).bool()
        diff_triu = diff_matrix[:, triu_mask]  # (batch_size, pair_num)
        
        return torch.sign(diff_triu).float()
    
    # 方法2: 预计算索引并重用 (备选)
    if not hasattr(kendall_ranking_correlation, 'c_pair_cache'):
        kendall_ranking_correlation.c_pair_cache = {}
    
    cache_key = (c, device)
    if cache_key not in kendall_ranking_correlation.c_pair_cache:
        # 在目标设备上生成索引
        i, j = torch.triu_indices(c, c, offset=1, device=device)
        kendall_ranking_correlation.c_pair_cache[cache_key] = (i, j)
    
    i, j = kendall_ranking_correlation.c_pair_cache[cache_key]
    
    def get_prank_with_cache(tensor):
        # 直接使用预计算的索引
        pairs_i = tensor[:, i]  # (batch_size, pair_num)
        pairs_j = tensor[:, j]  # (batch_size, pair_num)
        diff = pairs_j - pairs_i
        return torch.sign(diff).float()
    
    # 根据数据规模选择方法
    if c > 100:  # 大规模数据使用方法1
        support_prank = get_prank_fast(supports)
        query_prank = get_prank_fast(queries)
    else:  # 小规模数据使用方法2
        support_prank = get_prank_with_cache(supports)
        query_prank = get_prank_with_cache(queries)
    
    # 批量矩阵乘法
    scores = torch.mm(query_prank, support_prank.T)  # (n, m)
    
    return scores / pair_num

def segmented_kendall_correlation(perm1, perm2, num_segments=64, device='cpu'):
    # perm2 acts as the reference
    def to_torch(x, device):
        if not isinstance(x, torch.Tensor):
            return torch.tensor(x, device=device)
        return x.to(device)
    
    perm1 = to_torch(perm1, 'cpu')
    perm2 = to_torch(perm2, 'cpu')
    
    n = len(perm1)
    segment_size = n // num_segments
    remainder = n % num_segments
    is_divisible = (remainder == 0)
    
    # 预计算perm2中元素到位置的映射
    element_to_perm2_position = torch.zeros(n, dtype=torch.long, device='cpu')
    element_to_perm2_position[perm2] = torch.arange(n, device='cpu')
    
    # 收集所有段
    segments_perm1 = []
    segments_perm2 = []
    valid_indices = []
    
    for segment_idx in range(num_segments):
        start_idx = segment_idx * segment_size
        end_idx = start_idx + segment_size if segment_idx < num_segments - 1 else n
        
        segment_from_perm1 = perm1[start_idx:end_idx]
        if len(segment_from_perm1) > 1:
            positions_in_perm2 = element_to_perm2_position[segment_from_perm1]
            sorted_positions, _ = torch.sort(positions_in_perm2)
            segment_from_perm2 = perm2[sorted_positions]
            
            segments_perm1.append(segment_from_perm1)
            segments_perm2.append(segment_from_perm2)
            valid_indices.append(segment_idx)
    # cpu: 0.001~0.002s, cuda: 0.006s, so we do computation for this part on cpu and move back the result to device(e.g. cuda) later.

    if not segments_perm1:
        return torch.zeros(num_segments, device=device)
    
    if not is_divisible:
        batch_perm1 = torch.stack(segments_perm1[:-1]).to(device)
        batch_perm2 = torch.stack(segments_perm2[:-1]).to(device)
        last_corr = kendall_ranking_correlation(segments_perm1[-1].unsqueeze(0), segments_perm2[-1].unsqueeze(0), device)
        
        correlations_matrix = kendall_ranking_correlation(batch_perm1, batch_perm2, device)
        batch_correlations = correlations_matrix.diagonal()
        batch_correlations = torch.cat([batch_correlations, last_corr[0]]).cpu()
    else:
        batch_perm1 = torch.stack(segments_perm1).to(device)
        batch_perm2 = torch.stack(segments_perm2).to(device)
        correlations_matrix = kendall_ranking_correlation(batch_perm1, batch_perm2, device)
        batch_correlations = correlations_matrix.diagonal().cpu()

    segment_correlations = torch.zeros(num_segments, device='cpu')
    for idx, segment_idx in enumerate(valid_indices):
        segment_correlations[segment_idx] = batch_correlations[idx].item()
        
    return segment_correlations
    

class FeatureCache:
    def __init__(self, input_type='numpy'):
        self.input_type = input_type
        self.cache = dict()

    def _get_key(self, x):
        return tuple(x.tolist())

    def push(self, x):
        feature = self.get(x)
        if feature is None:
            feature = featurize(x, self.input_type)
            self.cache[self._get_key(x)] = feature
        return feature

    def get(self, x):
        return self.cache.get(self._get_key(x), None)


if __name__ == '__main__':
    n, dims = 10, 200
    # points = sobel_sampler(n, dims)
    # print(type(points))
    # print(points)
    # points = lhs_sampler(n, dims)
    # print(type(points))
    # print(points)
    points = permutation_sampler(n, dims)
    print(type(points))
    print(points)
    x1 = torch.Tensor(points[0]).view(1, -1)
    x2 = torch.Tensor(points[1]).view(1, -1)
    print(kendall_ranking_correlation(x1, x2))