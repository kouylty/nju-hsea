import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import defaultdict, deque
import math

from typing import List

class MLP(nn.Module):
    def __init__(self, obs_dim: int, n_actions: int, hidden_sizes: List[int] = [128, 128]):
        super().__init__()
        layers = []
        in_dim = obs_dim
        for hidden in hidden_sizes:
            layers.append(nn.Linear(in_dim, hidden))
            layers.append(nn.ReLU())
            in_dim = hidden
        layers.append(nn.Linear(in_dim, n_actions))
        self.net = nn.Sequential(*layers)
        # Better initial stability
        # for m in self.modules():
        #     if isinstance(m, nn.Linear):
        #         nn.init.kaiming_uniform_(m.weight, a=math.sqrt(5))
        #         if m.bias is not None:
        #             fan_in, _ = nn.init._calculate_fan_in_and_fan_out(m.weight)
        #             bound = 1 / math.sqrt(fan_in)
        #             nn.init.uniform_(m.bias, -bound, bound)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, a=0, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, obs, state=None, info={}):
        if not isinstance(obs, torch.Tensor):
            obs = torch.tensor(obs, dtype=torch.float32).to(next(self.parameters()).device)
        return self.net(obs), state

class Net(nn.Module):
    def __init__(self, dims, action_dim, d_model=64, nhead=4, num_layers=2):
        super().__init__()
        self.dims = dims
        self.seq_len = dims  # 每个部分长度都是 dims
        self.d_model = d_model

        # === Embedding 层 ===
        self.embedding = nn.Linear(1, d_model)

        # === 可学习的 Segment Embedding (p1, p2, o) ===
        self.segment_embedding = nn.Embedding(3, d_model)  # 3个segment

        # === 位置编码 (sin/cos) ===
        self.register_buffer("pos_encoding", self._build_pos_encoding(d_model))

        # Transformer 编码器
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.transformer_p = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.transformer_o = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Fitness 特征
        self.fitness_fc = nn.Linear(2, d_model)

        # LayerNorm
        self.norm_p = nn.LayerNorm(d_model)
        self.norm_o = nn.LayerNorm(d_model)
        self.fitness_norm = nn.LayerNorm(d_model)

        # Head
        self.head = nn.Sequential(
            nn.Linear(d_model * 4, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim)
        )

    def _build_pos_encoding(self, d_model):
        position = torch.arange(self.dims, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe = torch.zeros(self.dims, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe.unsqueeze(0)  # [1, self.dims, d_model]

    def encode(self, x, transformer, norm, segment_id):
        """
        x: [batch, dims]
        segment_id: 0, 1, 2 -> p1, p2, o
        """
        batch_size = x.size(0)
        x = x.view(batch_size, self.seq_len, 1)
        x_emb = self.embedding(x)

        # === 加入 Segment Embedding 和 Positional Encoding ===
        seg_emb = self.segment_embedding(torch.full((batch_size, self.seq_len), segment_id, device=x.device))
        pos_emb = self.pos_encoding[:, :self.seq_len, :].to(x.device)
        x_emb = x_emb + seg_emb + pos_emb

        x_emb = norm(x_emb)
        x_feat = transformer(x_emb)
        return x_feat.mean(dim=1)

    def forward(self, obs, state=None, info={}):
        if not isinstance(obs, torch.Tensor):
            obs = torch.tensor(obs, dtype=torch.float32).to(next(self.parameters()).device)

        f1_f2 = obs[:, :2]
        seq = obs[:, 2:]

        # === 拆分成 p1, p2, o ===
        p1 = seq[:, :self.dims]
        p2 = seq[:, self.dims:2*self.dims]
        o = seq[:, 2*self.dims:]

        # ---- 归一化到 [0, 1] ----
        p1 = p1 / (self.dims - 1)
        p2 = p2 / (self.dims - 1)
        o = o / (self.dims - 1)

        # ---- encode ----
        p1_feat = self.encode(p1, self.transformer_p, self.norm_p, segment_id=0)
        p2_feat = self.encode(p2, self.transformer_p, self.norm_p, segment_id=1)
        o_feat = self.encode(o, self.transformer_o, self.norm_o, segment_id=2)

        # ---- fitness ----
        f_feat = self.fitness_fc(f1_f2)
        f_feat = self.fitness_norm(f_feat)

        feat = torch.cat([p1_feat, p2_feat, o_feat, f_feat], dim=-1)
        logits = self.head(feat)
        return logits, state

class UniNet(nn.Module):
    def __init__(self, obs_dim: int, n_actions: int, num_segments: int, d_model=32, nhead=4, num_layers=2):
        super().__init__()

        # 状态 obs 组成： [f1_norm, f2_norm, self.kendall1, self.kendall2, self.step_count / self.dims],
        #                           np.array(pi1_pi2, dtype=np.float32),
        #                           np.array(self.kendall1_vec, dtype=np.float32),
        #                           np.array(self.kendall2_vec, dtype=np.float32)]
        # 网络架构：对最后两个 kendall vector 分别用 Transformer 编码，然后与前面的部分（前 self.fixed_part_len 个维度）拼接后过 FC 层得到动作值
        self.obs_dim = obs_dim
        self.num_segments = num_segments
        self.fixed_part_len = self.obs_dim - self.num_segments * 2
        self.d_model = d_model

        # === Embedding 层 ===
        self.embedding = nn.Linear(1, d_model)

        # === 可学习的 Segment Embedding ===
        self.segment_embedding = nn.Embedding(num_segments, d_model)

        # === 位置编码 (sin/cos) ===
        self.register_buffer("pos_encoding", self._build_pos_encoding(d_model))

        # 定义两个 Transformer 编码器 transformer1, transformer2，分别对 kendall1_vec 和 kendall2_vec 编码
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.transformer1 = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.transformer2 = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Head
        self.head = nn.Sequential(
            nn.Linear(self.fixed_part_len + d_model * 2, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, n_actions)
        )
    
    def _build_pos_encoding(self, d_model):
        position = torch.arange(self.num_segments, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe = torch.zeros(self.num_segments, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe.unsqueeze(0)  # [1, self.num_segments, d_model]

    def forward(self, obs, state=None, info={}):
        if not isinstance(obs, torch.Tensor):
            obs = torch.tensor(obs, dtype=torch.float32).to(next(self.parameters()).device)

        # 拆分 obs
        fixed_part = obs[:, :self.fixed_part_len]  # 前 self.fixed_part_len 个维度
        kendall1_vec = obs[:, self.fixed_part_len:self.fixed_part_len + self.num_segments]
        kendall2_vec = obs[:, self.fixed_part_len + self.num_segments:]

        # encode kendall1_vec
        batch_size = kendall1_vec.size(0)
        x1 = kendall1_vec.view(batch_size, self.num_segments, 1)
        x1_emb = self.embedding(x1)

        # === 加入 Segment Embedding 和 Positional Encoding ===
        seg_emb1 = self.segment_embedding(torch.arange(self.num_segments, device=x1.device).unsqueeze(0).repeat(batch_size, 1))
        pos_emb1 = self.pos_encoding[:, :self.num_segments, :].to(x1.device)
        x1_emb = x1_emb + seg_emb1 + pos_emb1

        x1_feat = self.transformer1(x1_emb)
        x1_feat = x1_feat.mean(dim=1)

        # encode kendall2_vec
        x2 = kendall2_vec.view(batch_size, self.num_segments, 1)
        x2_emb = self.embedding(x2)

        # === 加入 Segment Embedding 和 Positional Encoding ===
        seg_emb2 = self.segment_embedding(torch.arange(self.num_segments, device=x2.device).unsqueeze(0).repeat(batch_size, 1))
        pos_emb2 = self.pos_encoding[:, :self.num_segments, :].to(x2.device)
        x2_emb = x2_emb + seg_emb2 + pos_emb2

        x2_feat = self.transformer2(x2_emb)
        x2_feat = x2_feat.mean(dim=1)

        # 拼接所有特征
        feat = torch.cat([fixed_part, x1_feat, x2_feat], dim=-1)
        logits = self.head(feat)
        return logits, state

class UniNet_metadata(nn.Module):
    def __init__(self, obs_dim: int, n_actions: int, num_segments: int, d_model=32, nhead=4, num_layers=2, hidden_sizes=[512, 512]):
        super().__init__()

        # 状态 obs 组成： [f1_norm, f2_norm, self.kendall1, self.kendall2, self.step_count / self.dims],
        #                           np.array(pi1_pi2, dtype=np.float32),
        #                           np.array(self.kendall1_vec, dtype=np.float32),
        #                           np.array(self.kendall2_vec, dtype=np.float32)]
        # 网络架构：对最后两个 kendall vector 分别用 Transformer 编码，然后与前面的部分（前 self.fixed_part_len 个维度）拼接后过 FC 层得到动作值
        self.obs_dim = obs_dim
        self.num_segments = num_segments
        self.fixed_part_len = self.obs_dim - 2 * self.num_segments
        self.d_model = d_model

        # === Embedding 层 ===
        self.embedding = nn.Linear(1, d_model)

        # === 可学习的 Segment Embedding ===
        self.segment_embedding = nn.Embedding(num_segments + self.fixed_part_len, d_model)

        # === 位置编码 (sin/cos) ===
        self.register_buffer("pos_encoding", self._build_pos_encoding(d_model))

        # 定义两个 Transformer 编码器 transformer1, transformer2，分别对 kendall1_vec 和 kendall2_vec 编码
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.transformer1 = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.transformer2 = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Head
        # 根据 hidden_sizes 的长度动态构建 head
        layers = [nn.Linear(self.fixed_part_len + d_model * 2, hidden_sizes[0])]
        layers.append(nn.ReLU())
        for i in range(len(hidden_sizes) - 1):
            layers.append(nn.Linear(hidden_sizes[i], hidden_sizes[i + 1]))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(hidden_sizes[-1], n_actions))
        self.head = nn.Sequential(*layers)

    
    def _build_pos_encoding(self, d_model):
        position = torch.arange(self.num_segments + self.fixed_part_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe = torch.zeros(self.num_segments + self.fixed_part_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe.unsqueeze(0)  # [1, self.num_segments + self.fixed_part_len, d_model]

    def forward(self, obs, state=None, info={}):
        if not isinstance(obs, torch.Tensor):
            obs = torch.tensor(obs, dtype=torch.float32).to(next(self.parameters()).device)

        # 拆分 obs
        fixed_part = obs[:, :self.fixed_part_len]  # 前 self.fixed_part_len 个维度
        kendall1_vec = obs[:, self.fixed_part_len:self.fixed_part_len + self.num_segments]
        kendall2_vec = obs[:, self.fixed_part_len + self.num_segments:] 

        # 将 fixed_part 作为 meta data 拼接到 kendall1_vec 和 kendall2_vec 前
        kendall1_vec = torch.concat([fixed_part, kendall1_vec], dim=1)
        kendall2_vec = torch.concat([fixed_part, kendall2_vec], dim=1)

        # encode kendall1_vec
        batch_size = kendall1_vec.size(0)
        x1 = kendall1_vec.view(batch_size, self.num_segments + self.fixed_part_len, 1)
        x1_emb = self.embedding(x1)

        # === 加入 Segment Embedding 和 Positional Encoding ===
        seg_emb1 = self.segment_embedding(torch.arange(self.num_segments + self.fixed_part_len, device=x1.device).unsqueeze(0).repeat(batch_size, 1))
        pos_emb1 = self.pos_encoding[:, :self.num_segments + self.fixed_part_len, :].to(x1.device)
        x1_emb = x1_emb + seg_emb1 + pos_emb1

        x1_feat = self.transformer1(x1_emb)
        x1_feat = x1_feat.mean(dim=1)

        # encode kendall2_vec
        x2 = kendall2_vec.view(batch_size, self.num_segments + self.fixed_part_len, 1)
        x2_emb = self.embedding(x2)

        # === 加入 Segment Embedding 和 Positional Encoding ===
        seg_emb2 = self.segment_embedding(torch.arange(self.num_segments + self.fixed_part_len, device=x2.device).unsqueeze(0).repeat(batch_size, 1))
        pos_emb2 = self.pos_encoding[:, :self.num_segments + self.fixed_part_len, :].to(x2.device)
        x2_emb = x2_emb + seg_emb2 + pos_emb2

        x2_feat = self.transformer2(x2_emb)
        x2_feat = x2_feat.mean(dim=1)

        # 拼接所有特征
        feat = torch.cat([fixed_part, x1_feat, x2_feat], dim=-1)
        logits = self.head(feat)
        return logits, state

        
    

# 约束的格式： 
# 每一行两个数字 a, b，表示：
# 事件a在事件b之前发生

def coex2constraints(coex):
    '''
    将共现约束转换为事件约束。

    每一行两个数字 a, b，表示：
    事件a在事件b之前发生
    '''
    a, b = coex[:, 0], coex[:, 1]
    constraints1 = np.stack([a * 2, b * 2 + 1], axis=1) # 第一个约束：前一个物种的首现 < 后一个物种的末现
    constraints2 = np.stack([b * 2, a * 2 + 1], axis=1) # 第二个约束：后一个物种的首现 < 前一个物种的末现
    return np.vstack([constraints1, constraints2])

def FADLAD2constraints(FadLad):
    '''
    将 FADLAD 转换为事件约束。

    每一行两个数字 a, b，表示：
    事件a在事件b之前发生
    '''
    FadLad[:, 0] *= 2 # 首现
    FadLad[:, 1] = FadLad[:, 1] * 2 + 1 # 末现
    return FadLad

def compute_closure_optimized(constraints):
    """
    使用BFS计算传递闭包的优化版本，时间复杂度O(n × m)
    支持列表和NumPy数组输入
    
    参数:
    constraints: 约束列表或NumPy数组
    
    返回:
    与输入类型一致的传递闭包结果
    """
    if len(constraints) == 0:
        return constraints
    
    # 处理输入类型
    if isinstance(constraints, np.ndarray):
        constraints_list = constraints.tolist()
        return_numpy = True
    else:
        constraints_list = constraints
        return_numpy = False
    
    # 构建邻接表
    graph = defaultdict(list)
    all_nodes = set()
    
    for constraint in constraints_list:
        if len(constraint) >= 2:
            a, b = constraint[0], constraint[1]
            graph[a].append(b)
            all_nodes.add(a)
            all_nodes.add(b)
    
    # 初始化结果集合（包含所有原始约束）
    closure_set = set()
    for constraint in constraints_list:
        if len(constraint) >= 2:
            closure_set.add((constraint[0], constraint[1]))
    
    # 对每个节点进行BFS来寻找所有可达节点
    nodes_list = list(all_nodes)
    
    for start_node in nodes_list:
        if start_node not in graph:
            continue
            
        visited = set()
        queue = deque([start_node])
        
        while queue:
            current = queue.popleft()
            if current in visited:
                continue
            visited.add(current)
            
            # 添加新的约束关系
            if current != start_node:
                closure_set.add((start_node, current))
            
            # 将邻居加入队列
            for neighbor in graph.get(current, []):
                if neighbor not in visited:
                    queue.append(neighbor)
    
    # 转换为所需格式
    result = [list(pair) for pair in closure_set]
    
    if return_numpy:
        return np.array(result)
    else:
        return result