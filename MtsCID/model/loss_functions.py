from __future__ import absolute_import, print_function
import torch
import torch.nn as nn
from torch.nn import functional as F

class ContrastiveLoss(nn.Module):
    def __init__(self, temp_param, eps=1e-12, reduce=True):
        super(ContrastiveLoss, self).__init__()
        self.temp_param = temp_param
        self.eps = eps
        self.reduce = reduce
        # ContrastiveLoss 类，用于实现对比学习损失。类继承自 nn.Module 以便于参数管理和调用。构造函数接收三个参数：
        # temp_param：温度参数，用于控制 softmax 的平滑程度（本实现中未使用，但保留下来可能用于扩展）。
        # eps：用于数值稳定的微小常数（未直接使用）。
        # reduce：布尔标志，用于决定返回整个批次的平均损失还是逐时间步的损失矩阵

    def get_score(self, query, key):
        '''
        query : (NxL) x C or N x C -> T x C  (initial latent features)
        key : M x C     (memory items)
        '''
        qs = query.size()
        ks = key.size()

        score = torch.matmul(query, torch.t(key))   # Fea x Mem^T : (TXC) X (CXM) = TxM
        score = F.softmax(score, dim=1) # TxM

        return score
        #get_score 函数，用于计算查询向量与记忆原型之间的相似度：
        # 函数接收 query（查询向量）和 key（原型集合）。query 可以是 (N×L)×C（批次中所有时间步展平后）或 N×C，其中 T=N×L。
        # qs 和 ks 保存张量形状用于调试（未被后续使用）。
        # score = torch.matmul(query, torch.t(key))：计算查询和各个原型的点积。query 形状为 T×C，key 形状为 M×C，转置后为 C×M，点积结果是 T×M 矩阵，每行包含当前查询与所有原型的相似度。
        # score = F.softmax(score, dim=1)：对每个查询的相似度向量在原型维度上做 softmax，得到与论文中公式 (13) 中相同的注意力权重
    
    def forward(self, queries, items):
        '''
        anchor : query
        positive : nearest memory item
        negative(hard) : second nearest memory item
        queries : N x L x C
        items : M x C
        '''
        batch_size = queries.size(0)
        d_model = queries.size(-1)
        # 三元组对比损失的前向传播：
        # batch_size，d_model 得到批次大小和特征维度。

        # margin from 1.0 
        loss = torch.nn.TripletMarginLoss(margin=1.0, reduce=self.reduce)
        # loss：实例化三元组损失对象，设定 margin 为 1.0，如果reduce=True，损失会在批次内求平均。

        queries = queries.contiguous().view(-1, d_model)    # (NxL) x C >> T x C
        score = self.get_score(queries, items)      # TxM
        # queries：将输入查询从 N×L×C 展平为 T×C，其中 T=N×L
        # score：调用 get_score 计算每个查询与所有原型的注意力权重

        # gather indices of nearest and second nearest item
        _, indices = torch.topk(score, 2, dim=1)
        # indices：找出每个查询最相似的两个原型的索引（最大 softmax 权重）

        # 1st and 2nd nearest items (l2 normalized)
        pos = items[indices[:, 0]]  # TxC
        neg = items[indices[:, 1]]  # TxC
        anc = queries              # TxC
        # pos 选择得分最高（最相似）的原型作为正样本；neg = items[indices[:, 1]] 选择第二相似的原型作为硬负样本。
        # anc 将查询本身视为 anchor。注意作者未对查询或原型做 l2 归一化，这与下文的 NearestSim 不同。

        spread_loss = loss(anc, pos, neg)
        # spread_loss：计算三元组损失，鼓励 anchor 与正样本靠近、与负样本远离。这样可以提高原型间的区分度，在对应的原型学习任务中使不同原型捕获不同的关系模式。

        if self.reduce:
            return spread_loss
        spread_loss = spread_loss.contiguous().view(batch_size, -1)       # N x L
        # 若 self.reduce 为真则直接返回平均损失；否则将损失展开为 N×L 的矩阵，保留每个时间步的损失值。这在一些需要时序级别监督的任务中很有用

        return spread_loss     # N x L
        
class GatheringLoss(nn.Module):
    # 该类负责计算测试阶段所需的 关系偏差（Relationship Deviation），即每个查询表示与最近原型之间的距离，用于异常评分。它还可在训练阶段作为聚合损失，将查询表示拉近其最近原型。
    def __init__(self, reduction='none', memto_framework=True):
        super(GatheringLoss, self).__init__()
        self.reduction = reduction
        self.memto_framework = memto_framework
        # reduction 指定损失的归约方式：默认为 'none'，表示返回每个时间步的损失矩阵；当设为 'mean' 时，将返回标量平均值。
        # memto_framework：布尔值，用于兼容作者之前的 Memto 模型。

    def get_score(self, query, key):
        '''
        query : (NxL) x C or N x C -> T x C  (initial latent features)
        key : M x C     (memory items)
        '''
        score = torch.matmul(query, key.T)  # Fea x Mem^T : (TXC) X (CXM) = TxM
        score = F.softmax(score, dim=1)  # TxM
        return score
        # 一个得分计算方法，与 ContrastiveLoss 类似：参数 query、key 含义同前。
        # 直接使用矩阵乘法计算查询与原型的点积，并在原型维度上应用 SoftMax 得到注意力权重。此函数在本类中未直接使用，但保留作参考。
    
    def forward(self, queries, items):
        # GatheringLoss 的核心实现，其逻辑如下：
        '''
        queries : N x L x C
        items : M x C
        '''
        batch_size = queries.size(0)
        loss_mse = torch.nn.MSELoss(reduction=self.reduction)
        # batch_size 批次大小。
        # loss_mse：根据 reduction 参数实例化均方误差损失对象。

        #  To eliminate the impact of magnitude, we use the queries in the unit magnitude
        f = torch.fft.rfft(queries, dim=-2).permute(0, 2, 1)
        i_query_angle = torch.angle(f)
        unit_magnitude_queries = torch.fft.irfft(torch.exp(-1j * i_query_angle)).permute(0, 2, 1)
        #  查询表示的幅度归一化：
        #  torch.fft.rfft(queries, dim=-2).permute(0, 2, 1)：对查询在时间维 (L) 上作实数快速傅里叶变换 (rFFT)，然后换维度顺序，得到频域复数表示 f，形状为 N × (L/2+1) × C。
        #  torch.angle(f)：取复数的相角，代表每个频率的相位信息
        #  torch.fft.irfft(torch.exp(-1j * i_query_angle)).permute(0, 2, 1)：将相角恢复到时域，但用单位幅度 exp(-1j*angle)（幅度为1）重建，忽略原始幅值信息；
        #  最后调整维度回到 N×L×C。这一技巧与论文无直接对应，但参考了 Memto 等研究中使用单位幅值化特征来抑制幅度差异的影响，使得相似度计算主要依赖相位信息或形状特征。

        if self.memto_framework: #全局共享原型 items: (M,C) 对应论文里的固定正弦原型场景（全体样本共用一套）
            score = torch.einsum('bij,kj->bik', unit_magnitude_queries, items)
            # score = torch.einsum('bij,kj->bik', queries, items)
            _, indices = torch.topk(score, 1, dim=-1)
            step_basis = torch.gather(items.unsqueeze(0).repeat(batch_size, 1, 1), 1, indices.expand(-1, -1, items.size(-1))) # step_basis全局原型集合里，与第 b 个样本第 i 个时间步最相似的那个原型向量
            gathering_loss = loss_mse(queries, step_basis)
            # if self.memto_framework: 分支用于处理全局与批次级记忆：
            # 当为真时，认为 items 是全局原型 (M×C)，采用 torch.einsum( ) 计算每个批次中每个时间步与所有原型的点积，得到 batch_size×L×M 的得分矩阵。
            #   对最后一维 j（特征维）做内积，把每个 (N, L) 的查询向量去和 M 个全局原型分别点乘，得到score ∈ (N, L, M)，即每个时间步对每个原型的相似度。
            #   用归一化后的 unit_magnitude_queries 计算 score，让 score 实质上变成“余弦相似度”（items 通常也可被预归一化，或不归一化但不影响 topK 排序的方向性）。
            # 然后通过 torch.topk(score, 1) 找到每个查询最相似的原型索引；
            # items.unsqueeze(0).repeat(batch_size,1,1) 将原型复制到批次维(N,M,C)；gather 在 dim=1（原型维M） 上按 indices 选取,为了让 gather 在所有维度上形状对齐，需要把 indices 从 (N,L,1) 扩展为 (N,L,C),使得同一个原型索引在该时间步的 C 个特征维上被重复使用。
            #   torch.gather 根据索引选取对应原型，得到 step_basis（每个查询的最近原型）（N,L,C）。
            # 最后使用 loss_mse(queries, step_basis) 计算查询与该原型之间的均方误差。注意这里直接用未单位化的 queries 计算误差，即鼓励原特征向最近原型靠近。
        
        else: # 逐样本原型 items: (B,M,C)
            score = torch.einsum('bij,bkj->bik', unit_magnitude_queries, items)
            # score = torch.einsum('bij,bkj->bik', queries, items)
            _, indices = torch.topk(score, 1, dim=-1)
            C = torch.gather(items, 1, indices.expand(-1, -1, items.size(-1)))
            gathering_loss = loss_mse(queries, C)
            # 当为假时，表示每个样本拥有独立的原型集合 items（形状为 B×M×C）。得分计算使用 torch.einsum( )；随后按同样方式选取最相似原型并计算 MSE。

        if not self.reduction == 'none':
            return gathering_loss
            # if not self.reduction == 'none': return gathering_loss：如果使用 'mean' 或 'sum' 等归约方式，直接返回聚合后的损失。
        
        gathering_loss = torch.sum(gathering_loss, dim=-1)  # T
        gathering_loss = gathering_loss.contiguous().view(batch_size, -1)   # N x L
        # 当 reduction == 'none' 时，gathering_loss 形状为 N×L×C；通过 torch.sum(gathering_loss, dim=-1) 在特征维求和得到每个时间步的误差（关系偏差）；
        # 再重塑为 N×L。这个结果在测试阶段经过 softmax 再与重构误差结合形成最终的异常评分，如论文公式 (17) 所述

        return gathering_loss

class EntropyLoss(nn.Module):
    def __init__(self, eps=1e-12):
        super(EntropyLoss, self).__init__()
        self.eps = eps # 为 log(x) 的数值稳定性预留的极小量

    def forward(self, x):
        '''
        x (attn_weights) : TxM
        '''
        loss = -1 * x * torch.log(x + self.eps) # 按元素计算 -x*log(x+eps)（信息熵的逐元素公式）
        loss = torch.sum(loss, dim=-1) # 对最后一维 M 求和：得到每个时间步 T 的熵 H(x_t)
        loss = torch.mean(loss) # 对所有时间步取均值：得到批内平均熵
        return loss

class NearestSim(nn.Module):
    def __init__(self):
        super(NearestSim, self).__init__()
        
    def get_score(self, query, key):
        '''
        query : (NxL) x C or N x C -> T x C  (initial latent features)
        key : M x C     (memory items)
        '''
        qs = query.size()
        ks = key.size()

        score = F.linear(query, key)   # Fea x Mem^T : (TXC) X (CXM) = TxM
        score = F.softmax(score, dim=1) # TxM, 每个查询对 M 个 memory 的概率分布

        return score
    
    def forward(self, queries, items):
        '''
        anchor : query
        positive : nearest memory item
        negative(hard) : second nearest memory item
        queries : N x L x C
        items : M x C
        '''
        batch_size = queries.size(0)
        d_model = queries.size(-1)

        queries = queries.contiguous().view(-1, d_model)    # (NxL) x C >> T x C 展平：T=N*L；得到 T x C
        score = self.get_score(queries, items)      # TxM

        # gather indices of nearest and second nearest item
        _, indices = torch.topk(score, 2, dim=1) # 取“最近”和“次近” memory 的索引（按 softmax 后的概率排名——与按原始点积等价）

        # 1st and 2nd nearest items (l2 normalized)
        # 取第一近邻并做 L2 归一化；同样对 anchor（查询）做 L2 归一化
        pos = F.normalize(items[indices[:, 0]], p=2, dim=-1)  # TxC
        anc = F.normalize(queries, p=2, dim=-1)               # TxC

        # 负的余弦相似度（因为均 L2 归一化，点积=cosine）
        similarity = -1 * torch.sum(pos * anc, dim=-1)         # T
        similarity = similarity.contiguous().view(batch_size, -1)   # N x L
        
        return similarity     # N x L

def sce_loss(x, y, alpha=3):
    x = F.normalize(x, p=1, dim=-1) # L1 归一化（按最后一维），|x|之和=1
    y = F.normalize(y, p=1, dim=-1)

    loss = (1 - (x * y).sum(dim=-1)).pow_(alpha) # 1 - 向量点积（相似度），再做幂次放大

    return loss

