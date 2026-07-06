from math import sqrt
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import numpy as np

from model.embedding import InputEmbedding
from model.loss_functions import sce_loss

from utils.dataplot import plot_time_series_comparison
from model.RevIN import RevIN

'''
上面实现的 Transformer.py 正是 MtsCID 模型的核心网络部分：它搭建了论文中所述的“时间自编码器网络”和“变量间依赖编码器网络”，并实现了正弦原型交互。
简而言之，上分支通过频域线性层和自注意力层学习每个单一变量的时序依赖（粗粒度的时间依赖），下分支通过时域卷积和频域自注意力层学习变量之间的关系（粗粒度的跨变量依赖），并通过固定的正弦原型对这些关系进行进一步聚合。
剩下的重建部分由 Decoder 负责，将隐特征恢复到原始数据空间。
'''

class Decoder(nn.Module):
    # Decoder 类在模型中充当隐空间到原始序列空间的映射，对应于论文中时序自编码器（t-AutoEncoder）部分的最后解码器步骤（公式 (8) 中的 Decoder）
    # 它将融合了时序依赖的信息的隐特征 Z 通过线性映射还原出原始序列格式。
    def __init__(self, w_size, d_model, c_out, networks=['linear'], n_layers=1,
                 group_embedding='False', kernel_size=[1], patch_size=-1, activation='gelu', dropout=0.0, device='cpu'):
        super(Decoder, self).__init__()
        # 接下来是 Decoder 类的定义。该类继承自 nn.Module，表示一个解码器模块，其作用是将模型隐空间中的特征解码（或者映射）回多变量时间序列的输出维度。
        # 类的初始化方法 (__init__) 接受多个参数，其中：
        # w_size：窗口大小，即序列长度 L。
        # d_model：输入的特征维度（即来自编码器输出的通道数）。
        # c_out：输出通道数，通常等于时间序列的变量维度数。
        # networks：一个列表，指示解码器中要使用的网络层类型（如 ['linear'] 表示线性层）。
        # n_layers：表示嵌入层的层数。
        # group_embedding、kernel_size、activation、dropout、device等参数分别控制分组卷积的使用、多尺度卷积核大小、激活函数类型、丢弃率和设备类型等

        self.decoder = InputEmbedding(in_dim=d_model, d_model=c_out, n_window=w_size,
                                      dropout=dropout, n_layers=n_layers,
                                      branch_layers=networks,
                                      match_dimension='last',
                                      group_embedding=group_embedding,
                                      kernel_size=kernel_size, init_type='normal',
                                      device=device)
        # 创建了一个 InputEmbedding 实例，用于实现解码功能。此处调用 InputEmbedding 时，in_dim=d_model 表示输入特征维度是编码器输出的 d_model，而 d_model=c_out 则指定输出维度应为序列的最终通道数 c_out。
        # 同时传入了窗口大小、丢弃率、层数、要使用的网络层列表 networks、匹配维度方法 'last'、分组卷积标志 group_embedding、卷积核大小 kernel_size 以及初始化方式 'normal' 等参数。
        # 简而言之，Decoder 将编码得到的隐特征通过一系列（默认是线性层）映射回时间序列输出空间。

    def forward(self, x):
        """
        x : N x L x C(=d_model)
        """
        out = self.decoder(x)
        return out  # N x L x c_out
        # Decoder 的 forward 方法接收输入 x，其形状为 (N, L, C)，其中 N 是批大小，L 是序列长度（窗口大小 w_size），C=d_model 是输入特征维度。
        # 它直接将 x 传入构建好的 self.decoder，得到解码输出 out。输出 out 的形状为 (N, L, c_out)，即每个时间步对应 c_out 维的重构特征。

class TransformerVar(nn.Module):

    DEFAULTS = {}

    def __init__(self, config, n_heads=1, d_ff=128, dropout=0.3, activation='gelu', gain=0.02):
        super(TransformerVar, self).__init__()
        self.__dict__.update(TransformerVar.DEFAULTS, **config)
        # 接下来是核心的模型类 TransformerVar。该类也是继承自 nn.Module，用于构建整个 MtsCID 模型的双分支网络结构。
        # 类初始化时接收一个 config 字典以及一些可调超参数（如头数 n_heads、前馈网络维度 d_ff、丢弃率 dropout、激活函数 activation、权重初始增益 gain 等）。
        # 在初始化过程中，首先使用 self.__dict__.update 将 DEFAULTS 和给定的 config 合并到类实例中，以便直接通过 self.xxx 访问配置项。

        # Encoding
        branch1_group = self.branches_group_embedding.split('_')[0]
        branch2_group = self.branches_group_embedding.split('_')[1]
        # 在 MtsCID 中，模型采用了双分支架构，上分支用于捕获变数内部的时间依赖（t-AutoEncoder），下分支用于捕获变数间的关系（i-Encoder）。
        # 根据配置 branches_group_embedding（如 "True_False"）来确定每个分支是否采用分组卷积。代码通过字符串拆分得到 branch1_group 和 branch2_group，分别对应上分支和下分支的 group_embedding 设置。

        branch1_dim = self.input_c if self.branch1_match_dimension == 'none' else self.d_model
        branch2_dim = self.input_c if self.branch2_match_dimension == 'none' else self.d_model
        # 接下来，根据 branch*_match_dimension 参数决定每个分支的输入维度与输出维度。
        # 如果匹配方式是 'none'，则输出维度与输入维度相同；否则输出维度使用 d_model。
        # 因此 branch1_dim 和 branch2_dim 分别为上、下分支最后输出的特征维度（通常上分支为 d_model，下分支同样）。

        self.encoder_branch1 = InputEmbedding(in_dim=self.input_c, d_model=branch1_dim, n_window=self.win_size,
                                              dropout=dropout, n_layers=self.encoder_layers,
                                              branch_layers=self.branch1_networks,
                                              match_dimension=self.branch1_match_dimension,
                                              group_embedding=branch1_group,
                                              kernel_size=self.multiscale_kernel_size, init_type=self.embedding_init,
                                              device=self.device)

        self.encoder_branch2 = InputEmbedding(in_dim=self.input_c, d_model=branch2_dim, n_window=self.win_size,
                                              dropout=dropout, n_layers=self.encoder_layers,
                                              branch_layers=self.branch2_networks,
                                              match_dimension=self.branch2_match_dimension,
                                              group_embedding=branch2_group,
                                              kernel_size=self.multiscale_kernel_size,
                                              init_type=self.embedding_init, device=self.device)
        # 上分支的编码器 (encoder_branch1) 和下分支的编码器 (encoder_branch2) 均使用前述的 InputEmbedding 类来构建。
        # 它们的输入通道数均为原始多变量序列的变量数 self.input_c，输出通道数分别为上述计算得到的 branch1_dim 和 branch2_dim。
        # 同时传入窗口大小 n_window=self.win_size、丢弃率 dropout、编码器层数 n_layers=self.encoder_layers、以及具体的层类型列表 branch_layers=self.branch1_networks（上分支网络层）和 self.branch2_networks（下分支网络层）。
        # match_dimension 参数用于控制层间维度匹配策略（例如 'first' 表示第一层变换后输出 d_model 维度）。另外，上下分支的 group_embedding、卷积核大小 kernel_size=self.multiscale_kernel_size、初始化类型 init_type=self.embedding_init、设备 device 等均来自配置。
        # 这里要注意：上分支（t-AutoEncoder） 的层类型列表 branch1_networks 默认为 ['fc_linear', 'intra_fc_transformer', 'multiscale_ts_attention']，表示第一层做频率域的线性映射（对应论文公式 (2) 中的 fc-Linear），第二层做频率域的自注意力变换（fc-Transformer），最后一层做时域的多尺度时序自注意力（ts-Attention）。
        # 下分支（i-Encoder） 的层类型列表 branch2_networks 默认为 ['multiscale_conv1d', 'inter_fc_transformer']，即先对原始时序进行一维卷积（捕获局部时间依赖），然后在频率域进行跨变量的自注意力（inter_fc-Transformer）。这些设置严格对应了论文第2.3节和2.4节中所述的 t-AutoEncoder 和 i-Encoder 的流程

        self.activate_func = nn.GELU()
        self.dropout = nn.AlphaDropout(p=dropout)
        self.loss_func = nn.MSELoss(reduction='none')
        self.mem_R, self.mem_I = create_memory_matrix(N=branch2_dim,
                                                      L=self.win_size,
                                                      mem_type=self.memory_guided,
                                                      option='options2')
        # 模型使用 GELU 激活函数并配置了 AlphaDropout。损失函数暂设为均方误差（MSE），这用于计算时间序列重建误差。
        # 接着，根据下分支输出特征维度 branch2_dim 和窗口长度 self.win_size，调用 create_memory_matrix 创建用于原型模块的记忆矩阵 mem_R, mem_I。
        # 其中 mem_type=self.memory_guided 指定原型初始化方式（默认 sinusoid，见第2.5节），option='options2' 指定备选方案类型（当前代码中未处理该选项）。
        # 该函数最终返回两个矩阵：mem_R（基于余弦函数的原型）和 mem_I（基于正弦函数的原型）

        branch1_out_dim = self.output_c if self.branch1_match_dimension == 'none' else self.d_model

        model_dim = branch1_out_dim

        self.weak_decoder = Decoder(w_size=self.win_size,
                                    d_model=model_dim,
                                    c_out=self.output_c,
                                    networks=self.decoder_networks,
                                    n_layers=self.decoder_layers,
                                    group_embedding=self.decoder_group_embedding,
                                    kernel_size=self.multiscale_kernel_size,
                                    activation='gelu',
                                    dropout=0.0,       # The dropout in decoder is set as zero
                                    device=self.device)

        if self.branch1_match_dimension == 'none':
            self.feature_prj = lambda x: x
        else:
            self.feature_prj = nn.Linear(branch1_out_dim, self.output_c)
            # 接下来根据上分支的输出是否需要降维，定义 feature_prj 投影层。如果 branch1_match_dimension 为 'none'，则特征维度不变，feature_prj 设为恒等函数；
            # 否则用一个线性层将上分支输出（branch1_out_dim 维）投影到最终的输出通道数 self.output_c。

        for m in self.modules():
            if isinstance(m, nn.Linear):
                if self.embedding_init == 'normal':
                    torch.nn.init.normal_(m.weight.data, 0.0, gain)
                elif self.embedding_init == 'xavier':
                    torch.nn.init.xavier_normal_(m.weight.data, gain=gain)
                elif self.embedding_init == 'kaiming':
                    torch.nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
                elif self.embedding_init == 'orthogonal':
                    torch.nn.init.orthogonal_(m.weight.data, gain=gain)
                else:
                    torch.nn.init.uniform_(m.weight.data, a=-0.5, b=0.5)

                if hasattr(m, 'bias') and m.bias is not None:
                    torch.nn.init.constant_(m.bias.data, 0.0)

            elif isinstance(m, nn.Conv1d) or isinstance(m, nn.ConvTranspose1d):
                if self.embedding_init == 'normal':
                    torch.nn.init.normal_(m.weight.data, 0.0, gain)
                elif self.embedding_init == 'xavier':
                    torch.nn.init.xavier_normal_(m.weight.data, gain=gain)
                elif self.embedding_init == 'kaiming':
                    torch.nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
                elif self.embedding_init == 'orthogonal':
                    torch.nn.init.orthogonal_(m.weight.data, gain=gain)
                else:
                    torch.nn.init.uniform_(m.weight.data, a=-0.5, b=0.5)

                if hasattr(m, 'bias') and m.bias is not None:
                    torch.nn.init.constant_(m.bias.data, 0.0)
                # 然后进行模型参数的初始化。遍历 self.modules() 中的所有子模块，对线性层 (nn.Linear) 和卷积层 (nn.Conv1d 或 nn.ConvTranspose1d) 的权重进行初始化。
                # 根据配置 self.embedding_init 选择不同方式（如正态分布、Xavier、Kaiming、正交或均匀），偏置初始化为零。此步骤保证模型权重在训练开始时满足一定分布条件。
                # hasattr(object, name) 是 Python 的内置函数，用于检查对象是否具有指定名称的属性或方法。
                # object：要检查的对象
                # name：字符串形式的属性名
                # 返回值：如果对象有该属性返回 True，否则返回 False

    def forward(self, input_data, mode='train'):
        # 接下来是 forward 方法。它的输入 input_data 形状为 (B, L, enc_in)，即批大小 B，序列长度 L，每个时间步的特征维度为 enc_in。
        # 在 MtsCID 中，enc_in 通常等于输入变量数（input_c）。forward 中首先将输入复制给 z1 和 z2，分别进入上分支和下分支进行并行处理。
        """
        x (input time window) : B x L x enc_in
        """

        z1 = z2 = input_data
        t_query, t_latent_list = self.encoder_branch1(z1)
        i_query, _ = self.encoder_branch2(z2)
        # 对 z1（上分支输入），调用 self.encoder_branch1(z1)。根据前面 InputEmbedding 的实现，这一步会依次执行上分支各层操作，并返回两个值：t_query 和 t_latent_list，
        # 其中 t_query 是最后一层输出的特征（形状 (B, L, branch1_dim)），t_latent_list 是每层（经过归一化后的）中间特征列表。上分支的输出 t_query 表示经过时域和频域联合编码后的隐特征（对应论文第2.3节生成的 Z）。
        # 对 z2（下分支输入），调用 self.encoder_branch2(z2)，得到 i_query 和 _（这里我们忽略中间特征列表）。下分支输出 i_query 的形状也是 (B, L, branch2_dim)，它表示每个时间步在考虑了跨变量关系后的特征表征（对应论文第2.4节最终输出 O ￼）。

        # use dot production with static sinusoid basis
        mem = self.mem_R.T.to(self.device)
        # 下一步，使用之前创建的记忆矩阵 self.mem_R。mem_R 原本是 (N, L) 形状，其中 N = branch2_dim 是原型的个数，L 是时间步长度。
        # 代码将其转置并移动到当前设备上：mem = self.mem_R.T.to(self.device)，于是 mem 的形状为 (L, N)。这将作为原型向量集合，用于下分支与这些“正弦原型”间的交互。

        # differencing_q = (i_query - torch.roll(i_query, shifts=1, dims=-2))
        # It seems that using differencing is better than using i_query
        attn = torch.einsum('blf,jl->bfj', i_query, self.mem_R.to(self.device).detach())
        attn = torch.softmax(attn / self.temperature, dim=-1)
        # 然后计算注意力权重 attn：将下分支的查询 i_query（形状 (B, L, f)，其中 f=branch2_dim）与原型 self.mem_R（形状 (f, L)）做点积，采用爱因斯坦求和表示 torch.einsum。
        # 这里的逻辑相当于对每个批次和每个特征维度，将 i_query[b, :, :] 与 mem_R 做矩阵乘法，结果 attn 的形状为 (B, f, N)。随后除以温度并进行 softmax（在最后一个维度上），得到概率分布形式的注意力权重。
        # 从 MtsCID 论文第2.5节可知，这一步对应于将 i-Encoder 的输出与固定的正弦原型（记忆项）做交互来得到权重，公式 (13) 描述了 $\exp(\langle O_{:,t,:}, M_{i,:}\rangle/\tau)$ 的计算。
        # 这里 attn 即对应公式中的权重 $\omega_{ti}$（在进行 softmax 归一化之前）。

        queries = i_query
        combined_z = t_query
        combined_z = self.feature_prj(combined_z)
        out, _ = self.weak_decoder(combined_z)
        # 接着将上分支输出 t_query 赋值给 combined_z，再通过 feature_prj 进行投影（如果需要变换维度）。
        # combined_z 的形状仍为 (B, L, model_dim)，其中 model_dim 代表要传入解码器的特征维度。最后调用弱解码器（self.weak_decoder），将 combined_z 解码回输出空间。
        # 解码器的返回值包含重构的时间序列 out（形状 (B, L, c_out)）和潜在特征列表（此处忽略）。

        return {"out": out, "queries": queries, "mem": mem, "attn": attn}
        # 最终，forward 方法返回一个字典，包括："out" —— 解码得到的重构序列；"queries" —— 下分支的查询特征 i_query；"mem" —— 原型矩阵 mem；以及 "attn" —— 交互得到的注意力权重。
        # 这些输出用于计算损失和推理阶段的异常评分，其中重构误差对应时间依赖学习（t-AutoEncoder）的重构任务损失，注意力权重和原型相关的项对应原型学习损失（Prototype-Oriented Learning Task） ￼

    def get_attn_score(self, query, key, scale=None):
        # get_attn_score，用于计算注意力得分（带缩放因子），它直接实现了经典的点积注意力得分：
        # 将 query（形状如 (T\times C) 或 (N\times C)）与 key（形状如 (M\times C)）做矩阵乘法，然后乘以缩放因子 1/\sqrt{C}。
        # 该函数会返回未经过 softmax 的注意力矩阵（形状 (T, M)），可用于自定义稀疏化策略等，但在当前实现中没有在 forward 中使用。
        # sqrt(query.size(-1)) 保证了点积规模正常化（对应 Transformer 中的缩放因子） ￼

        """
        Calculating attention score with sparsity regularization
        query (initial features) : (NxL) x C or N x C -> T x C
        key (memory items): M x C
        """
        scale = 1. / sqrt(query.size(-1)) if scale is None else 1. / scale
        attn = torch.matmul(query, torch.t(key.to(self.device)))  # (TxC) x (CxM) -> TxM
        attn = attn * scale

        # attn = F.softmax(attn / self.temperature, dim=-1)
        # attn = torch.einsum('tl,kfl->tkf', query, key.to(self.device))  # (TxC) x (CxM) -> TxM
        # attn = attn.max(dim=1)[0]

        return attn

def generate_rolling_matrix(input_matrix):
    # generate_rolling_matrix(input_matrix)：给定一个形状为 (F, L) 的矩阵，返回一个形状 (L, F, L) 的张量。
    # 其中第 step 个矩阵切片是对输入矩阵沿最后一个维度右移 step 位得到的结果。这用于构造循环移位的原型集合（尽管在 create_memory_matrix 中仅在特定选项 'option4' 下使用，本代码默认没有用到）

    F, L = input_matrix.size()
    # 输入 input_matrix 为 (F, L)——F 个特征（或记忆条目），L 个时间步

    output_matrix = torch.empty(L, F, L)
    # 在 PyTorch 中预先分配一个空的 3D 张量用于存放结果，形状为 (L, F, L)，维度含义为 (移位索引, 特征/原型, 时间步)

    # Iterate over each step from 0 to L-1
    for step in range(L):
        # 在时间维度上进行循环平移。例如 step=1 时，原型的每一行向右移一位，最右边的元素回到开头。这样生成 L 个不同相位的原型矩阵
        rolled_matrix = input_matrix.roll(shifts=step, dims=1)
        # 将移位后的矩阵赋值到输出张量的第 step 个切片 将移位后的矩阵放到输出张量的相应位置，最终 output_matrix 包含了所有可能的循环移位。
        output_matrix[step] = rolled_matrix
    return output_matrix

def create_memory_matrix(N, L, mem_type='sinusoid', option='option1'):
    # 该函数负责根据不同策略生成一组固定的记忆矩阵，这些矩阵在整个模型训练过程中保持不更新（通过 torch.no_grad()）。参数含义：
    # N：原型数，即模型需要的记忆条目的数量，通常对应于变量数或隐藏维度。
    # L：时间步长，决定每个原型的长度。
    # mem_type：初始化方式，可选 'sinusoid' (正弦/余弦原型)、'uniform' (均匀随机)、'orthogonal_uniform' (正交随机)、'normal' (正态随机)、'orthogonal_normal' (正交正态随机) 以及这些类型后缀 '_only' 的变体。
    # option：附加处理选项；只有当 option == 'option4' 时才会调用 generate_rolling_matrix，其余情况下按默认初始化返回。源码中调用时未启用 option4，因此默认不会滚动波形。
   
    with torch.no_grad():
        # 使用 no_grad() 保护内层操作不被梯度跟踪，保证生成的记忆矩阵在训练过程中保持常量（不参与反向传播）

        if mem_type  == 'sinusoid' or mem_type  == 'cosine_only':
            row_indices = torch.arange(N).reshape(-1, 1) # 生成形状为 (N, 1) 的行索引向量
            col_indices = torch.arange(L) # 生成长度为 L 的列索引向量
            grid = row_indices * col_indices # 通过广播得到一个 (N, L) 的整数网格

            # Calculate the period values using the grid
            init_matrix_r = torch.cos((1 / L) * 2 * torch.tensor([torch.pi]) * grid)
            # 计算余弦矩阵，其公式正好对应论文中固定记忆项的定义 \cos\bigl(\tfrac{2\pi}{L},i,j\bigr) 这里乘以 2*torch.tensor([torch.pi]) 相当于 $2\pi$。
            init_matrix_i = torch.sin((1 / L) * 2 * torch.tensor([torch.pi]) * grid)
            # 计算正弦矩阵，与余弦矩阵互补。从命名看，r 表示“实部”，i 表示“虚部”。在复数形式中 $e^{j\theta} = \cos\theta + j\sin\theta$，这种表示可以看作使用复数正弦波作为原型，尽管后续代码主要用到了余弦部分。

        elif mem_type  == 'uniform' or mem_type  == 'uniform_only':
            init_matrix_r = torch.rand((N, L), dtype=torch.float)
            init_matrix_i = torch.rand((N, L), dtype=torch.float)
        elif mem_type  == 'orthogonal_uniform' or mem_type  == 'orthogonal_uniform_only':
            init_matrix_r = torch.nn.init.orthogonal_(torch.rand((N, L), dtype=torch.float))
            init_matrix_i = torch.nn.init.orthogonal_(torch.rand((N, L), dtype=torch.float))
        elif mem_type  == 'normal' or mem_type  == 'normal_only':
            init_matrix_r = torch.randn((N, L), dtype=torch.float)
            init_matrix_i = torch.randn((N, L), dtype=torch.float)
        elif mem_type  == 'orthogonal_normal' or mem_type  == 'orthogonal_normal_only':
            init_matrix_r = torch.nn.init.orthogonal_(torch.randn((N, L), dtype=torch.float))
            init_matrix_i = torch.nn.init.orthogonal_(torch.randn((N, L), dtype=torch.float))

        # rolling the wave
        if option == 'option4':
            init_matrix_r = generate_rolling_matrix(init_matrix_r)
            init_matrix_i = generate_rolling_matrix(init_matrix_i)
            # 调用 generate_rolling_matrix 分别对实部和虚部进行循环移位，返回的 init_matrix_r、init_matrix_i 形状从 (N, L) 变为 (L, N, L)。
            # 这种处理会产生每个原型不同相位的版本，可能用于让模型在不同时间相位上共享原型，增强时移不变性；但默认配置未使用该选项。

        if 'only' not in mem_type:
            return init_matrix_r, init_matrix_i
            # 即同时包含余弦与正弦部。模型随后可以将其看作一对复数原型
        else:
            return init_matrix_r, torch.zeros_like(init_matrix_r)
            # 如果 mem_type 包含 'only'，则返回只有实部，虚部设为零。这样可在实验中评估单独使用某种基函数的效果。
            
        # create_memory_matrix(N, L, mem_type='sinusoid', option='option1')：创建原型记忆矩阵。根据 mem_type 参数，可以选择多种初始化方式。
        # 当 mem_type 为 'sinusoid' 时，代码使用余弦和正弦函数生成矩阵：init_matrix_r[i,j] = cos((2π/L) * i * j)，init_matrix_i[i,j] = sin((2π/L) * i * j)，
        #   其中 $i=0,\dots,N-1$ 是原型索引，$j=0,\dots,L-1$ 是时间步索引（正如论文式(12) 所示）。
        # 其他选项（如随机、正交等）也支持。若选择了 'only' 后缀，则返回只有实部或虚部的原型矩阵，以及对应形状的零矩阵。