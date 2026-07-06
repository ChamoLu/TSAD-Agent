import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from math import sqrt
import os

def complex_operator(net_layer, x):
    '''
    当 x 为实数张量时，complex_operator 简化为普通的前向计算，相当于不区分实部虚部，直接返回网络层对输入的转换结果。
    当输入是复数张量时，这段代码确保对实部和虚部分别应用神经网络层，再将结果组合成复数。这对应于 MtsCID 论文中所述的频域双网络处理策略：
    例如在频域的 fc-Linear 和 fc-Transformer 模块中，实部和虚部由独立的网络参数进行变换。
    complex_operator 提供了一种统一接口，如果模型需要对复数信号进行线性变换或通过RNN/LSTM处理，就可以分别对实部和虚部执行相同结构的操作（如两个并行线性层或两个并行 LSTM），然后将结果合成为复数输出。
    这样可以保持复数信号的相位和幅值信息分别由对应网络处理，符合复数运算规则并避免直接对复数张量操作的不支持。
    简单来说，本函数实现了公式：(a+bi) ⟶ f(a) + i·g(b)，其中 f 和 g 是对应的网络变换，用以模拟对复数输入进行等价于对实部虚部分别处理的效果。
    '''
    if not torch.is_complex(x):
        return net_layer[0](x) if isinstance(net_layer, nn.ModuleList) else net_layer(x)
        # 定义函数 complex_operator(net_layer, x)，用于对输入 x 执行给定的网络层 net_layer，同时兼容实数和复数类型张量。
        # 使用 torch.is_complex(x) 判断输入张量是否为复数类型。如果 x 不是复数（即实数张量），根据 net_layer 的类型直接返回其对 x 的输出：
        # 如果传入的 net_layer 是一个 nn.ModuleList（模块列表），则取其第一个子层对 x 进行计算并返回结果；否则（net_layer 是单一模块)，直接调用 net_layer(x) 并返回输出。

    else:
        if isinstance(net_layer[0], nn.LSTM):
            return torch.complex(net_layer[0](x.real)[0], net_layer[1](x.imag)[0]) if isinstance(net_layer, nn.ModuleList) else torch.complex(net_layer(x.real)[0], net_layer(x.imag)[0])
        else:
            return torch.complex(net_layer[0](x.real), net_layer[1](x.imag)) if isinstance(net_layer, nn.ModuleList) else torch.complex(net_layer(x.real), net_layer(x.imag))
        # 输入 x 为复数张量的情况。进一步判断 net_layer[0] 是否为 nn.LSTM 类型，如果是，则按照 LSTM 的输出格式进行特殊处理：
        # 构造复数输出，其中实部由 net_layer[0] 作用于 x 的实部后得到（net_layer[0](x.real)[0] 表示LSTM对实部序列输出的隐藏状态序列），虚部由 net_layer[1] 作用于 x 的虚部得到，然后用 torch.complex(实部结果, 虚部结果) 合成为复数张量。
        # 如果传入的 net_layer 不是 ModuleList（即单个 LSTM 模块)，则通过单个 LSTM 分别处理实部和虚部（net_layer(x.real)[0] 和 net_layer(x.imag)[0]），再合成为复数输出。
        # 非 LSTM 的一般网络层：如果 net_layer 是 ModuleList，则用第0个子模块处理 x.real、第1个子模块处理 x.imag，否则就用同一个 net_layer 分别作用于实部和虚部，最后将两部分用 torch.complex 合成为一个复数张量返回。

def complex_einsum(order, x, y):
    '''
    该函数实现了爱因斯坦求和的复数扩展。对普通实数张量，torch.einsum 已能方便地根据指定模式计算多维矩阵乘法或内积。而对于复数张量，我们需要对实部和虚部分别计算并组合结果。上面的计算公式正是复数点积/矩阵乘法的实现。
    例如，在注意力分数计算中，我们需要 Q·KT（点积形式），如果 Q、K 是复数张量，那么采用此函数即可正确地计算其点积的复数结果，实部为 Q_realK_realT - Q_imagK_imagT，虚部为 Q_realK_imagT + Q_imagK_realT，这样复数信号的内积结果保持相位信息不丢失。
    若 Q 和 K 本就是实数，则走 else 分支，相当于普通矩阵乘法。这个设计保证了频域注意力计算的正确性：MtsCID 在频域的 Transformer 操作需要对复数频谱应用自注意力，complex_einsum 提供了底层支持。
    '''
    x_flag = True
    y_flag = True
    if not torch.is_complex(x):
        x_flag = False
        x = torch.complex(x, torch.zeros_like(x).to(x.device))
    if not torch.is_complex(y):
        y_flag = False
        y = torch.complex(y, torch.zeros_like(y).to(y.device))
        # 定义爱因斯坦求和函数 complex_einsum(order, x, y)，用于计算两个张量 x 和 y 按指定爱因斯坦求和公式 order 相乘的结果，并兼容复数。
        # 初始化标志 x_flag 和 y_flag 为 True，表示默认假设 x 或 y 是复数。如果判断 x 不是复数（实数张量），则将 x_flag 置False，并将 x 转换为复数形式：使用 torch.complex(x, torch.zeros_like(x).to(x.device)) 构造一个虚部为0的复数张量表示。这相当于把实数 x 转为复数表示（实部为自身，虚部全零）。
        # 对 y 执行类似的操作：若 y 非复数则置 y_flag=False 并转换成虚部为0的复数张量。经过这一步，如果原始输入中任一是实数，都被扩展为复数形式参与后续计算。

    if x_flag or y_flag:
        return torch.complex(torch.einsum(order, x.real, y.real) - torch.einsum(order, x.imag, y.imag),
                             torch.einsum(order, x.real, y.imag) + torch.einsum(order, x.imag, y.real))
    else:
        return torch.einsum(order, x.real, y.real)
        # 经过上一步，x 和 y 均已是复数张量（若原本为实数则已转换）。判断 x_flag or y_flag 是否为真：如果任一原始输入是复数（即至少一个 flag 没被置False），表示我们需要执行复数乘法逻辑。
        # 返回一个复数张量，实部通过 torch.einsum(order, x.real, y.real) - torch.einsum(order, x.imag, y.imag) 计算，虚部通过 torch.einsum(order, x.real, y.imag) + torch.einsum(order, x.imag, y.real) 计算。
        # 这正是复数乘法的规则：(a + bi) * (c + di) = (ac - bd) + (ad + bc)i，在张量形式下通过爱因斯坦求和实现。
        # 其中 order 是一个爱因斯坦求和字符串，如代码中稍后会用 'nlhd,nshd->nhls' 等形式，用于指定张量维度的乘法匹配关系；PyTorch 的 einsum 会根据 order对对应维度进行乘积求和。
        # 与之对应，else 分支：当 x_flag 和 y_flag都为 False 时表示两者原始都是实数张量，在这种情况下无需复杂的复数计算，直接返回 torch.einsum(order, x.real, y.real) 即可（此时 x.real 和 x 相同，因为 x 是纯实数转成的复数）。

def complex_softmax(x, dim=-1):
    '''
    Softmax 是将一个向量通过指数映射归一化为概率分布的操作。对于注意力机制，softmax 用于将点积得到的注意力分数转化为权重（归一化的注意力矩阵）。
    然而严格来说复数并没有直接定义softmax，这里的实现采取了一种近似方案：对复数的实部和虚部分别执行softmax。这意味着分别在实部和虚部上做指数归一化处理。
    这样做在数学上并非标准的复数softmax（因为softmax要求实数域下求和为1），但可能是一种经验性的处理，认为实部、虚部各自代表某种相关的注意力权重分布。
    这种实现至少保证了输出仍是形状相同的复数张量，并且在实部和虚部各自满足类似概率分布的归一化特性。在 MtsCID 中，频域Transformer的 QKT 结果往往是复数矩阵，通过该函数实质上对实部和虚部分别归一，可突出重要的频率相关性。需要注意，如果输入是纯实数则正常返回softmax结果。
    '''
    if not torch.is_complex(x):
        return torch.softmax(x, dim=dim)
    else:
        return torch.complex(torch.softmax(x.real, dim=dim), torch.softmax(x.imag, dim=dim))
    # 定义复数版本的 softmax 函数。检查输入 x 是否为复数张量。如果 x 不是复数，则直接使用普通的 torch.softmax(x, dim=dim) 对指定维度 dim 上的元素进行 softmax 运算并返回结果。
    # 若 x 是复数，则分别对其实部和虚部执行 softmax，即实部的softmax结果作为新复数的实部，虚部softmax结果作为新复数的虚部。

def complex_dropout(dropout_func, x):
    '''
    Dropout是一种正则化手段，通过在训练时随机将部分神经元输出置零来防止过拟合。对于复数张量，理想情况下希望对实部和虚部同步地执行相同的dropout掩码（即同一个元素的实部、虚部要么都保留，要么都置零），以免破坏复数的结构。
    然而 PyTorch 并没有直接支持对复数这样操作的dropout。这段代码最初的注释实现想分别对实部和虚部应用 dropout。但是若各自独立随机，会导致实部某元素被置零而虚部不置零，破坏复数对应关系；如果想让实部和虚部使用相同mask，则需要自定义mask。在当前实现中，作者选择直接跳过复数上的dropout（返回原值），避免了处理复杂的掩码同步问题。
    虽然这意味着对复数数据不执行失活正则化，但考虑到复数部分主要出现于频域Transformer中，可能出于训练稳定性的考虑（因为频谱的实部虚部具有耦合关系），不对其强制随机置零。简而言之，当输入是复数时，该函数相当于一个空操作，保持信号完整；当输入是实数时则正常应用dropout。这是一个设计折中，确保复数信号的幅值和相位不被不一致地干扰。
    '''
    if not torch.is_complex(x):
        return dropout_func(x)
    else:
        # return torch.complex(dropout_func(x.real), dropout_func(x.imag))
        return torch.complex(x.real, x.imag)
    # 定义兼容复数的 dropout 随机失活函数。判断 x 是否为复数张量，如果不是则直接调用传入的 dropout_func(x) 执行常规的 dropout 操作（dropout_func 由外部传入，例如 PyTorch 的 nn.Dropout 已实例化对象，其调用会随机将部分元素置零）。
    # 若 x 是复数，则进入 else 分支。这里第50行的代码被注释掉了，原意可能是使用 torch.complex(dropout_func(x.real), dropout_func(x.imag)) 对实部和虚部分别应用相同的dropout函数再组合。
    # 但目前实现直接返回 torch.complex(x.real, x.imag)，也就是对复数张量不施加任何丢弃，维持原值返回。

def complex_layernorm(norm_func, x):
    '''
    层归一化（Layer Normalization）在Transformer中常用于稳定和加速收敛，通过对每个样本的特征向量减均值除以方差，将其归一。对于复数数据，这里类似地分别归一化实部和虚部。
    这样处理隐含假设实部和虚部各自的分布可以独立归一化。在MtsCID的频域处理部分，曾提及对变换后的表示应用残差连接和层归一化。由于当时频域表示可能是复数（或分别处理的实、虚部分），此函数提供了一个简单途径：对实部虚部各自做LayerNorm。
    例如，如果 x = a + bi，则将 a 和 b 两个实张量各自归一化再组合成新的复数。这有助于维持复数信号的数值稳定性，又保留相对位置信息。不过，需要注意这样做没有考虑实部虚部之间的协方差，但目前这是常用的实用处理方式。
    '''
    if not torch.is_complex(x):
        return norm_func(x)
    else:
        return torch.complex(norm_func(x.real), norm_func(x.imag))
    # 定义兼容复数的层归一化函数。判断输入 x 是否为复数，如果不是则直接调用传入的 norm_func(x) 返回（例如 nn.LayerNorm 实例，对实数张量按最后一维做归一化）。
    # 如果 x 是复数，则返回 torch.complex(norm_func(x.real), norm_func(x.imag))，即将实部和虚部分别经过层归一化后再组成复数结果。

class PositionalEmbedding(nn.Module):
    '''
    这些公式来源于Transformer论文的位置编码设计【Vaswani et al. 2017】。偶数维使用正弦，奇数维使用余弦，并在指数中使用 $\frac{1}{10000^{2i/d_model}}$ 控制频率跨度。
    这样对于序列中不同的位置，会产生一个在各维度呈周期性变化的向量。模型可以依据这些周期信号推断出位置顺序关系。代码按公式逐元素填充，确保位置编码矩阵 pe 的每一行对应一个位置，每一列对应一个embedding维度，其中值按照正弦/余弦规律变化。
    升维后，pe 可与输入张量形状对齐。需要注意，这里的 pe 是在 __init__ 中一次性算好的常量，不会学习。位置编码在MtsCID 中用于在Transformer中注入时间步的位置信息，虽然下面看到具体调用时被注释掉了，但仍提供了可选的绝对位置信息输入方式，以帮助注意力机制区分不同时间索引。
    '''
    def __init__(self, d_model, max_len=5000):
        super(PositionalEmbedding, self).__init__()
        self.pe = torch.zeros((max_len, d_model), dtype=torch.float)
        self.pe.requires_grad = False

        pos = torch.arange(0, max_len).float().unsqueeze(1)
        _2i = torch.arange(0, d_model, step=2).float()
        # 定义位置嵌入（Positional Encoding）类，用于给序列加入位置信息。
        # 初始化函数 __init__：参数 d_model 是模型的嵌入维度，max_len 是支持的最大序列长度。首先创建大小为 (max_len, d_model) 的零张量 self.pe，类型为浮点，并将 requires_grad=False，表示这些位置编码不参与训练更新（它是固定的位置特征）。
        # 生成位置索引张量 pos = torch.arange(0, max_len).float().unsqueeze(1)，形状为 (max_len, 1)，每一行是一个位置索引0,1,2,…。
        # 生成 _2i = torch.arange(0, d_model, step=2).float()，即从0到d_model按步长2取值的张量，代表偶数维度索引（0,2,4,…）用于计算正弦和余弦。

        self.pe[:, ::2] = torch.sin(pos / (10000 ** (_2i / d_model)))
        if d_model % 2 == 0:
            self.pe[:, 1::2] = torch.cos(pos / (10000 ** (_2i / d_model)))
        else:
            self.pe[:, 1::2] = torch.cos(pos / (10000 ** (_2i / d_model)))[:, :-1]

        self.pe = self.pe.unsqueeze(0)
        # 利用公式填充位置编码张量的偶数列：self.pe[:, ::2] = torch.sin(pos / (10000 ** (_2i / d_model)))。
        # 这里对每个位置 pos 计算一个周期函数：分母 10000 ** (_2i / d_model) 随着维度索引增长而指数级增大，使得正弦函数在不同维度具有不同的波长，实现多频率编码。
        # 当 d_model 为偶数的情况，将奇数列填入 cos 值：self.pe[:, 1::2] = torch.cos(pos / (10000 ** (_2i / d_model)))。
        # 如果 d_model 是奇数：由于 _2i 生成的是偶数索引列表，最后一个奇数维度没有对应 cos，代码通过 [:, :-1] 截断确保形状匹配，再赋值余弦值。
        # 第75行将 self.pe 升维为 (1, max_len, d_model)。这样在batch维度上扩充了一维（值重复），便于之后直接加到输入张量上。

    def forward(self, x):
        return self.pe[:, :x.size(1)]
        # 定义前向函数 forward(self, x)，传入张量 x（通常是形状 (N, L, d_model) 的序列输入）。
        # 返回预先计算好的位置编码张量中前 L 个位置对应的部分：self.pe[:, :x.size(1)]，其形状为 (1, L, d_model)。通过广播机制，这个张量可加到输入上，为每个批次样本的每个时间步添加固定的位置向量。

class Attention(nn.Module):
    def __init__(self, window_size, mask_flag=False, scale=None, dropout=0.0):
        super(Attention, self).__init__()
        self.window_size = window_size
        self.mask_flag = mask_flag
        self.scale = scale
        self.dropout = nn.Dropout(p=dropout)
        # 定义注意力计算核心类 Attention。 __init__ 方法初始化，参数包括 window_size（窗口大小）、mask_flag（是否使用掩码标志）、scale（缩放因子，可选）和 dropout（dropout比率）。
        # 将这些参数保存为对象属性，其中 self.window_size 存储窗口大小，self.mask_flag 存储是否使用mask，self.scale 保存传入的缩放系数（若为None则稍后会自定计算），self.dropout 定义了一个 nn.Dropout(p=dropout) 层用于对注意力权重进行随机失活。

    def forward(self, queries, keys, values, attn_mask=None):
        '''
        queries : N x L x Head x d
        keys : N x L(s) x Head x d
        values : N x L x Head x d
        '''
        N, L, Head, C = queries.shape
        # 输入张量的形状：queries : N x L x Head x d，keys : N x L(s) x Head x d，values : N x L(s) x Head x d（代码中第91-92行有L(s)表示keys可能长度为L或不同长度L(s)）。
        # 这里 N 是批大小，L 是查询序列长度，Head 是注意力头数，d 是每个头的向量维度。可选参数 attn_mask 提供注意力掩码（用于屏蔽某些不该参与计算的位置）。
        # 通过 N, L, Head, C = queries.shape 解包获取 queries 张量的维度。其中 N是批次大小，L是查询序列长度，Head是多头数，C是每个head的通道维度（d）。

        scale = self.scale if self.scale is not None else 1. / sqrt(C)
        # 计算缩放系数 scale：如果 self.scale 在初始化时已提供，则使用它；否则默认使用 1/sqrt(C)。

        attn_scores = complex_einsum('nlhd,nshd->nhls', queries, keys) # N x Head x L x L
        # 调用 complex_einsum('nlhd,nshd->nhls', queries, keys) 计算注意力分数。这里 'nlhd,nshd->nhls' 是爱因斯坦求和的模式字符串：n表示batch维，l表示queries长度L，s表示keys长度L(s)，h表示注意力头维，d表示每头维度。
        # 该模式令 PyTorch 对 queries 和 keys 张量进行矩阵乘法：对相同的 n（批次）和 h（头）下，l 和 s 两个维度相乘并在 d 上求和，得到输出的 nhls 维度顺序。
        # 简单理解，这相当于对每个头，将 Q (形状 N×L×d) 与 K (形状 N×L(s)×d) 按最后一维做矩阵乘，计算出大小为 N×L×L(s) 的注意力分数矩阵，并保留多头维度 h 在结果中。因此 attn_scores 的形状是 (N, Head, L, L(s))

        attn_weights = complex_dropout(self.dropout, complex_softmax(scale * attn_scores, dim=-1))
        # 将前一步得到的 attn_scores 进行缩放和归一化，得到注意力权重矩阵 attn_weights。
        # 具体实现为：complex_softmax(scale * attn_scores, dim=-1) 先将 attn_scores 乘以上面计算的缩放系数 scale（对整个张量逐元素相乘），然后在最后一维 (dim=-1 对应 keys序列长度维度L(s)) 上执行softmax。
        # 由于我们可能处理复数，这里用了前面定义的 complex_softmax，从而对实部、虚部分别softmax。如果 attn_scores 是实数张量，那么效果等同于标准softmax。
        # softmax 输出的结果维度仍是 N×Head×L×L(s)，每个L(s)维上元素之和为1（对于复数则实部、虚部分别归一）。接着，用 complex_dropout(self.dropout, ... ) 对 softmax 结果施加 dropout。
        # self.dropout 是一个Dropout层对象，其 p 等于初始化设定的dropout率。complex_dropout 会检查输入是否复数并相应处理：对于实数权重，它实际会随机置零部分注意力权重；如果是复数权重，由于我们实现中跳过复数的dropout，基本不改变值（保持与softmax输出相同）。最终结果赋给 attn_weights。

        updated_values = complex_einsum('nhls,nshd->nlhd', attn_weights, values)  # N x L x Head x d
        # 通过另一项爱因斯坦求和将注意力权重应用到数值矩阵 V 上，以计算输出的“上下文”向量。调用 complex_einsum('nhls,nshd->nlhd', attn_weights, values) 得到 updated_values。
        # 这个模式 'nhls,nshd->nlhd' 代表：对相同批次 n 和头 h 下，将注意力权重 (n,h,l,s) 与值 values (n,s,h,d) 在 s 维进行相乘并求和，输出在 l（query位置）和 d（每头维度）上。
        # 直观理解，对于每个注意力头 h，每个查询位置 l，updated_values[n, l, h, d] = sum_{s}( attn_weights[n, h, l, s] * values[n, s, h, d] )。
        # 也就是说，针对每个 query 时间步，按softmax得到的权重组合对应的所有 value 向量，加权求和得到输出向量。计算结果形状为 N×L×Head×d，和 queries 保持一致。
        # 由于使用 complex_einsum，如果 values 是复数张量，则这里进行了复数域的加权和（实部和虚部分别正确加权求和）。

        return updated_values.contiguous(), attn_weights.permute(0, 2, 1, 3).mean(dim=-2)
        # 返回一个元组，其中第一个元素是 updated_values.contiguous()，第二个元素是处理后的 attn_weights。
        # updated_values.contiguous()将张量在内存中变得连续（contiguous），以防后续需要view或transpose（这是PyTorch常见操作，确保内存布局适合连续访问）。
        # attn_weights.permute(0, 2, 1, 3).mean(dim=-2) 处理注意力权重：首先 permute(0,2,1,3) 将维度顺序从 (N, Head, L, S) 调整为 (N, L, Head, S)，也就是把注意力矩阵变换为每个样本的每个查询时间步对应 [Head×Key] 的矩阵，然后 .mean(dim=-2) 在 Head 维上取平均（-2 指倒数第二维，这里对应 Head 维）。
        # 这样得到的结果形状为 (N, L, S)，是对所有注意力头的权重平均后的单头注意力矩阵。
        # 最终 forward 返回 (updated_values, attn_avg)，其中 attn_avg 即上述平均后的注意力矩阵。

class AttentionLayer(nn.Module):
    def __init__(self, w_size, d_model, n_heads, d_keys=None, d_values=None, mask_flag=False,scale=None, dropout=0.0):
        super(AttentionLayer, self).__init__()

        n_heads = n_heads if (d_model % n_heads) == 0 else 1

        z = d_model % n_heads if (d_model // n_heads) == 0 else (d_model // n_heads)

        self.d_keys = d_keys if d_keys is not None else z
        self.d_values = d_values if d_values is not None else z
        self.n_heads = n_heads
        self.d_model = d_model  # d_model = C
        # 定义多头注意力层 AttentionLayer，封装了上面的 Attention 类并处理 Q、K、V 的线性变换等。
        # 初始化参数列表，包括 w_size（窗口大小，同window_size）、d_model（模型维度，即输入特征维数C）、n_heads（注意力头数）、d_keys和d_values（每头键/值维度，可选，不提供则自动计算）、mask_flag、scale、dropout 等等。
        # 调整 n_heads：如果 d_model % n_heads != 0（不能整除），则将 n_heads 强制设为1。这样保证每个头可以均分输入维度。
        # 临时变量 z：如果 (d_model // n_heads) == 0（表示 n_heads 大于 d_model，大概率已经被前一步修正为1，所以一般不会发生），则令 z = d_model % n_heads（这种情况下一般z会等于d_model本身）；
        # 否则正常设置 z = d_model // n_heads，即平均每头的维度大小。
        # self.d_keys 和 self.d_values：如果初始化参数提供了对应值则用之，否则用上面计算的 z 作为每个注意力头键和值的维度。
        # 同时保存 self.n_heads 和 self.d_model（即输入总特征维度C）。

        self.pos_embedding = PositionalEmbedding(d_model=d_model)

        # Linear projections to Q, K, V
        self.W_Q = nn.Linear(self.d_model, self.n_heads * self.d_keys)
        self.W_K = nn.Linear(self.d_model, self.n_heads * self.d_keys)
        self.W_V = nn.Linear(self.d_model, self.n_heads * self.d_values)
        # self.pos_embedding = PositionalEmbedding(d_model=d_model)，初始化一个位置嵌入对象，供后续使用以给输入添加位置信息。
        # 定义用于投影输入的线性层：self.W_Q self.W_K self.W_V。这些 Linear 层将把输入维度 d_model 映射到 n_heads * d_keys 或 n_heads * d_values 维度的输出向量。

        # self.out_proj = nn.Linear(self.n_heads * self.d_values, self.d_model)
        self.out_proj = lambda x: x

        self.attn = Attention(window_size=w_size, mask_flag=mask_flag, scale=scale, dropout=dropout)
        # 原本意图定义 self.out_proj = nn.Linear( )，即将多头输出再次映射回原始维度的线性层，但这行被注释掉。
        # 取而代之，将 self.out_proj = lambda x: x，也就是用匿名函数实现一个恒等映射作为输出投影。
        # 换言之，当前实现中并没有额外的全连接层混合多头输出，而是直接使用多头拼接结果作为最终输出。
        # 初始化 self.attn = Attention( )，创建一个 Attention 类实例用于实际计算注意力权重和输出。其中传入的参数来自 AttentionLayer自身的初始化参数。

    def forward(self, input_data):
        '''
        input : N x L x C(=d_model)
        '''

        N, L, _ = input_data.shape
        # 定义 forward(self, input_data) 方法，期待输入 input_data 形状为 (N, L, C)（N批大小，L序列长度，C即d_model特征维度），
        # 通过 N, L, _ = input_data.shape 获取批次和序列长度，其中下划线 _ 表示我们不显式使用第三个返回值（即C）。

        # input_data = input_data  + self.pos_embedding(input_data)

        # Q = self.W_Q(input_data).contiguous().view(N, L, self.n_heads, -1)
        # K = self.W_K(input_data).contiguous().view(N, L, self.n_heads, -1)
        # V = self.W_V(input_data).contiguous().view(N, L, self.n_heads, -1)
        # 拟将输入加上位置编码self.pos_embedding(input_data)。这会调用上面的 PositionalEmbedding.forward，生成长度为L的位置编码 (1,L,C)，利用广播加到 input_data 上，使每个序列元素叠加其位置信息。
        # 由于被注释，这一版模型未显式注入位置编码，可能是因为在频域+多尺度patch的结构下绝对位置影响较小，或者作者发现不加效果更好。
        # 原打算通过 W_Q, W_K, W_V 线性层计算多头的Q,K,V：分别 Q = self.W_Q(input_data).contiguous().view(N, L, self.n_heads, -1)，后面的 K,V 类似。
        # 这意味着将 input_data 先通过线性层映射到 (N, L, n_heads*d_keys)形状，然后 view 成四维 (N, L, n_heads, d_keys)。这正是标准**多头注意力**中对Q,K,V的线性映射和拆分过程。然而这几行也被注释掉。

        Q = input_data.contiguous().view(N, L, self.n_heads, -1)
        K = input_data.contiguous().view(N, L, self.n_heads, -1)
        V = input_data.contiguous().view(N, L, self.n_heads, -1)
        # Q, K, V 也就是将输入沿最后一维均匀拆分成 n_heads 份，视作各头的 Q/K/V。
        # 这里 -1 让PyTorch自动计算每头的维度大小（根据前面确保的可整除关系，-1会推断为 d_model/n_heads）。
        # 因为Q,K,V都赋值为同一个reshape的结果，实际上它们内容相同（都是 input_data 拆分出的子张量）。

        updated_V, attn = self.attn(Q, K, V)  # N x L x Head x d_values
        out = self.out_proj(updated_V.view(N, L, -1))
        # out = self.out_proj(updated_V.view(N, L, -1) + input_data)

        return out, attn
        # 调用前面初始化的 self.attn(Q, K, V) 计算注意力，得到 updated_V, attn。根据 Attention.forward 定义，updated_V 形状是 (N, L, Head, d_values)，attn 是平均后的注意力矩阵 (N, L, L)（这里 L(s)=L 因为 Q,K来自同一序列）。
        # updated_V 相当于每个head输出的值。第157行将 updated_V reshape 成 (N, L, -1)，即把多头的输出拼接回去（Head 和 d_values 两个维度合并）。然后通过 self.out_proj(...) 处理并赋给 out。
        # 由于我们将 out_proj 定义为了恒等 lambda，这里实际上只是完成了 tensor 的 reshape，未做额外线性变换。因此 out 形状为 (N, L, Head*d_values)。通常 Head*d_values应等于 d_model（如果我们让 d_values = d_keys = d_model/n_heads，且 n_heads乘起来就是d_model）。
        # 在当前实现中，由于我们拆分时 d_keys = d_values = d_model/n_heads，拼接后 out 就恢复为 (N, L, d_model)，与输入维度相同。
        # 第158行是被注释掉的残差连接尝试：原本想让 out = self.out_proj(updated_V.view(...)) + input_data，即将注意力输出与原输入相加（Residual），然后再投影。但被注释意味着没有在这里加残差。最后第160行返回 (out, attn)。

class TemporalAttentionLayer(nn.Module):
    """Single Graph Temporal Attention Layer
    :param n_features: number of input features/nodes
    :param window_size: length of the input sequence
    :param dropout: percentage of nodes to dropout
    :param alpha: negative slope used in the leaky rely activation function
    :param embed_dim: embedding dimension (output dimension of linear transformation)
    :param use_gatv2: whether to use the modified attention mechanism of GATv2 instead of standard GAT
    :param use_bias: whether to include a bias term in the attention layer
    定义 TemporalAttentionLayer 继承 nn.Module；
    在 MtsCID 中，该层实现论文提出的 “ts-attention”，把滑动窗口内的每个时间戳视作图节点，用于建模粗粒度的时间依赖。
    """

    def __init__(self, n_features, window_size, dropout=0.2, alpha=0.01, embed_dim=None, use_gatv2=True, use_bias=False):
        super(TemporalAttentionLayer, self).__init__()
        self.n_features = n_features
        self.window_size = window_size
        self.dropout = dropout
        self.use_gatv2 = use_gatv2
        self.embed_dim = embed_dim if embed_dim is not None else n_features
        self.num_nodes = window_size
        self.use_bias = use_bias
        # n_features：每个时间步的输入特征维度，即变量数 $k$。
        # window_size：滑动窗口的长度 $L$，模型将其中的每个时间步视作一个节点，因此 window_size 也是节点个数 (num_nodes)。
        # dropout：控制注意力权重的随机丢弃率，有助于防止过拟合。
        # alpha：原注释中提到它用于 LeakyReLU 的负斜率；代码中最终使用 GELU 激活，alpha 参数未直接用到，但保留以兼容其它版本。
        # embed_dim：输出的嵌入维度，若未指定默认为 n_features。注意在 GATv2 模式下该值会被乘以 2（见下文）。
        # use_gatv2：布尔值，决定使用 GATv2 版本还是原始 GAT 版本。两者的差异在于线性映射的顺序不同。
        # use_bias：是否为每对节点添加可学习偏置项

        # Because linear transformation is performed after concatenation in GATv2
        if self.use_gatv2:
            self.embed_dim *= 2
            lin_input_dim = 2 * n_features
            a_input_dim = self.embed_dim
        else:
            lin_input_dim = n_features
            a_input_dim = 2 * self.embed_dim
        # GATv2 模式：先拼接两个节点的原始特征，经过线性变换再计算注意力。由于后续拼接的两个节点特征长度为 $2\times n_features$，lin_input_dim 被设为 2 * n_features。
        # 同时作者将 embed_dim 乘以 2，是因为他们后续可能希望增加表示的容量。用于计算注意力权重的参数 a 的输入维度 a_input_dim 等于 embed_dim。
        # 原始 GAT：先对每个节点的特征做线性变换得到嵌入 $h_i$ (embed_dim 维)，然后拼接得到 $[h_i || h_j]$ 并与向量 a 点乘得到 $e_{ij}$。
        # 故 lin_input_dim = n_features（输入维度），a_input_dim = 2 * embed_dim。 

        self.lin = nn.Linear(lin_input_dim, self.embed_dim)
        self.a = nn.Parameter(torch.empty((a_input_dim, 1)))
        # nn.init.xavier_uniform_(self.a.data, gain=1.414)
        nn.init.xavier_normal_(self.a.data, gain=1.414)
        # self.lin：一个全连接层，用于将拼接后的特征映射到嵌入空间。对 GATv2 来说，输入是两个节点特征拼接的向量，输出为 embed_dim；对原始 GAT 来说，输入是单个节点特征，输出为 embed_dim。
        # self.a：可学习的参数向量，形状为 (a_input_dim, 1)。它用于计算注意力分数 $e_{ij}$，即 $e_{ij} = \mathbf{a}^T \cdot \text{feature}_{ij}$。
        # 权重初始化采用 Xavier 正态分布，以保持训练稳定。

        # if self.use_bias:
        #     self.bias = nn.Parameter(torch.empty(window_size, window_size))

        # self.leakyrelu = nn.LeakyReLU(alpha)
        self.leakyrelu = nn.GELU()
        self.sigmoid = nn.Sigmoid()
        self.norm = nn.LayerNorm(window_size)
        # 激活函数：代码使用 GELU（Gaussian Error Linear Unit），其输出更平滑，对梯度友好。
        # self.sigmoid：在原先版本中用于对聚合后的输出做通道间门控（类似门控循环单元），但本版代码并未使用它，后文也注释掉相应操作。
        # self.norm：LayerNorm(window_size) 对最后一维长度为 window_size 的向量做归一化。对注意力分数进行层归一化可以缓解梯度爆炸，使不同时间步的分数范围更一致，类似论文中对注意力图施加规范化。

    def forward(self, x):
        # x shape (b, n, k): b - batch size, n - window size, k - number of features
        # For temporal attention a node is represented as all feature values at a specific timestamp

        # 'Dynamic' GAT attention
        # Proposed by Brody et. al., 2021 (https://arxiv.org/pdf/2105.14491.pdf)
        # Linear transformation applied after concatenation and attention layer applied after leakyrelu
        # 输入解释：x 的形状为 (b, n, k)。其中 b 为批大小（batch size），n = window_size 表示时间窗口内的节点个数，每个节点的特征是长度为 k = n_features 的向量，对应该时间点的所有变量。
        # 注释强调：在这一层中，节点就是时间步，特征就是该时刻所有变量的值。

        if self.use_gatv2:
            a_input = self._make_attention_input(x)                # (b, n, n, 2*n_features)
            a_input = self.leakyrelu(self.lin(a_input))          # (b, n, n, embed_dim)
            e = torch.matmul(a_input, self.a).squeeze(3)  # (b, n, n, 1)
            e = self.norm(e)
            # self._make_attention_input(x)：调用辅助函数生成所有节点对的拼接特征。输出形状 (b, n, n, 2*n_features)，第 3、第 4 维分别对应被关注节点 i 和邻居节点 j，最后一维是 $[\mathbf{x}_i || \mathbf{x}_j]$。
            # self.leakyrelu(self.lin(a_input))：先用 self.lin 将拼接后的特征从维度 2*n_features 变换到 embed_dim 维，并应用 GELU 激活。这一步体现了 GATv2 的思想：先拼接再线性变换，使映射结果依赖于两端节点的组合，而不是分别独立地映射。
            # torch.matmul(a_input, self.a).squeeze(3)：将映射后的特征与注意力向量 self.a 做点乘，得到注意力分数张量 e。张量形状初为 (b, n, n, 1)，squeeze(3) 去除最后一维，结果为 (b, n, n)。这里的 $e_{ij}$ 衡量时间步 $i$ 对时间步 $j$ 的注意程度。
            # self.norm(e)：在最后一维 (长度为 n) 上做层归一化，使得每个节点的分数在统计上更平衡，减少训练不稳定性。

        # Original GAT attention
        else:
            Wx = self.lin(x)                                                  # (b, n, n, embed_dim)
            a_input = self._make_attention_input(Wx)                          # (b, n, n, 2*embed_dim)
            e = self.leakyrelu(torch.matmul(a_input, self.a)).squeeze(3)      # (b, n, n, 1)
            # self.lin(x)：首先对每个节点独立地做线性变换，x 的每一行 (b, n, k) 被投影到维度 embed_dim，得到 (b, n, embed_dim)。注释中原写 (b, n, n, embed_dim)，实际应为 (b, n, embed_dim)。
            # self._make_attention_input(Wx)：对变换后的节点表示构造所有节点对的拼接特征 (b, n, n, 2*embed_dim)，此时每一对拼接的是 $[W\mathbf{x}_i || W\mathbf{x}_j]$。
            # self.leakyrelu(torch.matmul(a_input, self.a)).squeeze(3)：与 GATv2 类似，对拼接后的表示乘以注意力向量并经过 GELU 得到分数矩阵 (b, n, n)。

        # if self.use_bias:
        #     e += self.bias  # (b, n, n, 1)

        # Attention weights
        attention = torch.softmax(e, dim=2)
        attention = torch.dropout(attention, self.dropout, train=self.training)
        # torch.softmax(e, dim=2)：对 e 在维度 2（邻居维度）上使用 Softmax，令每个 $e_{ij}$ 转化为权重 $α_{ij}$，并满足 $\sum_j α_{ij} = 1$。这一步对应 ts‑Attention 网络中的公式 (6)，即对注意力分数归一化为概率权重。
        # torch.dropout(attention, self.dropout, train=self.training)：在训练时对注意力权重施加 Dropout，以概率 self.dropout 随机丢弃部分连接，使模型更加鲁棒；测试时该操作不起作用。由于 torch.dropout 会创建与输入形状相同的随机 mask，丢掉的权重不会参与加权和。

        # h = self.sigmoid(torch.matmul(attention, x))    # (b, n, k)
        h = torch.matmul(attention, x)    # (b, n, k)
        # 对权重矩阵 attention 与原始输入 x 做矩阵乘法，形状 (b, n, n) × (b, n, k) 结果为 (b, n, k)。等式可以理解为 $\mathbf{h}i = \sum_j α{ij} \mathbf{x}_j$，即对每个节点 $i$，按照注意力权重对其他时间步的特征求加权和。这里并没有将自己的特征单独保留或者加残差连接，输出完全来自注意力聚合。
        return h, attention
        # 函数返回 (h, attention)，其中 h 是聚合后的特征，形状 (b, n, k)；attention 是归一化后的权重矩阵 (b, n, n)，可用于可视化依赖关系或进一步加权。注意力矩阵的每一行表示当前时间步对其他时间步的关注程度。

    def _make_attention_input(self, v):
        # 该函数的作用是构造所有节点对 $(i,j)$ 的拼接特征，用于注意力分数的计算
        # v 的输入形状为 (b, n, d)，其中 b 是批大小，n = K 是窗口长度，d 为特征维度（可能是原始特征或线性映射后的 embed_dim）
        """Preparing the temporal attention mechanism.
        Creating matrix with all possible combinations of concatenations of node values:
            (v1, v2..)_t1 || (v1, v2..)_t1
            (v1, v2..)_t1 || (v1, v2..)_t2

            ...
            ...

            (v1, v2..)_tn || (v1, v2..)_t1
            (v1, v2..)_tn || (v1, v2..)_t2

        """

        K = self.num_nodes
        blocks_repeating = v.repeat_interleave(K, dim=1)  # Left-side of the matrix
        blocks_alternating = v.repeat(1, K, 1)  # Right-side of the matrix
        combined = torch.cat((blocks_repeating, blocks_alternating), dim=2)

        if self.use_gatv2:
            return combined.view(v.size(0), K, K, 2 * self.n_features)
        else:
            return combined.view(v.size(0), K, K, 2 * self.embed_dim)

    
        # v.repeat_interleave(K, dim=1)：对第二维（节点维）重复每个时间步 $K$ 次，得到形状 (b, n*K, d)。例如原 v 中第一个时间步的向量将在前 K 行全部出现，对应矩阵左侧。

        # v.repeat(1, K, 1)：在第二维上整体重复整个序列 $K$ 次，得到形状 (b, n*K, d)。这样第 0~K‑1 行为原序列，K~2K‑1 行又是原序列，以此类推，形成矩阵右侧。
        # torch.cat((blocks_repeating, blocks_alternating), dim=2)：沿特征维度拼接左侧和右侧，得到 (b, n*K, 2d)，其中每行是 $[\mathbf{v}_i || \mathbf{v}_j]$。
        # 最后 reshape 成 (b, K, K, 2d)：重塑后三维对应于 (节点 i, 节点 j, 拼接特征维度)。选择使用哪个维度取决于 self.use_gatv2：如果是 GATv2，拼接的是原始特征，维度为 2 * n_features；否则是线性映射后的嵌入，维度为 2 * embed_dim。
