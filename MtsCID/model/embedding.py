import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from model.attn_layer import (PositionalEmbedding,
                              AttentionLayer,
                              complex_dropout,
                              complex_operator)
from model.Conv_Blocks import Inception_Block
from model.multi_attention_blocks import Inception_Attention_Block

from model.RevIN import RevIN

class EncoderLayer(nn.Module):
    def __init__(self, attn, d_model, d_ff=None, dropout=0.1, activation='relu'):
        super(EncoderLayer, self).__init__()
        # d_ff = d_ff if d_ff is not None else 4 * d_model
        self.attn_layer = attn
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(p=dropout)
        self.activation = F.relu if activation == 'relu' else F.gelu
        # 首先是 EncoderLayer 类，它代表 Transformer 编码器层中的一个子模块。
        # 它持有一个注意力子层 attn_layer（假定已经实现了自注意力机制）和一个层归一化 self.norm = LayerNorm(d_model)，以及一个 Dropout 和激活函数（ReLU 或 GELU）。

    def forward(self, x):
        """
        x : N x L x C(=d_model)
        """
        out, attn = self.attn_layer(x)
        y = complex_dropout(self.dropout, out)
        return y
        # 在 forward 方法中，输入 x（形状 (N, L, d_model)）先经过注意力层 self.attn_layer(x)，得到输出 out（以及注意力权重 attn）
        # 随后通过 complex_dropout(self.dropout, out) 应用丢弃层（complex_dropout 是为了支持实部和虚部同时丢弃的自定义函数）。最终返回 y。
        # 注意这里代码中并未显式应用残差连接和归一化，可能假设 self.attn_layer 已经包含了相应操作（或者这是一处遗漏），但我们这里按照代码直接说明功能：该层对输入进行注意力变换和 dropout，并返回变换后的特征。

# Transformer Encoder
class Encoder(nn.Module):
    def __init__(self, attn_layers, norm_layer=None):
        super(Encoder, self).__init__()
        self.attn_layers = nn.ModuleList(attn_layers)
        self.norm = norm_layer
        # 其次是 Encoder 类，它串联了多个 EncoderLayer（或者其他注意力层）。初始化时传入一个 ModuleList 的注意力子层 attn_layers 以及可选的归一化层 norm_layer。

    def forward(self, x):
        """
        x : N x L x C(=d_model)
        """

        for attn_layer in self.attn_layers:
            x, _ = attn_layer(x)

        if self.norm is not None:
            x = self.norm(x)

        return x
        # 在 forward 中，它依次将输入 x 通过每个注意力层，最后如果提供了 self.norm，则做一次归一化后返回结果。
        # 此模块可用于堆叠若干个自注意力层（尽管在本代码中未直接看到引用这个 Encoder 类的地方，也许在 AttentionLayer 内部或其他处使用）。

class TokenEmbedding(nn.Module):
    def __init__(self, in_dim, d_model, n_window=100, n_layers=1, branch_layers=['fc_linear', 'intra_fc_transformer'],
                 group_embedding='False', match_dimension='first', kernel_size=[5], multiscale_patch_size=[10, 20],
                 init_type='normal', gain=0.02, dropout=0.1):
        super(TokenEmbedding, self).__init__()

        self.window_size = n_window
        self.d_model = d_model
        self.n_layers = n_layers
        self.branch_layers = branch_layers
        self.group_embedding = group_embedding
        self.match_dimension = match_dimension
        self.kernel_size = kernel_size
        self.multiscale_patch_size = multiscale_patch_size

        # For the input is data in the frequency domain, n_network is two for real and imagery
        component_network = ['real_part', 'imaginary_part']
        num_in_fc_networks = len(component_network)
        # in_dim 是输入特征维度（变量数）；d_model 是输出嵌入维度，用于扩张特征空间。
        # n_window 指滑动窗口的长度，每个时间步视为一个节点，稍后许多层会用到它。
        # branch_layers 是一个字符串列表，决定了网络将串联哪些模块。根据论文描述，上支路首先做频域映射（fc‑Linear）、然后在频域建模依赖（fc‑Transformer）、最后在多尺度补丁上做 ts‑Attention；
        # 下支路先用卷积在时域捕捉局部依赖，再用 fc‑Transformer 在频域捕捉变量关系。
        # component_network 包含 'real_part' 和 'imaginary_part'；频域张量是复数形式，因此后续 fc‑Linear 和 fc‑Transformer 均需要分别处理实部和虚部。

        self.encoder_layers = nn.ModuleList([])
        self.norm_layers = nn.ModuleList([])

        for i, e_layer in enumerate(branch_layers):
            if self.match_dimension == 'none':
                updated_in_dim = in_dim
                extended_dim = in_dim
                # 所有层的输入输出维度都保持为最初的 in_dim，即不进行维度扩展。
            elif (i == 0 and self.match_dimension == 'first') or (len(branch_layers) < 2):
                updated_in_dim = in_dim
                extended_dim = d_model
                # 如果共有多层，那么**第一层**的输出维度设置为 `d_model`，其余层输入输出维度都等于 `d_model`；也就是说在最开始就扩展到目标维度。若仅有一层，同样扩展到 `d_model`。
                # 这种设置在论文的上支路较常见，即先用 fc‑Linear 把每个变量映射到 `d_model` 维，再经过 Transformer 等模块进行复杂建模。
            elif (i == 0) and (not self.match_dimension == 'first'):
                updated_in_dim = in_dim
                extended_dim = in_dim
            elif (i + 1 < len(branch_layers)) and (self.match_dimension == 'middle'):
                updated_in_dim = extended_dim
                extended_dim = d_model
                # 当层数多于 2 时，前三个分支的输入输出维度保持不变，到中间某层（`i+1 < len(branch_layers)`）时将输出扩展到 `d_model`。之后所有层都使用 `d_model` 作为输入和输出维度。
                # 这样的设置适用于先做一些卷积或线性操作保持原维度，再在中间阶段扩展表示能力。
            elif i + 1 == len(branch_layers):
                updated_in_dim = extended_dim
                extended_dim = d_model
            else:
                updated_in_dim = extended_dim
                extended_dim = extended_dim
                # 默认或 `'last'`：只有在**最后一个子模块**才将输出扩展到 `d_model`；其余层的输入输出维度保持不变。
                # 这适用于想在最终输出之前才扩张维度，比如先对原始维度进行频域操作，再将结果映射到更高维用于后续模块。

            if 'conv1d' in e_layer or 'deconv1d' in e_layer:
                if self.group_embedding == 'False':
                    groups = 1
                else:
                    if extended_dim >= updated_in_dim and extended_dim % updated_in_dim == 0:
                        groups = updated_in_dim
                    elif extended_dim < updated_in_dim and updated_in_dim % extended_dim == 0:
                        groups = extended_dim
                    else:
                        print(f"The conv1d/deconv1d layer {i} of encoder is non-grouped convolution!")
                        groups = 1

            # 紧接着，根据 e_layer 的不同，将具体的层模块添加到 self.encoder_layers 和对应的归一化层 self.norm_layers 中：
            if e_layer == 'dropout':
                self.encoder_layers.append(nn.Dropout(p=dropout))
                self.norm_layers.append(nn.Identity())
                # - 'dropout'：如果层类型是 dropout，则在 encoder_layers 添加 nn.Dropout，norm_layers 添加身份映射 nn.Identity()。

            elif e_layer == 'fc_linear':
                self.encoder_layers.append(nn.ModuleList([nn.Linear(updated_in_dim, extended_dim, bias=False)
                                                         for _ in range(num_in_fc_networks)])
                                           )
                self.norm_layers.append(nn.ModuleList([nn.LayerNorm(extended_dim) for _ in range(num_in_fc_networks)]))
                # - 'fc_linear'：实现论文中的 fc-Linear（频率域全连接）。这里为实部和虚部分别添加一组 nn.Linear(updated_in_dim, extended_dim, bias=False)，
                # 共两条线性变换（使用 ModuleList 保存），并在 norm_layers 中加入两个 LayerNorm(extended_dim)。

            elif e_layer == 'linear':
                self.encoder_layers.append(nn.ModuleList([nn.Linear(updated_in_dim, extended_dim, bias=False)
                                                          for _ in range(num_in_fc_networks)]))
                self.norm_layers.append(nn.ModuleList([nn.LayerNorm(extended_dim) for _ in range(num_in_fc_networks)]))
                # -'linear'：类似 fc_linear，添加两个全连接层用于实虚部处理，并对应两个 LayerNorm。

            elif e_layer == 'multiscale_conv1d':
                for _ in range(n_layers):
                    self.encoder_layers.append(Inception_Block(in_channels=updated_in_dim,
                                                               out_channels=extended_dim,
                                                               kernel_list=kernel_size,
                                                               groups=groups
                                                               )
                                                   )
                    self.norm_layers.append(nn.ModuleList([nn.LayerNorm(self.window_size)
                                                           for _ in range(num_in_fc_networks)]))
                    # - 'multiscale_conv1d'：添加多尺度卷积块 Inception_Block。根据 n_layers（例如多次堆叠），每次插入一个 Inception_Block()，
                    # 这里的 kernel_list 提供了多尺度卷积核（对应论文中对时间域进行多尺度卷积捕捉不同时间窗口的模式）。
                    # 对应地，在 norm_layers 中添加两个 LayerNorm(self.window_size)（这里对时间步维度进行归一化）。

            elif e_layer == 'inter_fc_transformer':
                w_model = self.window_size // 2 + 1
                attention_layer = AttentionLayer(w_size=extended_dim, d_model=w_model, n_heads=1, dropout=dropout)
                self.encoder_layers.append(nn.ModuleList([EncoderLayer(attn=attention_layer,
                                                                       d_model=w_model,
                                                                       d_ff=128,
                                                                       dropout=dropout,
                                                                       activation='gelu'
                                                                       )
                                                          for _ in range(num_in_fc_networks)])
                                           )
                self.norm_layers.append(nn.ModuleList([nn.LayerNorm(self.window_size)
                                                       for _ in range(num_in_fc_networks)]))
                # - 'inter_fc_transformer'：添加跨变量的全连接 Transformer（论文 i-Encoder 使用的频域Transformer）。
                # 计算注意力时的序列长度设为 $w_model = \frac{L}{2}+1$（即频域长度），注意力层 AttentionLayer(w_size=extended_dim, d_model=w_model, n_heads=1, dropout=dropout) 构建一个单头自注意力，然后用它构造两个并行的 EncoderLayer（实部和虚部各一个）。
                # 在 norm_layers 中加入两个 LayerNorm(self.window_size)。

            elif e_layer == 'intra_fc_transformer':
                w_model = self.window_size // 2 + 1
                attention_layer = AttentionLayer(w_size=w_model, d_model=extended_dim, n_heads=1, dropout=dropout)
                self.encoder_layers.append(nn.ModuleList([EncoderLayer(attn=attention_layer,
                                                                       d_model=extended_dim,
                                                                       d_ff=128,
                                                                       dropout=dropout,
                                                                       activation='gelu'
                                                                       )
                                                          for _ in range(num_in_fc_networks)])
                                           )
                self.norm_layers.append(nn.ModuleList([nn.LayerNorm(self.window_size)
                                                       for _ in range(num_in_fc_networks)]))
                # - 'intra_fc_transformer'：添加在单个变量内部的全连接 Transformer（论文 t-AutoEncoder 的频域Transformer）。
                # 这里注意力层 AttentionLayer(w_size=w_model, d_model=extended_dim, n_heads=1, dropout=dropout) 将 w_model = L//2+1（频域长度）作为查询长度，输出维度为 extended_dim。
                # 同样使用两个 EncoderLayer 存入 encoder_layers，norm_layers 添加两个 LayerNorm(self.window_size)。

            elif e_layer == 'multiscale_ts_attention':
                self.encoder_layers.append(Inception_Attention_Block(w_size=self.window_size,
                                                                     in_dim=extended_dim,
                                                                     d_model=extended_dim,
                                                                     patch_list=multiscale_patch_size))
                # self.norm_layers.append(nn.LayerNorm(extended_dim))
                self.norm_layers.append(nn.Identity())
                # - 'multiscale_ts_attention'：添加多尺度时序注意力块 Inception_Attention_Block。
                # 这个模块会将输入数据划分为不同大小的时间补丁，并在每个补丁内部计算自注意力，捕获不同粒度的时序依赖（对应 t-AutoEncoder 中基于多尺度补丁的 ts-Attention ）。
                # Inception_Attention_Block(w_size=self.window_size, in_dim=extended_dim, d_model=extended_dim, patch_list=multiscale_patch_size) 指定输入和输出维度都为 extended_dim，补丁列表为多个给定大小。
                # norm_layers 对于该层使用 Identity，因为注意力块内部通常已包含归一化。
            else:
                raise ValueError(f'The specified model {e_layer} is not supported!')
                # - **其他未知层类型**：如果提供了未知的层类型，则抛出错误提醒不支持。

        self.dropout = nn.Dropout(p=dropout)
        self.criterion = nn.MSELoss(reduction='none')
        self.activation = nn.GELU()
        # self.activation = nn.LeakyReLU(0.2, inplace=True)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                if init_type == 'normal':
                    torch.nn.init.normal_(m.weight.data, 0.0, gain)
                elif init_type == 'xavier':
                    torch.nn.init.xavier_normal_(m.weight.data, gain=gain)
                elif init_type == 'kaiming':
                    torch.nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
                elif init_type == 'orthogonal':
                    torch.nn.init.orthogonal_(m.weight.data, gain=gain)
                else:
                    torch.nn.init.uniform_(m.weight.data, a=-0.5, b=0.5)

                if hasattr(m, 'bias') and m.bias is not None:
                    torch.nn.init.constant_(m.bias.data, 0.0)

            elif isinstance(m, nn.Conv1d) or isinstance(m, nn.ConvTranspose1d):
                if init_type == 'normal':
                    torch.nn.init.normal_(m.weight.data, 0.0, gain)
                elif init_type == 'xavier':
                    torch.nn.init.xavier_normal_(m.weight.data, gain=gain)
                elif init_type == 'kaiming':
                    torch.nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
                elif init_type == 'orthogonal':
                    torch.nn.init.orthogonal_(m.weight.data, gain=gain)
                else:
                    torch.nn.init.uniform_(m.weight.data, a=-0.5, b=0.5)
                if hasattr(m, 'bias') and m.bias is not None:
                    torch.nn.init.constant_(m.bias.data, 0.0)
                # 在构造完所有层后，还对整个模块内的线性和卷积层做了一次统一初始化，类似于 TransformerVar 中那样，根据 init_type（正态、Xavier、Kaiming 等）设置权重分布。

    def forward(self, x):

        B, L, C = x.size()
        latent_list = []
        residual = None
        amplitudeRevIN = RevIN(int(L//2 + 1))
        # 完成初始化后就是 TokenEmbedding 的 forward 方法。它的输入 x 形状为 (B, L, C)。首先记录批大小 B、序列长度 L、通道数 C。
        # 然后创建一个 latent_list 列表用于存储各层输出，residual 用于后面可能的残差连接。
        # 接着实例化 RevIN(int(L//2 + 1))，这一行代码似乎是希望对振幅谱做逆归一化，但在以下代码中并未实际使用（可能遗留或备用）。

        for i, (embedding_layer, norm_layer) in enumerate(zip(self.encoder_layers, self.norm_layers)):
            if self.branch_layers[i] not in ['linear', 'fc_linear', 'multiscale_ts_attention']:
                x = x.permute(0, 2, 1)
                # 然后通过 zip(self.encoder_layers, self.norm_layers) 遍历所有层。由于部分层需要将输入通道和时间步维度互换（例如卷积和 Transformer 层通常对 [batch, channels, length] 形状操作），
                # 代码首先检查层类型：如果当前层不是 'linear'、'fc_linear' 或 'multiscale_ts_attention'，则对输入张量 x 做转置 x = x.permute(0, 2, 1)，将维度 (B, L, C) 变为 (B, C, L)。这样可以将通道数放在第二维，以适配 Conv1d 或按通道进行的操作。
            if self.branch_layers[i] == 'multiscale_conv1d':
                x = complex_operator(embedding_layer, x)
            elif self.branch_layers[i] == 'multiscale_ts_attention':
                x = complex_operator(embedding_layer, x)
                # 根据层的类型，进行不同的计算：
                # 1. 'multiscale_conv1d' 或 'multiscale_ts_attention'：这两种情形都使用 complex_operator(embedding_layer, x)。
                # complex_operator 是一个自定义函数，用于处理含有实部和虚部的复数张量，或者同时处理实部网络和虚部网络。
                # 对于 'multiscale_conv1d'，embedding_layer 是一个或多个 Inception_Block，该操作将在通道维上进行；
                # 对于 'multiscale_ts_attention'，embedding_layer 是一个多尺度注意力块 Inception_Attention_Block。这两种情况下，代码就直接把当前张量送入对应模块。

            elif self.branch_layers[i] in ['fc_linear']:
                x = torch.fft.rfft(x, dim=-2)
                x = complex_operator(embedding_layer, x)
                x = torch.fft.irfft(x, dim=-2)
                # 'fc_linear'：对应 t-AutoEncoder 的第一步频域投影。这里先使用 torch.fft.rfft(x, dim=-2) 对张量在倒数第二维（注意 x 在此应是 (B, L, C) 形式，因为 'fc_linear' 在前面的 if 中不会触发转置）进行实值快速傅里叶变换，将时域信号转到频域。
                # 输出仍是实数，因为 .rfft 会返回频谱的实部和虚部在一个复数张量中。
                # 然后对这个复频谱应用线性网络：complex_operator(embedding_layer, x) 会分别对实部和虚部应用之前在初始化中加入的两组 Linear 网络（分别对应论文的 $W^{(r)}$ 和 $W^{(i)}$），即实现式 (2) 中的 $H W^{(r)}$ 和 $H W^{(i)}$。
                # 之后调用 torch.fft.irfft(x, dim=-2) 将结果转换回时域。总体来说，这一步实现了论文中 fc-Linear 模块在频域的映射，捕获频域的特征。

            elif self.branch_layers[i] in ['inter_fc_transformer']:
                x = torch.fft.rfft(x, dim=-1)
                x = complex_operator(embedding_layer, x)
                x = torch.fft.irfft(x, dim=-1)
                # 'inter_fc_transformer'：对应 i-Encoder 中的跨变量频域自注意力。
                # 首先对 (B, L, C) 格式的输入在最后一个维度 dim=-1 做傅里叶变换：x = torch.fft.rfft(x, dim=-1)，此时 x 的形状应为 (B, C, f)，其中 f=L//2+1 是频域长度。
                # 然后通过 complex_operator(embedding_layer, x) 应用之前构造的 EncoderLayer （包含注意力）。这里没有显式地在 Python 代码中看到转置操作，但请注意：在循环开始时，对于 'inter_fc_transformer' 也会执行 permute(0,2,1)，因此 x 在这一时刻实际上已经是 (B, C, L)，所以对 dim=-1 做FFT实际上是对 L 做FFT。
                # 应用注意力后，再用 torch.fft.irfft(x, dim=-1) 变换回时域。这个过程对应论文第2.4节中 fc-Transformer 沿变量维度进行自注意力的实现（式(11)） 。

            elif self.branch_layers[i] in ['intra_fc_transformer']:
                x = torch.fft.rfft(x, dim=-1)
                x = x.permute(0, 2, 1)
                x = complex_operator(embedding_layer, x)
                x = x.permute(0, 2, 1)
                x = torch.fft.irfft(x, dim=-1)
                # 'intra_fc_transformer'：对应 t-AutoEncoder 中的变量内部频域自注意力。
                # 此时，在上一步骤结束后，x 仍是 (B, L, C) 格式，于是第一个条件又会执行 permute 操作将其变为 (B, C, L)。
                # 然后执行 x = torch.fft.rfft(x, dim=-1) 将每个变量的时序变换到频域，得到 (B, C, f)。
                # 接下来代码又一次把通道和频率维交换：x = x.permute(0, 2, 1) 得到 (B, f, C) 形状，表示频率作为“序列长度”，变量通道作为特征维度。
                # 然后应用 complex_operator(embedding_layer, x)，相当于在频域对每个变量内部进行自注意力交互（这里的网络维度设置使得输出特征与输入一致）。
                # 随后又将形状恢复：x = x.permute(0, 2, 1) 回到 (B, C, f)，再执行 torch.fft.irfft(x, dim=-1) 恢复到时域 (B, C, L)。
                # 这段操作实现了论文式 (3)-(4) 所描述的频域Transformer处理过程，得到重新归一化后的 Z 。

            else:
                x = complex_operator(embedding_layer, x)
                # 其他（默认）情况：如果层类型不是上述几种，则认为是普通的时间域操作，只需要将输入 x 直接送入相应模块：x = complex_operator(embedding_layer, x)。
                # 这通常对应 'linear'（普通线性层）或可能的其他操作。complex_operator 会分别应用实部和虚部的网络。

            x = complex_operator(norm_layer, x)
            # x = self.activation(x)
            # x = self.dropout(x)
            if self.branch_layers[i] not in ['linear', 'fc_linear', 'multiscale_ts_attention']:
                x = x.permute(0, 2, 1)
            # 处理完该层的主要操作后，紧接着应用对应的归一化层：x = complex_operator(norm_layer, x)。
            # 如果该层类型在第一次判断中触发了维度交换（即不属于 ['linear','fc_linear','multiscale_ts_attention']），那么在完成当前层计算后需要将张量维度交换回来：x = x.permute(0, 2, 1)。
            # 这一步在代码中通过在循环末尾对列表类型再次检查 if self.branch_layers[i] not in ['linear','fc_linear','multiscale_ts_attention']: 实现。
            # 换言之，对于除上述三种外的所有层，我们先把 (B,L,C) 变成 (B,C,L)，做操作后再恢复 (B,L,C) 形式。

            latent_list.append(x)

            # After each transformer layer, a residual connection is used
            if residual is not None:
                if x.shape == residual.shape and 'transformer' in self.branch_layers[i]:
                    x += residual

            if self.branch_layers[i] in ['linear', 'fc_linear']:
                residual = x
                # 每经过一层后，将当前 x（归一化后的输出）加入 latent_list。
                # 此外，如果上一层保留了一个残差 residual（对应长跳），并且当前层也是一种 Transformer 类型（这里判断是否含有 'transformer'），则将残差加到当前输出上，实现残差连接。
                # 代码中对残差的更新规则是：只有在层类型为 'linear' 或 'fc_linear' 时更新 residual = x。
                # 因此，残差实际上是上一层是线性层时保存的输出；当之后遇到 Transformer 层时再加回来。这实现了论文中提到的“残差连接”和归一化 。

        return x, latent_list
        # 完成所有层迭代后，TokenEmbedding.forward 返回最终的特征 x（形状 (B, L, C')，其中 C' 是最后一层的输出维度）和 latent_list 中记录的每层特征列表。
        # 这个 x 随后会被 InputEmbedding 接收。

class InputEmbedding(nn.Module):
    def __init__(self, in_dim, d_model, n_window, device, dropout=0.1, n_layers=1, use_pos_embedding='False',
                 group_embedding='False', kernel_size=5, init_type='kaiming', match_dimension='first',  branch_layers=['linear']):
        super(InputEmbedding, self).__init__()
        self.device = device
        self.token_embedding = TokenEmbedding(in_dim=in_dim, d_model=d_model, n_window=n_window,
                                              n_layers=n_layers, branch_layers=branch_layers,
                                              group_embedding=group_embedding, match_dimension=match_dimension,
                                              init_type=init_type, kernel_size=kernel_size,
                                              dropout=0.1)
        self.pos_embedding = PositionalEmbedding(d_model=d_model)
        self.use_pos_embedding = use_pos_embedding
        self.dropout = nn.Dropout(p=dropout)
        # 最后是 InputEmbedding 类，它将 TokenEmbedding 与可选的位置编码结合。
        # 初始化时接收输入维度 in_dim、输出嵌入维度 d_model、窗口大小 n_window、设备 device、丢弃率 dropout、层数 n_layers、是否使用位置编码 use_pos_embedding、分组嵌入 group_embedding、卷积核大小 kernel_size、初始化类型 init_type、匹配维度 match_dimension 以及具体的 branch_layers 列表。
        # 它内部创建一个 TokenEmbedding 子模块（负责实际的层堆叠）和一个 PositionalEmbedding（用于加入绝对位置编码）。

    def forward(self, x):
        x = x.to(self.device)

        x, latent_list = self.token_embedding(x)

        if self.use_pos_embedding == 'True':
            x = x + self.pos_embedding(x).to(self.device)

        return self.dropout(x), latent_list
        # InputEmbedding.forward(x) 先将输入 x 移到指定设备，然后调用 self.token_embedding(x)，得到 (x, latent_list)。其中 x 经过上述各层变换的输出（形状 (B, L, d_model)）。
        # 如果 use_pos_embedding 被设置为 'True'，则计算位置编码 self.pos_embedding(x) 并加到 x 上，以注入序列的位置信息（论文中未明确提及，但在 Transformer 通常会使用位置编码）。
        # 随后对 x 应用 dropout，最终返回一个元组 (x, latent_list)。这意味着 InputEmbedding 的输出除了转换后的特征张量外，还带有各层潜在特征列表，以便上层可以获取中间信息（TransformerVar.forward 里即使用到了这部分潜在信息，至少在上分支中记录了 t_latent_list）。

'''
综上所述，embedding.py 实现了输入序列的嵌入过程，包括多尺度卷积、频域线性变换、频域自注意力和补丁注意力等模块。
结合 MtsCID 论文的描述：上分支的 TokenEmbedding 通过 ['fc_linear', 'intra_fc_transformer', 'multiscale_ts_attention'] 的层序列实现了论文 2.3 中的频域编码器和时序自注意力 ；
下分支的 TokenEmbedding 通过 ['multiscale_conv1d', 'inter_fc_transformer'] 实现了论文 2.4 中的时域卷积和跨变量频域编码器 。
这里的 complex_operator 和分离实部/虚部的设计正是为了在频域上处理复数（变换后的实虚部分开处理），符合论文中同时处理频域实部和虚部的要求 。
残差连接、层归一化、位置编码等 Transformer 常用技巧也在代码中体现（如对 Transformer 层后的输出进行归一化和可能的跳连）。
而 RevIN 虽然在代码中出现，但实际逻辑没有使用上，可能预留用于对变换后的频谱幅度做归一化，类似文中处理幅度信息的想法。
最后，在回答中参考了 MtsCID 论文中对 t-AutoEncoder 和 i-Encoder 以及原型交互模块的描述，例如论文对 fc-Linear、fc-Transformer、补丁注意力、卷积和原型的说明，以帮助理解代码与论文设计的对应关系。
总体而言，这些代码逐行实现了论文中提出的双分支网络及其各个组成模块，将多变量时间序列分别在时域和频域上进行编码和交互，以捕获粗粒度的时序依赖和变量间关系。
'''