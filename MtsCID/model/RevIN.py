import torch
import torch.nn as nn

'''
RevIN.py 文件实现了 Reversible Instance Normalization（可逆实例归一化）算法。RevIN 是 Kim 等人在 ICLR 2022 提出的方法，
旨在针对时间序列数据中的非平稳性和分布漂移问题，通过可逆的归一化和反归一化操作，使模型对不同分布的鲁棒性提高，同时保留每个实例自身的统计特性。
简单来说，在模型输入前，对每个时间序列样本独立进行标准化（按其均值和方差），使其具有零均值单位方差；模型输出后，再用先前保存的均值和方差将结果还原到原始尺度。
这样处理可以消除不同序列之间的尺度差异和趋势影响，缓解训练集与测试集分布不一致的问题，同时不丢失各实例的独有特征。
MtsCID 采用 RevIN 作为数据归一化手段，可以确保模型学到的是归一化后的正常模式，在检测时再转换回原始值计算误差，从而更准确地区分异常点。
'''

class RevIN(nn.Module):
    def __init__(self, num_features: int, eps=1e-5, affine=True, device=-1):
        """
        :param num_features: the number of features or channels
        :param eps: a value added for numerical stability
        :param affine: if True, RevIN has learnable affine parameters
        """
        super(RevIN, self).__init__()
        # RevIN 继承自 nn.Module。初始化参数：num_features: 特征数量，即每个样本的通道数（变量数）。RevIN 会对每个特征独立归一化，因此需要知道有多少特征。
        # eps: 数值稳定常数，默认 1e-5，用于避免除零或极小标准差时数值不稳定。
        # affine: 布尔值，默认为 True，表示是否使用可学习的仿射参数（缩放 $\gamma$ 和偏移 $\beta$）。如果 True，则在标准化后乘以 $\gamma$ 加上 $\beta$，这些参数会在训练中学习调整。设置 False 则不使用仿射变换，仅执行纯统计标准化。
        # device: 指定在哪个设备上初始化 RevIN 参数。如果为 -1（默认），则自动选择当前可用设备。

        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        self.device = torch.device(f'cuda:{device}' if (torch.cuda.is_available() and device > 0) else 'cpu')
        if self.affine:
            self._init_params()
            # 首先调用父类构造函数，确保 nn.Module 正常初始化。保存 num_features, eps, affine 为对象属性。
            # 确定 self.device: 如果 device > 0 且有可用GPU，则指定相应GPU，否则使用 CPU。这样可以将RevIN内部参数放在正确的设备上。这里 device 参数允许用户选择某个GPU编号；默认 -1 则使用 CPU。
            # 如果 self.affine 为 True，则调用 _init_params() 方法初始化仿射参数（缩放和偏置）。对于 False 则跳过，意味着RevIN将无可学习参数，仅根据数据统计做无参数归一化。

    def forward(self, x, mode:str):
        if mode == 'norm':
            self._get_statistics(x)
            x = self._normalize(x)
        elif mode == 'denorm':
            x = self._denormalize(x)
        else: raise NotImplementedError
        return x
        # RevIN 的 forward 方法根据 mode 参数执行不同操作：
        # mode == 'norm': 正常化模式。表示在输入数据进入模型前，执行归一化：调用 self._get_statistics(x) 计算输入 x 的均值和标准差（保存为内部属性）,将 x = self._normalize(x)，对数据进行归一化处理（减均值除以标准差，并应用仿射变换如果启用）。
        # mode == 'denorm': 反归一化模式。表示模型输出后，将标准化的数据还原; 直接执行 x = self._denormalize(x)，利用之前保存的统计量将数据反变换回原尺度。
        # 如果传入其他无效的 mode 字符串，则抛出 NotImplementedError 以提示错误。
        # 函数最后返回处理后的 x。也就是说，使用 RevIN 时需明确调用一次 forward(x, 'norm') 来归一化输入，通过模型处理后，再调用 forward(output, 'denorm') 来还原输出。这种接口设计要求调用方控制两步，确保_get_statistics所保存的均值、方差用于对应的 denorm。

    def _init_params(self):
        # initialize RevIN params: (C,)
        self.affine_weight = torch.ones(self.num_features)
        self.affine_bias = torch.zeros(self.num_features)
        self.affine_weight = self.affine_weight.to(device=self.device)
        self.affine_bias = self.affine_bias.to(device=self.device)
        # 当需要仿射变换时，这个函数创建两个可训练参数：
        # affine_weight: 大小为 (num_features,) 的张量，初始化为全1（各特征的缩放初始为1，不改变尺度）。
        # affine_bias: 大小为 (num_features,) 的张量，初始化为全0（各特征的偏置初始为0，不偏移）。接着，将这两个张量移动到指定的 self.device（如GPU）上。
        # 注意：这里直接将张量赋给对象属性，但未包装为 nn.Parameter。这意味着这两个张量在默认情况下不会被视为可学习参数（不会自动出现在 model.parameters() 中）。
        # 然而，由于 RevIN 的思路中 $\gamma, \beta$ 是需要训练的，理应将其注册为参数。可能原作者假定将 RevIN 用于前处理和后处理，不参与反向梯度（即不优化 $\gamma, \beta$）。
        # 但按照RevIN原论文，$\gamma, \beta$ 确实是可学习的参数。这里或许是实现上的简化，虽然未用 nn.Parameter 明确注册，但训练时如果将 RevIN 包含在模型中，这两个属性也会跟随梯度更新（因为它们是 tensor，但不是Parameter，PyTorch不会自动跟踪，不过如果显式把 RevIN 加入优化器参数列表则可以）。
        # 总之，这是一个实现细节。就算法逻辑而言，可以认为 $\gamma$ 初始化为1、$\beta$ 为0，可以学习调整归一化后的分布 ￼。
        
    def _get_statistics(self, x):
        dim2reduce = tuple(range(1, x.ndim-1))
        self.mean = torch.mean(x, dim=dim2reduce, keepdim=True).detach()
        self.stdev = torch.sqrt(torch.var(x, dim=dim2reduce, keepdim=True, unbiased=False) + self.eps).detach()
        # 这个函数对输入张量 x 计算每个特征的均值和标准差，并保存：
        # dim2reduce = tuple(range(1, x.ndim-1)): 构造要归约的维度元组。对于输入 x 通常形状是 (B, T, C) 或 (B, …, C)，也就是批次、时间步、特征。
        # 在常见场景中，x.ndim 会是3（三维张量：批次、时间、通道）。那么 range(1, x.ndim-1) = range(1, 2) 只有一个值 1，即时间步维度。该 tuple用于指定对哪些维求均值/方差。
        # 直观理解：对每个样本每个特征独立计算统计量，也就是对 batch维和特征维之外的维度进行归约。在 (B,T,C) 中，就是对 T 维求平均和方差，保留 (B,1,C) 形状。
        # self.mean : 计算 x 在指定维度上的均值。keepdim=True 保留被归约的维度形状，以便后续在张量减法中直接使用。.detach() 将结果从计算图中分离，防止均值在反向传播中产生梯度（因为均值作为统计量，不需要梯度更新）。
        #   计算结果 self.mean 的形状是 (B, 1, C)，即每个样本每个特征的均值。
        # self.stdev: 计算 x 在指定维度上的方差，用 torch.var（unbiased=False 表示使用有偏估计，即除以 N 而非 N-1，更适合深度学习场景）。然后加上一个小值 eps 增强数值稳定性，再开方得到标准差。
        #   keepdim=True 使结果形状为 (B, 1, C)。同样用 .detach()分离梯度。self.stdev 即每个样本每个特征的标准差（的平滑估计）。
            
    def _normalize(self, x):
        x = x - self.mean
        x = x / self.stdev
        if self.affine:
            x = x * self.affine_weight
            x = x + self.affine_bias
        return x
        # 这个函数执行实际的标准化变换：
        # x = x - self.mean: 广播减法，将每个样本每个通道的数据减去对应均值。由于 self.mean 形状是 (B,1,C)，减法会扩展到整个时间维，每个值都减去该样本该通道的均值。结果 x 现在以0为均值。
        # x = x / self.stdev: 再除以标准差，类似地 self.stdev (B,1,C) 会广播，对每个样本每个通道进行标准化，使得每个通道数据方差为1（在加入 eps 情况下近似）。此时 x 各样本各通道均值约0，标准差约1。
        # 如果 self.affine 为 True，则进一步应用可学习的仿射变换：
        # x = x * self.affine_weight: 乘以缩放系数 $\gamma$。注意 affine_weight 是大小为 (C,) 的一维张量，而此时 x 是 (B,T,C)。广播机制会将 $\gamma_c$ 应用于对应通道的所有时间步和所有样本。
        # x = x + self.affine_bias: 加上偏置 $\beta$，形状 (C,) 同样通过广播加到每个样本每个时间步的对应通道上。经过这一步，标准化数据可以根据学习到的 $\gamma,\beta$ 做线性调整，不再严格均值0方差1，但这样允许模型调整不同通道的归一化程度 ￼——例如某些通道可能需要保留较大方差或特定均值。
        # 返回归一化后的 x。

    def _denormalize(self, x):
        if self.affine:
            x = x - self.affine_bias
            x = x / (self.affine_weight + self.eps*self.eps)
        x = x * self.stdev
        x = x + self.mean
        return x
        # 该函数执行将标准化后的数据还原为原始尺度：
        # 如果使用仿射：
        #     - x = x - self.affine_bias: 先减去偏置 $\beta$。这与归一化阶段最后一步的加偏置相逆。
        #     - x = x / (self.affine_weight + self.eps*self.eps): 再除以缩放 $\gamma$。这里除了 $\gamma$ 还加了一个极小值 $\epsilon^2$ 到分母，可能是出于数值安全，防止 $\gamma$ 恰为0导致除零（但一般不会发生）。可以认为近似就是除以 $\gamma$，对应逆Affine变换：$\hat{Y}*{t,c} = \frac{\tilde{Y}*{t,c} - \beta_c}{\gamma_c}$ 。
        # - 接下来：
        #     - x = x * self.stdev: 乘回标准差，将之前除以标准差的操作还原。因为在 _get_statistics 中 self.stdev保存了归一化时用的每样本每通道标准差，这里逐元素相乘，相当于 $Y^{(t,c)} = \hat{Y}_{t,c} \cdot \sigma_c$ 。
        #     - x = x + self.mean: 加回均值，将数据整体抬升回原来的平均水平，对应 $Y_{t,c} = Y^{(t,c)} + \mu_c$ 。
        # 返回反归一化后的 x。现在 x 应当与原始输入在同一尺度分布下（如果Affine也训练到接近1和0，则完全一致）。
        # 通过 denorm 后，模型的输出可以与原数据直接比较（如计算重构误差或异常分数）
        # 这与 RevIN 论文中的反归一化阶段完全对应 ：先逆仿射变换，再恢复原始均值和方差。最终确保模型输出恢复到原始数据的尺度，以便进行评价。
        # 例如，在异常检测中，我们可能对重构序列应用反归一化，然后和原序列比较，计算误差来判别异常。这一步骤保证比较是公平和有意义的（否则如果数据在归一化空间比较，尺度不同无法直接阈值判断）。

'''
总结 RevIN.py：这个模块提供两个主要接口 'norm' 和 'denorm' 来对数据进行可逆的逐实例归一化。整个过程可以概括为
- 训练/推理前：对于每个序列实例，计算其每个维度的均值和标准差，标准化数据使之均值0方差1，并可选地乘以$\gamma$加$\beta$做线性变换。
- 模型处理：将归一化后的数据送入后续模型（如 MtsCID 的核心网络）进行运算，此时模型无需关心不同序列的原始分布差异，增强了对不同分布的鲁棒性 。
- 输出后：对模型输出执行反归一化，乘回标准差加回均值（先逆仿射）恢复原尺度，从而模型的预测结果可以与原始数据进行意义正确的比较 。

在 MtsCID 中，引入 RevIN 可以缓解训练集与测试集之间潜在的分布漂移 。因为异常检测常面临不同时间段、不同指标统计性质变化的问题（例如节假日流量波动等），RevIN 通过实例归一化降低了这些因素对模型的干扰 。
同时，每个时间序列实例各自归一化，并不会把不同实例混合假设同分布，适应测试阶段未知的新分布 。仿射参数的存在也允许模型在需要时部分还原或调整归一化效果，以避免过度平滑重要特征 。
总的来说，RevIN 提供了一个简单有效的预处理/后处理模块，保证 MtsCID 模型专注于学习“正常模式”的形状，而不受绝对值尺度和慢速变化趋势的影响，从而提升异常检测的准确性。
'''