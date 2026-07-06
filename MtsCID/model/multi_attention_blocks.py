import torch
import torch.nn as nn
from einops import rearrange
from model.attn_layer import AttentionLayer

'''
multi_attention_blocks.py 文件定义了一个多头注意力块类 Inception_Attention_Block，用于在上分支（时间自编码器网络 t-AutoEncoder）中从多尺度的补丁中学习时间依赖关系 。
根据论文，上分支在将每个变量的序列经过频域变换和频域Transformer处理后，会重新回到时间域，并通过一系列组内变量的时间序列注意力网络（ts-Attention networks）在多尺度补丁的注意力图上学习时间依赖 。
简言之，模型把时间序列切分为不同大小的片段（patch），计算这些片段之间的注意力图，然后将注意力结果映射回原始序列长度并聚合，以捕获粗粒度的时间依赖模式。Inception_Attention_Block 正是实现这一系列操作的模块。
'''

'''
Inception_Attention_Block 完全贴合了论文中 ts-Attention 多尺度模块 的流程，实现从注意力图（注意力矩阵）中学习时间依赖并重构序列表示的功能。
这一模块在 MtsCID 模型上分支中应用：上分支每个变量先经过频域变换与频域Transformer学习频率依赖，然后转换回时域，交给本模块对不同时间尺度的patch应用注意力，提取粗粒度的时间模式，最后进入解码器重构序列。
正如论文所述：“将得到的表示转换回时域，并通过一组组内变量的 ts-Attention 网络，从多尺度分块的注意力图中学习时间依赖关系”
'''

class Inception_Attention_Block(nn.Module):
    def __init__(self, w_size, in_dim, d_model, patch_list=[10, 20], init_weight=True):
        super(Inception_Attention_Block, self).__init__()
        self.w_size = w_size
        self.in_dim = in_dim
        self.d_model = d_model
        self.patch_list = patch_list
        # Inception_Attention_Block 继承自 nn.Module。构造函数参数如下：
        # w_size: 窗口大小，即时间序列子序列长度 L（window size）。例如上分支传入子序列长度。
        # in_dim: 输入特征维度，即每个时间步的维度（可能对应前面频域Transformer输出的嵌入维度 *d* 或类似）。
        # d_model: 注意力模型维度。这个参数将用于定义 AttentionLayer 的内部维度。
        # patch_list: 列出要使用的补丁大小列表，默认为 [10, 20]。表示将时间序列划分为长度为10和20的patch进行多尺度处理。可以根据需要调整，以捕获不同粒度的时间关系。
        # init_weight: 是否初始化线性层权重，默认 True。
        # 将 w_size, in_dim, d_model, patch_list 保存为对象属性。需要注意，patch_list 里的每个值 p 应当能整除 w_size，否则无法正好划分为整数个patch。

        patch_attention_layers = []
        linear_layers = []
        for patch_size in self.patch_list:
            patch_number = w_size // patch_size
            patch_attention_layers.append(AttentionLayer(w_size=patch_number,
                                          d_model=patch_size,
                                          n_heads=1
                                                         )
                                          )
            linear_layers.append(nn.Linear(patch_number, patch_size))
        self.patch_attention_layers = nn.ModuleList(patch_attention_layers)
        self.linear_layers = nn.ModuleList(linear_layers)
        # 这一段核心实现了**多尺度补丁注意力**的构造：
        # - 初始化两个列表：patch_attention_layers 用于存储针对各补丁尺度的注意力层，linear_layers 用于存储对应的线性投影层。
        # - 遍历 patch_list 中每个补丁长度 patch_size：
        #     - 计算 patch_number = w_size // patch_size，即将长度为 w_size 的序列划分为长度为 patch_size 的片段后得到的**片段个数**。例如，若窗口长度60，patch_size=10，则 patch_number=6，即序列可分成6个长度10的片段。
        #     - 构造一个注意力层 AttentionLayer(...) 并加入列表：
        #         - w_size=patch_number：这里传入的 w_size 参数（注意：这个名称与总体窗口长度同名，但在此上下文中意味着**补丁的数量**）。可能用于内部位置编码或AttentionLayer内部对序列长度的设置。传 patch_number 意味着注意力层将处理序列长度为该patch数量的输入。
        #         - d_model=patch_size：模型维度设为补丁长度。这是一个关键设计：令注意力层的特征维度等于补丁长度，使注意力计算在一个空间中进行——这与论文中的公式相符。
        #           论文在公式(6)中表示对于每个patch序列，计算注意力图 $A_{p_i} \in \mathbb{R}^{n_i \times n_i}$，其中 $n_i$ 是patch个数 。这里采用 $d_{\text{model}} = p_i$ 似乎使注意力的输出可以方便地映射回补丁长度。
        #         - n_heads=1：使用单头注意力。因为我们关注的是patch之间的相关性，而非多子空间的不同注意力模式，因此设置单头即可返回一个注意力矩阵（或相应的表征）。论文中提到**attention map**，因此单头注意力足够提供一组注意力权重。
        #         - 这里创建的 AttentionLayer 预期对输入序列进行自注意力，返回注意力输出和（可能）注意力权重。注意，AttentionLayer 未显式提供 dropout 等参数，可能是简化实现以提取注意力图。
        #     - 随后为该patch尺度添加一个线性层：nn.Linear(patch_number, patch_size)。这是一个全连接层，输入维度为 patch_number，输出维度为 patch_size。其作用是在后续将注意力结果从“patch空间”映射回“时间步长空间”。
        #       如论文所述，每个注意力图 $A_{p_i}$ 通过可学习参数 $M_{p_i} \in \mathbb{R}^{n_i \times p_i}$ 映射回原始patch大小 。这里的线性层就对应论文中的 $M_{p_i}$：它以 patch数 $n_i$ 维的输入，输出 patch长度 $p_i$ 维的向量。
        # - 将构造好的 patch_attention_layers 和 linear_layers 列表分别封装为 nn.ModuleList 保存为对象属性。这使得它们被注册为子模块列表，在 forward 中可以逐个使用。
        # 通过以上循环，Inception_Attention_Block 就为每种指定的补丁大小准备了一个注意力层和一个线性映射层。一种补丁大小对应一种“视野”，例如 patch10 关注更局部的时间片段关系，patch20 则关注更长时间段的关系。
        # 多尺度 patch 的设计类似于 Inception 思想，将不同粒度的注意力结果融合，对应论文所说“从多尺度分块的注意力图中学习时间依赖关系” 。

        if init_weight:
            self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
                    # 与前面的卷积模块类似，这里遍历模块的子模块，对所有线性层（nn.Linear）进行权重初始化。
                    # 采用Kaiming正态初始化线性层权重（合适于ReLU激活，尽管这里线性层后未必直接跟随ReLU，但Kaiming初始化仍是一种良好选择）。偏置若存在则初始化为0。这样可以确保线性映射层在训练初期不会产生偏移，有利于稳定训练。

    def forward(self, x):
        B, _, _ = x.size()
        res_list = []
        # 这是 Inception_Attention_Block 的核心计算过程
        # B, _, _ = x.size(): 获取输入张量 x 的批次大小 B（以及其他两个维度占位符 _表示暂不使用）。按预期，输入 x 形状应为 *(B, L, d_model)*，即批次数、序列长度（窗口长度）和特征维度。
        # 例如，在示例中 x 是 (3, 60, 32)。在论文上分支中，这个 x 可以看作经过频域处理和 iDFT+LayerNorm 后得到的表示 $Z \in \mathbb{R}^{B \times L \times d}$ 。
        # 初始化 res_list = [] 列表用于存放不同patch尺度处理后的结果。

        for i, p_size in enumerate(self.patch_list):
            z = rearrange(x, 'b (w p) c  -> (b c) w p', p=p_size).contiguous()
            # 这里使用 einops.rearrange 将输入张量 x 重排成含补丁的形状。模式字符串 'b (w p) c -> (b c) w p' 表示：
            # - 原张量维度标记为 b (w p) c，其中 b 是批次维度，(w p)表示时间长度L被视作 w * p，c是特征维度。
            # - p=p_size 告诉 rearrange，将 L 划分为若干块，每块长度为 p_size，从而 w = L / p_size 是补丁数量。
            # - 转换后的新形状 (b c) w p：即将原先的批次维度和特征维度合并为一个维度 (b c)，补丁数 w 作为新序列长度维度，补丁长度 p 作为新特征维度。
            #   举例：若 B=3, L=60, c=32，且当前 p_size=10，则 w = 60/10 = 6。重排后 z 形状变为 $((3*32), 6, 10)$，即 $96 \times 6 \times 10$。
            #   这里每一行对应一个样本的一个特征，共有 $B \times c$ 行，每行长度为patch数量6，每个元素有patch长度10的特征。
            #   通俗地说，我们把每个样本每个特征通道的时间序列都分成6段，每段长度10，并把这些段看成新的“小序列”来处理。这样做的目的是在每个特征内部计算时间片段之间的依赖关系，不混合不同特征，以捕捉组内（单变量）的时间依赖 。
            #   这对应论文中提到的“**channel-independent manner**”将 $Z$ 切分成多尺度patch ——每个embedding通道（或者说每个变量）的序列独立地切割成patch。
            #   调用 .contiguous() 确保重排后的张量在内存中是连续的，便于后续操作（有时 rearrange 会返回一个视图，为安全起见转为实际内存布局）。

            _, z = self.patch_attention_layers[i](z)
            # 将重排后的 z 输入第 i 个 patch_attention_layer（对应当前 patch_size）。AttentionLayer 预期返回两个值（通常可能是 (attn_weights, attn_output) 或 (context, output)）。
            # 这里通过 _, z = ... 忽略了第一个返回值，把第二个返回值赋给 z。这暗示 AttentionLayer 的实现可能返回 (None, output) 或 (attn_weights, context) 等。
            # 推测：为了提取注意力映射，AttentionLayer 可能实现的是自注意力机制，其中 $Q=K=V=z$，序列长度为补丁数，embedding维度为补丁长度。
            # 由于 n_heads=1，计算得到 $n_i \times n_i$ 的注意力矩阵，其与数值乘积得到的输出也是 $(b*c) \times n_i \times p_i$ 形状（与输入形状相同）。
            # 这里 z 作为输出，形状应该还是 (b*c, w, p)，即每个序列（每个通道）的每个补丁位置得到一个新的 $p$-维表示。可以认为，这一步通过自注意力在patch序列之间传播信息，得到每个patch的新的表示（融合了整个序列patch间依赖）。
            # 而忽略的第一个返回值可能是注意力权重矩阵 $A_{p_i}$。在论文中，注意力图 $A_{p_i} = \text{Softmax}\big((Z_{p_i} Z_{p_i}^T)/\sqrt{p_i}\big)$ 。
            # 代码没有显式返回或使用 $A_{p_i}$，而是直接利用注意力后的输出继续计算。但线性层可以承担将注意力隐式应用的作用（如下）。

            z = self.linear_layers[i](z)
            # 将经过注意力机制处理的 z 输入对应的线性层。回忆初始化时 linear_layers[i] = Linear(patch_number, patch_size)：输入维度是 patch数量 (n_i = w)，输出维度是 patch长度 (p_i = patch_size)。
            # 然而，当前 z 张量形状是 (b*c, w, p)，即 (batch*channel, patch_number, patch_size)。要应用线性层（期望输入形状最后一维大小=patch_number），需要让 patch_number 作为特征维度。
            # 这里可能存在隐含的张量转置：通常 nn.Linear 会将最后一维看作特征维度进行映射。如果直接传入 (b*c, w, p) 张量，线性层会将大小为 p 的最后一维当作输入特征，与所需的 patch_number=w 不符。
            # 因此，推断 AttentionLayer 输出 z 时已经交换了维度，将补丁数作为最后一维。例如 AttentionLayer 可能返回的是注意力矩阵作用于某个值矩阵后的结果，其形状可能是 (b*c, p, w)。也可能 AttentionLayer 在实现中对 z 进行了转置。所以这里直接对 z 调用线性层是成立的

            z = rearrange(z, '(b c) w p -> b (w p) c', b=B).contiguous()
            # 这一步将前面合并的 (b c) 维度拆开，并将补丁重新拼接成原始序列：
            # 输入形状假设为 (b*c, n_i, p_i) (经过 linear 后)，模式 '(b c) w p -> b (w p) c' 会：
            # 把前面的 (b c) 维展开为原来的 b 和 c 两个维度（这里通过参数 b=B 告诉 einops 原批次大小是多少，以便正确拆分）。
            # - 将 w（patch数量）和 p（patch长度）维度组合回单个维度 (w p)，这相当于把补丁拼回完整序列，c 维保持为特征维度。
            # - 结果张量 z 的形状重新变成 (B, L, c)，即批次×序列长度×特征维度，和输入 x 形状一致
            # （除了特征维度是否变化需注意：这里特征维度原为 d_model，处理后仍应为 d_model，因为 linear输出维度 = p_i，然而 p_i 各不相同，如何处理？实际上由于我们对每个patch尺度分别处理并重组，下一步会融合这些结果）。
            #  .contiguous() 同样用于保证内存连续。

            res_list.append(z)
            # 收集结果： 将得到的 z 加入 res_list。
        res = torch.stack(res_list, dim=-1).mean(-1)
        # 当循环处理完所有补丁尺寸后，res_list 包含了每种 patch尺度重构出的张量，它们形状都是 (B, L, c)（与输入相同维度）。
        # 使用 torch.stack(res_list, dim=-1) 在最后一维叠加，得到 shape (B, L, c, m)，其中 m = len(patch_list) 是使用的尺度数量。然后对最后一维求均值 .mean(-1)，即对不同patch尺度的结果取平均，得到融合后的结果 res，形状恢复为 (B, L, c)。
        return res


# Example usage
if __name__ == "__main__":
    kernel_sizes = [3, 60, 16]  # Example kernel sizes
    model = Inception_Attention_Block(w_size=60, in_dim=16, d_model=32)
    print(model)
    # 在 main 代码块中演示模块使用。这里定义了一个 kernel_sizes 列表。然后创建 Inception_Attention_Block 实例：w_size=60（窗口长度60）、in_dim=16、d_model=32、其余参数采用默认（patch_list=[10,20]）。
    # 打印模型结构信息，确认子模块列表（AttentionLayer和Linear层）的构成正确。

    # Test the model with a random input
    input_tensor = torch.randn(3, 60, 32)  
    output = model(input_tensor)
    print(output.shape)  
    # 生成随机输入张量 input_tensor 形状为 (3, 60, 32)，表示批次3，序列长度60，特征维度32（对应于 d_model）。
    # 将其传入模型得到输出，并打印输出形状。按照模块逻辑，输出应与输入形状相同，即 (3, 60, 32)：批次3，长度60，特征维32。
    # 代码中注释“Should be [1, 10] for 10 classes”明显不适用于本模块，同样是一个无关的残留注释，实际输出不会是 [1,10]。此处应是开发者拷贝示例时的疏忽。总之，这段示例确认模块在给定输入下能够运行，并产生与输入相匹配的输出形状。
    # 通过对 multi_attention_blocks.py 的逐行解析，我们看到 Inception_Attention_Block 实现了 MtsCID 中时间依赖学习的关键步骤——利用多尺度的自注意力机制，
    # 从每个变量序列的多个时间分辨率上提取依赖特征并融合 。这与下分支的卷积模块互补：上分支注重组内时间模式，捕获粗粒度的时间依赖，下分支注重变量间关系捕获粗粒度的跨变量依赖。
    # 两个分支相辅相成，分别在各自领域提取正常模式，从而提升异常检测性能。
