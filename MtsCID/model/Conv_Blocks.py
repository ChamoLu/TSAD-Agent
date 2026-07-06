import torch
import torch.nn as nn

'''
总结来说，Inception_Block 在前向过程中对输入序列并行应用多个尺寸的一维卷积，然后**求平均融合**这些输出，得到富含多尺度信息的特征表示。
这实现了论文中所述的利用卷积捕获**粗粒度时间依赖**的思想，同时由于多尺度融合，能更好地强调序列中的显著模式并忽略微小的噪声扰动。这一模块在 MtsCID 的**下分支（变量间依赖编码器网络）中用于对每个变量的序列进行卷积处理，形成表示 $T$。
正如论文描述，下分支首先用特定卷积核的卷积在时间域捕获各变量的局部依赖，获得表示 $T$ 。代码中实际通过多种核尺寸并行卷积实现了这一过程，并通过 depthwise 分组卷积保证对每个变量各自处理。
这样的粗粒度卷积表示有两个好处：(1)提升单个时间步语义、对噪声更鲁棒，(2) 缓解不同时间步错位和变量不同步问题 。这些优势与论文对该卷积步骤功能的分析完全一致。
值得一提的是，论文指出该卷积输出 $T$ 会进一步转换到频域并通过频域 Transformer 捕获变量间关系，最后与**正弦原型交互模块**结合。也就是说，Conv_Blocks.py 定义的模块完成了下分支的第一步（时间域粗粒度卷积）。
随后，在模型其他部分（非本文件代码）中，输出将经过离散傅里叶变换 (DFT) 得到频率成分，再经过“fc-Transformer”（频率成分 Transformer）在频域沿变量维度学习跨变量依赖 。得到的表示转换回时域并经残差连接和层归一化，得到最终的下分支表示 $O$ 。
此时，MtsCID 引入**正弦原型交互模块（p-i Module）**，用一组固定的正弦函数原型与表示 $O$ 进行交互 。这些原型是不同周期的正弦函数组合，作为固定记忆项，与 $O$ 点积并Softmax得到权重，用于强调跨变量关系的模式 。
这样做可将复杂的变量组合模式简化为有限集合，提升模型对正常变量间关系模式的学习能力，并提高鲁棒性和检测精度 。正弦原型模块避免了训练不稳定性，并利用不同周期正弦函数的固定组合使跨时间步的模式更加显著。
简言之，下分支整体流程是：**卷积提取粗粒度时间模式 -> 频域 Transformer 学习变量间关系 -> 正弦原型交互突出常见模式**  。而其中本文件 Inception_Block 实现的正是第一步卷积部分，为后续步骤提供稳健的粗粒度特征基础 。
'''
'''
综上，Conv_Blocks.py 定义的 Inception_Block 模块通过并行多尺度卷积捕获序列的粗粒度时间模式，并在 MtsCID 模型下分支中用于增强单变量的时间依赖特征表示，为后续频域变换和跨变量注意力奠定基础 。
'''
class Inception_Block(nn.Module):
    '''
    Conv_Blocks.py 文件主要定义了一个 Inception 风格的一维卷积模块，用于提取粗粒度的时间依赖特征。根据论文，这种卷积模块被用于变量间依赖关系编码器网络（i-Encoder）的开头部分，以在时间域捕捉局部粗粒度模式。
    论文指出，对时间序列应用一维卷积可以提供两个关键优势：其一是捕捉粗粒度时间依赖增强了单个时间步的语义表示（提高抗噪性），其二是缓解了数据收集中的时间步错位和不同变量不同步的问题。这正是该模块的设计初衷。
    '''
    def __init__(self, in_channels, out_channels, kernel_list=[1, 3, 5], groups=1, init_weight=True):
        super(Inception_Block, self).__init__()
        # Inception_Block 继承自 nn.Module，表示这是一个自定义的神经网络模块。构造函数接受参数：in_channels 输入通道数，out_channels 输出通道数，kernel_list 卷积核尺寸列表（默认为 [1, 3, 5]），groups 分组数（默认为1），以及 init_weight 指示是否初始化权重。
        # Inception_Block 的命名和设计灵感来自 Inception结构，即 使用多种尺寸的卷积核并行提取特征，以捕获不同尺度的模式。这样做可以同时获取不同时间跨度的特征，有助于捕获粗粒度的时间依赖模式。
        # 值得注意的是，参数 groups 默认=1 表示普通卷积；如果将 groups 设置为等于 in_channels，则会实现深度可分离卷积（每个通道各自卷积而不混合），从而保证各变量/通道的卷积独立进行。
        # 例如，论文的方法中变量间依赖的卷积通常对每个变量单独作用（即不混合通道），以在不破坏变量间关系的前提下平滑每个变量的时间模式。这可以增强单变量序列模式的鲁棒性并对齐不同变量的步调。
        # 因此，在实际使用中常将 groups 设为输入通道数，实现每个变量各自卷积，缓解不同变量信号不同步的问题

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_list = kernel_list
        # 这里将输入/输出通道数和卷积核列表保存为类成员变量，方便在方法中使用。kernel_list 包含多个卷积核尺寸，表示将并行使用这些不同长度的卷积滤波器。

        kernels = []
        for i in self.kernel_list:
            kernels.append(nn.Conv1d(in_channels,
                                     out_channels,
                                     kernel_size=i,
                                     padding='same',
                                     padding_mode='circular',
                                     bias=False,
                                     groups=groups))
        self.convs = nn.ModuleList(kernels)
        # 这段代码遍历每个卷积核尺寸 i，为每个尺寸创建一个一维卷积层（nn.Conv1d），并将它加入 kernels 列表。关键参数解释如下：
        # in_channels 和 out_channels：每个卷积层的输入输出通道，与模块参数一致。
        # kernel_size=i：卷积核长度为当前迭代的值，例如1、3、5等。核尺寸不同可以捕获不同时间尺度的模式：小核捕捉细节变化，大核捕捉更平滑的长期趋势。
        # padding='same'：使用“same”填充，即在输入序列两端填充适当数量，使卷积输出长度与输入相同。这样可以方便地保持时间序列长度不变，避免裁剪边缘。PyTorch 2.x 版本支持 padding='same' 自动进行对称填充。
        # padding_mode='circular'：环形填充模式，表示填充时从序列另一端取值（循环）而非填充零值。环形填充可减轻卷积核在序列边缘处由于零填充值引入的不真实效应，假定序列开头和结尾连接，是一种处理时间序列边界的方法。
        # bias=False：不使用偏置项。通常在有BatchNorm或其他归一化时可省略偏置，这里禁用偏置使卷积纯粹提取变化特征。
        # groups=groups：使用传入的分组数。默认1表示标准卷积，若等于输入通道数则实现每通道独立卷积（深度卷积）。如上所述，将 groups 设为 in_channels 可以使每个变量的卷积滤波彼此独立，提取每个变量自身的时间模式，而不直接混合变量间信号，有助于后续跨变量关系的学习
        # nn.ModuleList(kernels) 将卷积层列表封装为模块列表并赋给 self.convs。这样这些卷积子模块会被注册到 Inception_Block 内，参与模型参数管理和梯度更新。

        if init_weight:
            self._initialize_weights()
            # 如果构造函数参数 init_weight=True（默认就是True），则调用内部方法 _initialize_weights() 对卷积层进行权重初始化。良好的权重初始化有助于模型更快收敛并提高性能。

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        # 这个方法遍历模块内的所有子模块 (self.modules() 会遍历自身及子层)。对于每一个卷积层 (nn.Conv1d 实例)，使用Kaiming He正态初始化（nn.init.kaiming_normal_）来初始化卷积核权重。
        # 参数 mode='fan_out' 和 nonlinearity='relu' 指定了He初始化适用于ReLU激活的设置，确保权重初始分布适中，有助于训练稳定。
        # 由于创建卷积时已将 bias=False，一般不会有偏置；但代码仍防备性地检查 m.bias，如果存在则将偏置初始化为0（常规做法）。

    def forward(self, x):
        res_list = []
        for i in range(len(self.kernel_list)):
            res_list.append(self.convs[i](x))
        res = torch.stack(res_list, dim=-1).mean(-1)
        return res
        # forward 方法定义了模块的前向计算逻辑。当输入张量 x 通过模块时，将依次经过这里的代码：
        # res_list = [] 初始化结果列表，用于收集各卷积核的输出。
        # for i in range(len(self.kernel_list)): 遍历每一种卷积核尺寸对应的卷积层：
        #     self.convs[i](x) 对输入执行第 i 个卷积操作。由于 padding='same'，每个卷积输出与输入 x 在时间长度维度相同，但在通道维度为 out_channels。
        #     每个卷积提取了一种尺度的特征。例如，核大小5的卷积输出捕获长度5的局部模式，核大小1则相当于恒等映射（或逐点卷积）捕获每个时间点自身的特征。将每个卷积输出附加到 res_list 列表。
        # 完成循环后，res_list 包含了形状相同的张量（批次数 × 输出通道 × 时间步长），每个对应一种卷积核的输出。接下来：
        #     torch.stack(res_list, dim=-1) 会将列表中的张 量沿新维度叠加，形成一个新的张量。这里 dim=-1 表示在最后一维叠加，因此叠加后的张量形状为 *(B, out_channels, 时间步长, K)*，其中 *K* 是卷积核数量（即 len(kernel_list))。
        #     .mean(-1) 对最后一维取均值，即对不同卷积核的结果取平均。这一步相当于将多种尺度的卷积特征进行融合，得到综合的表示 res。融合多尺度特征能够提高鲁棒性，让模型同时关注短期和长期模式。
        #     论文提出**粗粒度**的依赖模式捕获，即非单一点细粒度，而是综合多种时间范围的信息 。这里的多尺度卷积平均正契合了这一思想：通过不同尺度卷积提取的模式取平均，可以捕获粗粒度的时间模式，对局部噪声或细微变化不敏感，从而突出主要模式 。
        # return res 返回融合后的结果张量。输出形状与输入 x 相同：(批次, out_channels, 时间长度)，因为经过卷积和融合后通道变为所设定的 out_channels，时间步长保持不变。


# Example usage
if __name__ == "__main__":
    kernel_sizes = [3, 5, 7]  # Example kernel sizes
    model = Inception_Block(in_channels=3,
                            out_channels=6,
                            kernel_list=kernel_sizes,
                            groups=3
                            )
    print(model)
    # 这一段放在 if __name__ == "__main__": 块下，仅在直接运行该脚本时执行，用于演示 Inception_Block 的用法。
    # 这里创建了一个卷积核列表 kernel_sizes = [3, 5, 7] 作为例子，然后实例化 Inception_Block：输入通道=3，输出通道=6，卷积核列表使用上述三个尺寸
    # groups=3（将分组数设为3，等于输入通道数，实现每通道单独卷积）。实例化后打印 model 对象，将输出模块的结构信息。

    # Test the model with a random input
    input_tensor = torch.randn(5, 3, 32)  # Batch size of 5, 3 channels, 32 feature
    output = model(input_tensor)
    print(output.shape)  # Should be [1, 10] for 10 classes
    # 这里构造了一个随机张量 input_tensor 来测试模型。torch.randn(5, 3, 32) 生成形状为 (5, 3, 32) 的张量，表示批次大小5、通道数3、序列长度32的随机输入。
    # 随后将其传入模型 output = model(input_tensor)，得到输出张量，并打印输出形状。按逻辑推断，Inception_Block 会对每个输入样本的每个通道进行多尺度卷积，
    # 输出形状应为 (5, 6, 32)：批次5、输出通道6、序列长度32（与输入长度相同）。
    # 代码中打印形状的注释却写着 “Should be [1, 10] for 10 classes”，这与此模块实际功能无关，像是残留的注释错误。正确来说，这里打印的输出形状应是 (5, 6, 32)。这一示例代码只是用于开发调试，表明模块可以正确处理输入并输出预期形状。
