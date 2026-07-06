import copy
import argparse
from torch.backends import cudnn # 导入 PyTorch 的 CUDNN 后端配置。启用 cudnn.benchmark=True 可以让卷积算法根据输入大小自动选择最快实现，提升训练速度。 
from utils.utils import * # 导入项目内的实用函数。utils/utils.py 包含设置随机种子、创建目录、绘图、输出指标等辅助工具
# ours
from solver import Solver # 导入核心训练类 Solver，其内部封装了数据加载、模型构建、训练、评估和保存模型等逻辑。Solver 会调用 t‑AutoEncoder、i‑Encoder 和正弦原型交互模块等组件。
'''
main.py 作为整个 MtsCID 项目的启动脚本，负责读取命令行参数、根据配置初始化 Solver、反复训练并评估模型，并记录最终结果。
它把论文中的算法流程（多尺度卷积与注意力、双分支自编码器、对比学习与原型矩阵）连接为一个可运行的实验管道。
'''

def str2bool(str_v):
    return str_v.lower() in ['true']
    # 该函数把字符串转化为布尔值。命令行中布尔参数传入的是字符串，例如 'True' 或 'False'，该函数用小写判断是否为 'true'，在其它模块中使用。

def main(config_setting):
    # set_seed(42)
    cudnn.benchmark = True
    # 让 CUDNN 通过自动调优选择最佳卷积实现，提升不同输入大小下的运算效率

    if not os.path.exists(config_setting.model_save_path):
        mkdir(config_setting.model_save_path)

    result_list = {key: [] for key in metric_list}
    # 创建 result_list 字典，用于记录多次运行的各项评估指标
    solver = Solver(vars(config_setting))
    # 实例化 Solver，并将配置转成字典传入

    for i in range(config_setting.run_times):
        # To ensure that the model parameters are re-initialized before each round.
        solver.model_init(vars(config_setting))
        print(f"--------------------------- Round {i+1} -----------------------------")
        if not config_setting.test_only:
            solver.train()
        eval_results = solver.test(vars(config_setting))

        for key, value in eval_results.items():
            if key in result_list:
                result_list[key].append(value)
        # 通过 for i in range(config_setting.run_times) 循环训练多次，以减小随机性。每次循环：
        # 调用 solver.model_init 重新初始化模型权重，确保各轮独立
        # 若 test_only 为 False，则调用 solver.train() 进行训练；然后调用 solver.test() 在验证集/测试集上评估并返回评价指标
        # 遍历 eval_results 的键值，将每次运行的指标加入 result_list

    final_eval_results = {key: '-' for key in metric_list}
    for key, value in result_list.items():
        if key in final_eval_results:
            final_eval_results[key] = f"{np.mean(result_list[key]):.4f}±{np.std(result_list[key]):.4f}"
    dump_final_results(vars(config_setting), final_eval_results)
    # 训练结束后创建 final_eval_results，初值为 '-'，随后计算各指标均值和标准差，
    # 如 np.mean(result_list[key])±np.std(result_list[key])。这样做是为了报告多次运行的平均性能。
    # 调用 dump_final_results 将最终结果保存到 CSV 文件

    print(f"----------------------{config_setting.dataset} Evaluation Results-----------------------")
    print(f"{Color.CYAN}pc-a: {np.mean(result_list['pc_adjust']):.4f}{result_list['pc_adjust']}{Color.RESET}")
    print(f"{Color.CYAN}rc-a: {np.mean(result_list['rc_adjust']):.4f}{result_list['rc_adjust']}{Color.RESET}")
    print(f"{Color.CYAN}f1-a: {np.mean(result_list['f1_adjust']):.4f}{result_list['f1_adjust']}{Color.RESET}")
    # 以彩色格式打印调整后的精度 pc_adjust、召回率 rc_adjust 和 F1 分数 f1_adjust 的均值及各轮具体值，Color 类定义了终端颜色常量，用于输出更易读的结果。

    del solver
    # 最后 del solver 手动删除 Solver 对象，释放 GPU 显存和 CPU 内存。虽然 Python 会自动垃圾回收，但在循环中显式删除可以减少显存占用

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # 通过 argparse.ArgumentParser() 定义大量参数，用于灵活配置模型和实验：

    parser.add_argument('--framework', nargs="+", type=str, default=['MtsCID'])
    parser.add_argument('--test_only', default=False, action="store_true")
    # --framework 用于选择框架（默认为 MtsCID）
    # test_only 若设为 True，则仅执行测试部分，不进行训练，这在已有训练好的模型时可用。

    # Data setting
    parser.add_argument('--dataset', type=str, default='NIPS_TS_Water')
    parser.add_argument('--win_size', type=int, default=100)
    parser.add_argument('--data_path', type=str, default='./data/NIPS_TS_GECCO/')
    # --dataset：指定数据集名称，例如 NIPS_TS_Water。该字段用于选择正确的文件和标准化参数。
    # --win_size：滑动窗口长度 L，决定每次输入模型的时间步数。论文中强调通过滑动窗口切分时间序列，注意力和卷积操作在窗口内捕获模式。
    # --data_path：数据存放目录。

    # Model setting
    parser.add_argument('--input_c', type=int, default=9)
    parser.add_argument('--output_c', type=int, default=9)
    parser.add_argument('--d_model', type=int, default=9)
    parser.add_argument('--temperature', type=float, default=0.1)
    # --input_c / --output_c：输入和输出的变量数 C。默认两者相同，表示自编码器尝试重构原始多变量序列。
    # --d_model：中间嵌入维度。t‑AutoEncoder 中的 fc-Transformer 使用该维度作为隐藏单元。
    # --temperature：正弦原型交互模块中 softmax 的温度参数 $\tau$，控制注意力分布的锐度

    # 编码器层与网络结构
    parser.add_argument('--encoder_layers', type=int, default=1, help="The number of encoder layers")
    parser.add_argument('--branches_group_embedding', type=str, default='False_False', choices=['True_True', 'True_False', 'False_True', 'False_False'], help="The parameter is used only when conv1d is employed in the encoder layer")
    parser.add_argument('--multiscale_kernel_size', nargs="+", type=int, default=[5], help="The parameter is used when conv1d is employed in the encoder layer")
    parser.add_argument('--multiscale_patch_size', nargs="+", type=int, default=[10, 20], help="The parameter is used when multi-attention is employed in the encoder layer")
    parser.add_argument('--branch1_networks', nargs="+", type=str, default=['fc_linear', 'intra_fc_transformer', 'multiscale_ts_attention'])
    parser.add_argument('--branch1_match_dimension', type=str, default='first', choices=['none', 'first', 'middle', 'last'])
    parser.add_argument('--branch2_networks', nargs="+", type=str, default=['multiscale_conv1d', 'inter_fc_transformer'])
    parser.add_argument('--branch2_match_dimension', type=str, default='first', choices=['none', 'first', 'middle', 'last'])
    # --encoder_layers：编码器层数。目前实验中多数设置为 1，也提供接口便于叠加更多层。
    # --branches_group_embedding：字符串形如 'True_False'，表示两条分支的卷积是否采用分组嵌入。组卷积可以让不同变量组共享卷积核，减小参数量，同时防止不同变量间干扰。
    # --multiscale_kernel_size：卷积核大小列表。例如 [5] 表示单尺度 1D 卷积核大小为 5；若给出 [3,5,7] 则表示多尺度卷积并行操作。多尺度卷积是论文中捕获粗粒度时间依赖的关键组成，也称 “multiscale_conv1d”。
    # --multiscale_patch_size：用于多头注意力中 patch 的大小列表，针对 multiscale_ts_attention 模块。论文指出对每个时间窗口再分割为不同尺度的 patch，并通过自注意力学习 patch 之间的关系，以捕获更长和更短的时间结构。
    # --branch1_networks：t‑AutoEncoder 使用的模块序列，默认包括 fc_linear（全连接映射）、intra_fc_transformer（基于全连接的 Transformer，工作在频率域）以及 multiscale_ts_attention（多尺度时间注意力）。这些模块实现论文“粗粒度时间依赖”部分。
    # --branch1_match_dimension：指定 t‑分支各模块输入输出维度匹配方式（例如 'first' 表示保持输入维度）。
    # --branch2_networks：i‑Encoder 使用的模块序列，默认包括 multiscale_conv1d（多尺度卷积捕获局部时间模式）和 inter_fc_transformer（全连接 Transformer 捕获变量间关系）。该分支与 memory_guided 原型交互模块协同实现论文“粗粒度 inter‑variates 关系”部分
    # --branch2_match_dimension：指定 i‑分支模块之间维度匹配策略。

    # 解码器设置
    parser.add_argument('--decoder_networks', nargs="+", type=str, default=['linear'])
    parser.add_argument('--decoder_layers', type=int, default=1, help="The number of encoder layers")
    parser.add_argument('--decoder_group_embedding', type=str, default='False', choices=['True', 'False'])
    # --decoder_networks：解码器模块列表，默认为简单的 linear 层，用于将编码表示映射回原始维度 C。
    # --decoder_layers：解码器层数。
    # --decoder_group_embedding：是否在解码器中使用组卷积，通常与编码器设置

    # 嵌入和记忆设置
    parser.add_argument('--embedding_init', type=str, default='normal')
    parser.add_argument('--memory_guided', type=str, default='sinusoid')
    # --embedding_init：嵌入初始化方式，如 normal（高斯）、uniform、orthogonal 等。嵌入指模型中用于映射输入到隐藏空间的参数矩阵。
    # --memory_guided：选择正弦原型矩阵的生成方式，如 sinusoid。这是论文中“正弦原型交互模块”的配置项，对应 create_memory_matrix 函数。

    # 重构损失聚合方式
    parser.add_argument('--aggregation', type=str, default='normal_mean', choices=['normal_mean', 'mean', 'max', 'harmonic_mean', 'harmonic_max'])  # for recon loss, max is better than mean
    # 选择重构误差的聚合方式，论文提出同时利用时间域和频域的重构误差，通过调和平均（harmonic_mean/harmonic_max）消除单域噪声的影响，使异常分数更加鲁棒。

    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--model_save_path', type=str, default='checkpoints')
    # 控制 PyTorch DataLoader 的工作线程数、模型保存目录

    # Training setting
    parser.add_argument('--num_epochs', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--patience', type=int, default=10)
    parser.add_argument('--peak_lr', type=float, default=2e-3)
    parser.add_argument('--end_lr', type=float, default=5e-5)
    parser.add_argument("--weight_decay", default=5e-5, type=float)
    parser.add_argument("--warmup_epoch", default=0, type=int)
    # 训练的总epochs、batch大小、早停等待轮数 (patience)、学习率起止值 (peak_lr/end_lr)、权重衰减 (weight_decay) 以及 warm-up 轮数。
    # Solver 会根据这些参数构建 PolynomialDecayLR 学习率调度器。

    # Device parameter
    parser.add_argument('--device', type=str, default="cuda:0")
    # 指定计算设备，如 cuda:0。如果系统无可用 GPU，则会在后续代码中自动回退到 CPU。

    # Loss weight hyperparameters
    parser.add_argument('--alpha', type=float, default=1.0)
    # 调节熵正则化项的权重。论文中通过熵正则化控制原型注意力的分布，让模型关注少数重要模式

    # Parameters setting for evaluation
    parser.add_argument('--threshold_setting', type=str, default='optimal', choices=['preset', 'optimal'])
    parser.add_argument('--anomaly_ratio', type=float, default=1.0, help="The parameter is used when threshold_setting is set as 'preset'")
    parser.add_argument('--run_times', type=int, default=5, help="The number of times to run for evaluating the result")
    # --threshold_setting：异常阈值的确定方式，preset 表示按给定比例选择阈值；optimal 表示在验证集上寻找使 F1 分数最大的阈值。论文中指出，对于半监督异常检测，合理的阈值选择是影响性能的关键因素。
    # --anomaly_ratio：当 threshold_setting='preset' 时，代表被认为是异常的样本比例。
    # --run_times：与主函数中的 run_times 对应，决定重复实验次数。

    # debug parameters
    parser.add_argument('--plot_data', type=str, default='False', choices=['True', 'False'])
    parser.add_argument('--anomaly_only', type=str, default='False', choices=['True', 'False'])
    # --plot_data：若为 'True'，训练过程中会绘制原始和重构序列的对比图，用于可视化分析。
    # --anomaly_only：若为 'True'，则只在包含异常的时间段评估模型，通常用于加速调试。

    default_args = parser.parse_args()
    # 解析所有命令行参数到 Namespace 对象。

    for frame_work in default_args.framework:
        config = copy.copy(default_args)
        config.framework = frame_work
        # 程序支持同时运行多个框架。迭代 framework 列表，为每种框架创建独立配置 config，copy.copy 生成浅拷贝，避免在修改 config 时影响原配置。

        print(f'--------------------- Framework: {frame_work} -------------------')
        if ('conv1d' not in config.branch1_networks) or (len(config.branch1_networks) < 2):
            updated_group_embedding = 'False'
        else:
            updated_group_embedding = config.branches_group_embedding.split('_')[0]
            # 判断 t‑分支是否包含 conv1d 模块。若不包含或模块数少于 2 个，则关闭组卷积。这是因为组卷积只有在卷积层与另一个层组合时才有意义；若只有一个模块，组卷积无法体现。否则，保留用户指定的第一段开关。
            # 组卷积来自论文“组嵌入”的思想，用于不同变量组共享卷积核，减少参数并提高特征共享能力。
            # (len(branch1_networks)<2) 有些奇怪，可能原意是判断是否同时使用卷积和 Transformer；若只使用卷积或只使用 Transformer，组卷积可能没有意义。

        if ('conv1d' not in config.branch2_networks) or (len(config.branch2_networks) < 2):
            config.branches_group_embedding = updated_group_embedding + '_False'
        else:
            config.branches_group_embedding = updated_group_embedding + '_' + config.branches_group_embedding.split('_')[1]
            # 如果没有使用 conv1d 或模块数少于 2，则把 branches_group_embedding 的第二段设为 'False'；
            # 否则保留原来第二段。最终 branches_group_embedding 形如 'True_False' 或 'False_True' 表示 t‑分支和 i‑分支是否启用组卷积。

        config.device = torch.device(config.device if torch.cuda.is_available() else "cpu")
        # 若当前环境有可用的 GPU，则使用指定的 GPU；否则退回 CPU

        args = vars(config)
        # print('--------------------------- Parameters Setting-------------')
        for k, v in sorted(args.items()):
            print('%s: %s' % (str(k), str(v)))
            # 将配置对象转为字典并按键排序逐一打印，便于确认实验参数完整
        # print('--------------------------- End -------------------------------')
        main(config)
        # 最后调用前面定义的 main 函数开始训练与评估。
