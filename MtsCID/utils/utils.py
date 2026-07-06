import os
import random
import csv
import torch
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
try:
    from kmeans_pytorch import kmeans
except Exception:
    kmeans = None
import time
try:
    import pandas as pd
except Exception:
    pd = None
from datetime import datetime
# os、time 用于文件和时间操作；random、numpy、torch 用于随机数和张量计算；
# torch.nn.functional as F 提供损失和激活函数；
# Variable 是早期 PyTorch 用来封装张量并记录梯度的类，现已不常用；
# kmeans_pytorch.kmeans 在此文件中未使用，但可能在其他模块（如原型生成）用到；
# pandas 用于保存结果到 CSV；
# datetime 用于生成带时间戳的文件名。
'''
utils.py提供训练和评估所需的工具，包括随机种子设置、目录管理、颜色输出以及关键的时频域损失融合函数、指标列表和结果记录函数。
这些工具使得主代码更加简洁，并体现论文中对时间域和频域联合建模以及多指标评估的要求
'''

class Color:
    RED = '\033[91m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    MAGENTA = '\033[95m'
    CYAN = '\033[96m'
    RESET = '\033[0m'
    # 定义终端颜色代码，方便在终端输出彩色文本，如在 main.py 中打印评估指标时使用

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    np.random.seed(seed)
    random.seed(seed)
    # torch.backends.cudnn.benchmark = False
    # torch.backends.cudnn.deterministic = True
    # 设置 CPU 和 GPU 的随机种子，保证结果可复现 。对于多 GPU 训练也设置了 cuda.manual_seed_all。注释掉的两行可以让 cuDNN 使用确定性算法，但会降低速度。

def to_var(x, volatile=False):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x, volatile=volatile)
    # 早期 PyTorch 需要用 Variable 包装张量才能计算梯度。函数检查 GPU 是否可用，如果可用则将输入张量移动到 GPU。此函数未在当前版本代码中调用。

def mkdir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def harmonic_loss_compute(t_loss, f_loss, operator='mean'):
    """
    Parameters
    ----------
    t_loss: loss in the time domain. It is expected in the shape [B, L, C]
    f_loss: loss in the frequency domain. It is expected in the shape [B, L, C]
    operator: ['mean', 'max', 'harmonic_mean', 'harmonic_max']

    Returns
    -------
    return harmonic_loss
    """
    assert operator in ['normal_mean', 'mean', 'max', 'harmonic_mean', 'harmonic_max']
    # t_loss 和 f_loss 是在 Solver 的训练过程中计算的时域与频域重构误差，形状为 [B, L, C]（批大小、时间/频率长度、通道数）。
    # operator 指定融合方式，必须是预定义的五种之一 ￼。

    t_wa = t_loss.mean(dim=-2, keepdim=True)
    f_wa = f_loss.mean(dim=-2, keepdim=True)
    t_wm = t_loss.max(dim=-2, keepdim=True)[0]
    f_wm = f_loss.max(dim=-2, keepdim=True)[0]
    # 对时间域和频域在长度维度上分别求平均和最大。比如，t_wa 是每个通道在时间维度上的平均重构误差，f_wm 是在频率维度上的最大重构误差。这些统计量用于生成 softmax 权重。

    if operator == 'mean':
        loss = (t_loss * torch.softmax(f_wa, dim=-1)).max(dim=-1)[0]
    elif operator == 'max':
        loss = (t_loss * torch.softmax(f_wm, dim=-1)).max(dim=-1)[0]
    elif operator == 'harmonic_mean':
        nt_loss = (t_loss * torch.softmax(f_wa, dim=-1)).mean(dim=-1)
        nf_loss = (f_loss * torch.softmax(t_wa, dim=-1)).mean(dim=-1)
        loss = (nt_loss + nf_loss) / 2
    elif operator == 'harmonic_max':
        nt_loss = (t_loss * torch.softmax(f_wm, dim=-1)).max(dim=-1)[0]
        nf_loss = (f_loss * torch.softmax(t_wm, dim=-1)).max(dim=-1)[0]
        loss = (nt_loss + nf_loss) / 2
    elif operator == 'normal_mean':
        loss = t_loss.mean(dim=-1) * f_loss
        # 随后根据 operator 使用不同策略融合：
        # mean：将时间域每个点的误差乘以频域平均误差的 softmax 权重，再取最后一维（通道维度）最大值。这样会让频域中高误差位置对时间域误差更敏感。
        # max：相同思路但使用频域最大误差
        # harmonic_mean：用交叉 softmax 权重分别求时间域和频域的加权均值，然后取二者平均 ￼。这一设计来源于论文中为了避免单一域噪声而采用调和平均的思想。
        # harmonic_max：使用最大值代替均值再取平均 ￼。
        # normal_mean：简单地将时间域均值乘以频域误差 ￼。
        # 函数最后返回融合后的损失。在训练中，这一融合函数让模型同时关注时域和频域，减少只在某一域出现异常时的误报，符合 MtsCID 中时间-频域联合建模的核心理念。
    return loss

metric_list = ['pc_adjust', 'rc_adjust', 'f1_adjust', 'af_pc', 'af_rc', 'af_f1', 'vus_roc', 'vus_pr', 'auc_pr', 'auc_roc', 'thresh', 'trt', 'tst']
# pc_adjust、rc_adjust 和 f1_adjust 是在异常检测任务中常用的调整后精度、召回率和 F1 分数（按滑窗对齐） ￼。
# af_pc、af_rc、af_f1 代表按照人工标注的异常区间计算的指标，不做滑窗调整。
# vus_roc、vus_pr、auc_pr、auc_roc 分别是受试者工作特征曲线和精确率召回曲线下的面积，用于评价阈值敏感性。
# thresh 保存最佳阈值，trt、tst 记录训练和测试耗时。

def dump_final_results(params, eval_results):
    # 结果保存函数
    benchmark_results = []
    timestamp = ['time']

    config_list = ['framework', 'run_times', 'dataset', 'win_size', 'd_model',
                   'branches_group_embedding', 'multiscale_kernel_size', 'multiscale_patch_size',
                   'branch1_networks', 'branch1_match_dimension',
                   'branch2_networks', 'branch2_match_dimension',
                   'decoder_networks', 'decoder_group_embedding',
                   'memory_guided', 'embedding_init', 'aggregation', 'alpha',
                   'threshold_setting', 'anomaly_ratio', 'temperature',
                   'num_epochs', 'batch_size', 'peak_lr', 'end_lr', 'weight_decay', 'patience']

    df_title = timestamp + config_list + metric_list

    benchmark_results.append(datetime.now().strftime("%Y%m%d-%H%M%S"))
    for k in config_list:
        if k in params.keys():
            benchmark_results.extend([params[k]])
        else:
            benchmark_results.extend(['-'])
    benchmark_results.extend([eval_results[k] for k in metric_list if k in eval_results.keys()])
    # 该函数将参数和评估结果写入 CSV 文件以便后续分析。
    # benchmark_results 首先添加当前时间戳 ￼。
    # 遍历预定义的 config_list，将参数字典中对应值依次添加到结果列表，若某参数缺失则填 '-' ￼。
    # 然后按照 metric_list 添加评估结果 ￼。

    os.makedirs('./results', exist_ok=True)
    result_file_name = f'./results/MtsLINE_benchmark_result.csv'
    if pd is not None and os.path.exists(result_file_name):
        df = pd.read_csv(result_file_name, encoding='utf8')
        df.loc[len(df)] = benchmark_results
        df.to_csv(result_file_name, index=False)
    elif pd is not None:
        pd.DataFrame([benchmark_results], columns=df_title).to_csv(result_file_name, index=False, encoding='utf8')
    else:
        file_exists = os.path.exists(result_file_name)
        with open(result_file_name, 'a', newline='', encoding='utf8') as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(df_title)
            writer.writerow(benchmark_results)
        # 确保结果文件夹存在 ￼。
        # 如果文件已存在则读取并追加新的行；否则创建新文件。这里文件名中用了 MtsLINE，可能是项目早期的命名或共享脚本，实际写入的是当前模型 MtsCID 的结果。

# multiscale_kernel_size 对应多尺度卷积 ￼；memory_guided 对应固定正弦原型矩阵 ￼；alpha 控制熵正则化，防止注意力过于分散 ￼；
# harmonic_loss_compute 实现时频域损失的调和组合 ￼
