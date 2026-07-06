# Some code based on https://github.com/thuml/Anomaly-Transformer

import os
import time
import math
import logging
import builtins
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from utils.utils import *
from model.loss_functions import sce_loss
from model.lr import PolynomialDecayLR
from model.Transformer import TransformerVar
from model.loss_functions import *
from data_factory.data_loader import get_loader_segment
try:
    from sklearn.metrics import (precision_score,
                                 recall_score,
                                 f1_score,
                                 auc,
                                 roc_auc_score,
                                 average_precision_score,
                                 precision_recall_curve,
                                 )
except Exception:
    from metrics.basic_metrics import (precision_score,
                                       recall_score,
                                       f1_score,
                                       auc,
                                       roc_auc_score,
                                       average_precision_score,
                                       precision_recall_curve,
                                       )

from metrics.metrics import *
from metrics import point_adjustment
from metrics import ts_metrics_enhanced

os.environ["CUDA_VISIBLE_DEVICES"] = '0'

def adjust_learning_rate(optimizer, epoch, initial_lr, step_size=2, decay_factor=0.9):
    # 此函数在 train 中被注释掉，作者选择使用 PolynomialDecayLR 调度器。根据论文实现，训练时采用多项式衰减学习率
    lr_adjust = {epoch: initial_lr * (decay_factor ** ((epoch - 1) // step_size))}
    # 根据当前 epoch 计算应调整到的学习率 通过字典 lr_adjust 保存对应关系。
    if epoch in lr_adjust.keys():
        lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
            # 如果当前 epoch 在 lr_adjust 中，则遍历优化器的每个参数组，将其学习率更新为对应值，并打印更新提示
        print(f'Updating learning rate to {lr}')

class OneEarlyStopping:
    def __init__(self, patience=10, verbose=False, dataset_name='', delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.inf
        self.delta = delta
        self.dataset = dataset_name
        # 初始化计数器 counter、最佳得分 best_score、早停标志 early_stop、最小验证损失 val_loss_min...

    def __call__(self, val_loss, model, path):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
            self.counter = 0
            # 每完成一轮训练后调用，将当前验证损失转换为 score=-val_loss，与历史最佳比较。
            # 如果首次调用则保存检查点；如果当前分数比最佳差 delta 以上则递增 counter，若超过 patience 则置 early_stop=True；
            # 否则更新最佳分数并保存新检查点（调用 save_checkpoint）。

    def save_checkpoint(self, val_loss, model, path):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), os.path.join(path, str(self.dataset) + f'_checkpoint.pth'))
        self.val_loss_min = val_loss
        # save_checkpoint 将模型参数保存到指定路径 dataset_checkpoint.pth，同时更新 val_loss_min

class Solver(object):
    # Solver 类封装了模型初始化、训练、验证和测试等过程
    DEFAULTS = {}

    def __init__(self, config): # 参数 config 为配置字典

        self.scheduler = None
        self.model = None
        self.optimizer = None
        self.__dict__.update(Solver.DEFAULTS, **config)
        # 初始化空的 scheduler, model, optimizer；将 config 更新到实例属性中，使其可通过 self.parameter 访问配置项

        self.train_time_per_epoch = 0.0
        self.test_time_per_epoch = 0.0
        # 定义记录每轮训练和测试时间的变量 train_time_per_epoch, test_time_per_epoch
        
        self.train_loader, self.vali_loader = get_loader_segment(self.data_path,
                                                                 batch_size=self.batch_size,
                                                                 win_size=self.win_size,
                                                                 mode='train',
                                                                 dataset=self.dataset)
        self.test_loader = get_loader_segment(self.data_path,
                                              batch_size=self.batch_size,
                                              win_size=self.win_size,
                                              mode='test',
                                              dataset=self.dataset)
        # 调用 get_loader_segment 创建训练集 (train_loader) 和验证集 (vali_loader)。参数：self.data_path: 数据文件路径。
        # batch_size: 每批窗口个数。 win_size: 滑窗长度。 mode='train' 或 'test' 指定训练/测试。 dataset: 数据集名称。
        # 函数将多维时间序列划分为重叠的窗口（子序列），正是 MtsCID 处理的基本单元

        self.entropy_loss = EntropyLoss()
        self.criterion = nn.MSELoss(reduction='none')
        # 实例化 EntropyLoss()（来自 model.loss_functions）。论文中互变量分支采用熵损失作为辅助任务 该损失鼓励原型分布信息更加平滑
        # 设置重构损失 self.criterion ，MSE 在时间序列模型中常用于度量重建误差

        self.logger = logging.getLogger()
        self.logger.setLevel(logging.INFO)

        formatter = logging.Formatter('%(asctime)s - %(message)s')
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)

        # Check if the stream handler is already added
        if not any(isinstance(handler, logging.StreamHandler) for handler in self.logger.handlers):
            self.logger.addHandler(stream_handler)
            # Redirect print to logger
            self._redirect_print_to_logger()
        # 设置日志模块：获取默认 logger，设置等级为 INFO；定义格式；创建流处理器；
        # 若当前 logger 无该处理器则添加，并调用 _redirect_print_to_logger() 重定向内置 print。这样所有 print 调用都通过 logger 输出，便于记录训练日志

    def _redirect_print_to_logger(self):
        def print_to_logger(*args, **kwargs):
            message = " ".join(map(str, args))
            self.logger.info(message)
            # Replace the built-in print function with the custom one
            builtins.print = print_to_logger
            # 内部函数 print_to_logger 将所有参数拼接为字符串并通过 self.logger.info 输出；然后将内置 print 指向该函数。此举确保项目所有 print() 调用自动记录到日志。

    def model_init(self, config):
        self.model = TransformerVar(config)
        # self.model 创建模型实例。根据论文，TransformerVar 通过双分支捕获时间依赖和互变量依赖，并返回 out（重构序列）、attn（注意力图）、queries（从互变量分支得到的查询向量）和 mem（固定原型）
        # self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.peak_lr)
        
        self.optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, self.model.parameters()),
                                           lr=self.peak_lr, weight_decay=self.weight_decay)

        self.scheduler = PolynomialDecayLR(self.optimizer,
                                           warmup_updates=self.warmup_epoch * self.batch_size,
                                           tot_updates=self.num_epochs * self.batch_size,
                                           lr=self.peak_lr,
                                           end_lr=self.end_lr,
                                           power=1.0)
        # 使用 torch.optim.AdamW 优化器，并过滤掉不需要梯度的参数；学习率为配置中的 peak_lr，权重衰减 weight_decay 用于正则化
        # 创建 PolynomialDecayLR 调度器, 论文的实验部分指出使用多项式衰减学习率

        if torch.cuda.is_available() and torch.device(self.device).type == 'cuda':
            self.model = torch.nn.DataParallel(self.model, device_ids=[0], output_device=0).to(self.device)
            # 如果支持 CUDA，则将模型放置在指定 GPU 上，并使用 torch.nn.DataParallel 进行多 GPU 训练；device_ids=[0] 指只使用第 0 号 GPU
        else:
            self.model = self.model.to(self.device)

    def vali(self, vali_loader):
        # 定义 vali()，用于在验证集上计算平均损失
        self.model.eval()
        # 将模型置于评估模式 eval()，停用 dropout/BN

        valid_loss_list = []
        valid_re_loss_list = []
        valid_intra_loss_list = []
        # 初始化列表用于保存各批次的总损失、重构损失和熵损失

        for i, (input_data, _) in enumerate(vali_loader):
            input_data = input_data.float().to(self.device)
            output_dict = self.model(input_data)
            # 获取 input_data 和标签（未使用）,将数据转为浮点并移动到设备，输入模型得到 output_dict

            output = output_dict['out']
            attn = output_dict['attn']

            rec_loss = self.criterion(output, input_data).mean()
            attn_loss = torch.zeros_like(rec_loss) if attn is None else self.entropy_loss(attn) * self.alpha
            # 计算重构损失 rec_loss = MSE(output, input) 后取平均。这里按元素无 reduction，后面 .mean() 得单标量
            # 若 attn 为空（可能某些模型不返回注意力），则熵损失为零；否则通过 self.entropy_loss(attn) * self.alpha 计算原型分支的熵损失乘权系数 alpha（论文中即参数 λ，用于平衡两种任务

            loss = rec_loss + attn_loss

            valid_re_loss_list.append(rec_loss.detach().cpu().numpy())
            valid_intra_loss_list.append(attn_loss.detach().cpu().numpy())
            valid_loss_list.append(loss.detach().cpu().numpy())
            # 将三种损失转换成 NumPy 并存储。由于 reduction='none'，每个元素的损失被平均后存储

        return np.average(valid_loss_list), np.average(valid_re_loss_list), np.average(valid_intra_loss_list)
        # 返回三个损失的平均值：valid_loss_list（总损失）、valid_re_loss_list（重构损失）和 valid_intra_loss_list（熵损失）

    def train(self): # 函数封装训练逻辑：建立模型保存目录、循环训练每个 epoch，并在验证集监控性能

        # print("======================TRAIN MODE======================")
        if not os.path.exists(self.model_save_path):
            os.makedirs(self.model_save_path)
        early_stopping = OneEarlyStopping(patience=self.patience, verbose=True, dataset_name=self.dataset)
        train_steps = len(self.train_loader)
        # 如果 model_save_path 不存在则创建，用于保存最优权重。实例化 OneEarlyStopping，使用配置中的 patience 和 dataset 名称。计算 train_steps 为训练集批次数

        training_start_time = time_now = time.time()
        # 记录训练开始时间 training_start_time 与 time_now

        for epoch in tqdm(range(self.num_epochs)): # 使用 tqdm 显示进度条，循环 num_epochs 个 epoch
            iter_count = 0
            loss_list = []
            rec_loss_list = []
            intra_loss_list = []
            # 重置计数器和损失列表 loss_list, rec_loss_list, intra_loss_list

            # adjust_learning_rate(self.optimizer, epoch, self.peak_lr)
            epoch_time = time.time()

            self.model.train()
            # 将模型切换到训练模式 train()
            for i, (input_data, labels) in enumerate(self.train_loader):
                self.optimizer.zero_grad()
                iter_count += 1
                input_data = input_data.float().to(self.device)
                output_dict = self.model(input_data)
                # 遍历训练集的每个批次：
                # 调用 optimizer.zero_grad() 清除之前的梯度
                # 增加迭代计数器 iter_count
                # 将批次输入数据转为浮点并放到设备；通过模型得到 output_dict

                output = output_dict['out']
                attn = output_dict['attn']
                # 取出重构输出 output 和注意力 attn

                rec_loss = self.criterion(output, input_data).mean()
                attn_loss = torch.zeros_like(rec_loss) if attn is None else self.entropy_loss(attn) * self.alpha
                loss = rec_loss + attn_loss
                # 计算重构损失和熵损失（与 vali 同）
                loss_list.append(loss.detach().cpu().numpy())
                rec_loss_list.append(rec_loss.detach().cpu().numpy())
                intra_loss_list.append(attn_loss.detach().cpu().numpy())

                if (i + 1) % 100 == 0:
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.num_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()
                    # 每处理 100 个批次，计算训练速度（speed）和剩余时间，打印提示，并重置 iter_count 与 time_now

                loss.backward()
                self.optimizer.step()
                # 反向传播 loss.backward() 并通过 optimizer.step() 更新参数
            print("\nEpoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            # 每个 epoch 结束打印耗时

            train_loss = np.average(loss_list)
            train_rec_loss = np.average(rec_loss_list)
            train_intra_loss = np.average(intra_loss_list)
            valid_loss, valid_re_loss, valid_intra_loss = self.vali(self.vali_loader)
            print(
                f"Epoch: {epoch + 1}, Steps: {train_steps} | Train Loss: {train_loss:.7f} | Vali Loss: {valid_loss:.7f}")
            print(
                f"Epoch: {epoch + 1}, Steps: {train_steps} | Train reconstruction Loss: {train_rec_loss:.7f} | Entropy Loss : {train_intra_loss:.7f}")
            print(
                f"Epoch: {epoch + 1}, Steps: {train_steps} | Valid reconstruction Loss: {valid_re_loss:.7f} | Entropy Loss : {valid_intra_loss:.7f}")
            # 打印当前 epoch 的训练和验证损失
            early_stopping(valid_loss, self.model, self.model_save_path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

        self.train_time_per_epoch = round((time.time() - training_start_time) / (epoch + 1), 3)
        # 更新 train_time_per_epoch 为平均每个 epoch 的训练时间
        return

    def _load_model_checkpoint(self):
        model_name = os.path.join(str(self.model_save_path), str(self.dataset) + '_checkpoint.pth')
        checkpoint = torch.load(model_name, map_location=self.device)
        model_state = self.model.state_dict()

        checkpoint_has_module = any(key.startswith('module.') for key in checkpoint.keys())
        model_has_module = any(key.startswith('module.') for key in model_state.keys())

        if checkpoint_has_module and not model_has_module:
            checkpoint = {key.replace('module.', '', 1): value for key, value in checkpoint.items()}
        elif model_has_module and not checkpoint_has_module:
            checkpoint = {'module.' + key: value for key, value in checkpoint.items()}

        self.model.load_state_dict(checkpoint)

    def _collect_loader_energy(self, data_loader, criterion, gathering_loss, collect_details=False):
        energy_list = []
        label_list = []
        variable_error_list = []

        with torch.no_grad():
            for input_data, labels in data_loader:
                input_data = input_data.float().to(self.device)
                output_dict = self.model(input_data, mode='test')

                output = output_dict['out']
                queries = output_dict['queries']
                mem_items = output_dict['mem']

                rec_loss = criterion(input_data, output)
                latent_score = torch.softmax(gathering_loss(queries, mem_items) / self.temperature, dim=-1)
                loss = harmonic_loss_compute(rec_loss, latent_score, self.aggregation)

                energy_list.append(loss.detach().cpu().numpy())
                if collect_details:
                    label_list.append(labels.detach().cpu().numpy())
                    variable_error_list.append(rec_loss.detach().cpu().numpy())

        energy = np.concatenate(energy_list, axis=0).reshape(-1)
        if not collect_details:
            return energy, None, None

        labels = np.concatenate(label_list, axis=0).reshape(-1)
        variable_errors = np.concatenate(variable_error_list, axis=0).reshape(-1, self.input_c)
        return energy, labels, variable_errors

    def detect(self, params=None, include_details=True):
        """
        Run MtsCID detection and return model evidence for downstream services.

        The original test() method returns aggregate metrics for paper-style
        evaluation. This method keeps the same threshold/metric protocol, while
        also exposing per-timestamp scores, predicted labels and variable-level
        reconstruction errors for the Agent layer.
        """
        self._load_model_checkpoint()
        self.model.eval()

        print("============================DETECT MODE============================")
        print(f"Dataset: {self.dataset}")

        criterion = nn.MSELoss(reduction='none')
        gathering_loss = GatheringLoss(reduction='none', memto_framework=True)

        if self.threshold_setting == 'preset':
            train_energy, _, _ = self._collect_loader_energy(self.train_loader, criterion, gathering_loss)
            valid_energy, _, _ = self._collect_loader_energy(self.vali_loader, criterion, gathering_loss)
            combined_energy = np.concatenate([train_energy, valid_energy], axis=0)
            thresh = np.percentile(combined_energy, 100 - self.anomaly_ratio)
        else:
            thresh = None

        start_time = time.time()
        test_energy, test_labels, variable_errors = self._collect_loader_energy(
            self.test_loader,
            criterion,
            gathering_loss,
            collect_details=True
        )
        self.test_time_per_epoch = round(time.time() - start_time, 3)

        if self.threshold_setting == 'optimal':
            thresh = np.percentile(test_energy, 100 - self.anomaly_ratio)

        print("Threshold :", thresh)
        pred = (test_energy > thresh).astype(int)
        test_labels = np.array(test_labels).astype(int)

        if self.threshold_setting == 'optimal':
            results = ts_metrics_enhanced(test_labels, point_adjustment(test_labels, test_energy), pred)
        else:
            gt = test_labels.astype(int)
            print(f"pred: {pred.shape}, gt: {gt.shape}")
            events = get_events(gt)
            _, _, _, _, _, _, threshold_setting_results = get_point_adjust_scores(gt, pred, test_energy, events)
            results = ts_metrics_enhanced(test_labels, point_adjustment(test_labels, test_energy), pred)
            results['pc_adjust'] = threshold_setting_results['pc_adjust']
            results['rc_adjust'] = threshold_setting_results['rc_adjust']
            results['f1_adjust'] = threshold_setting_results['f1_adjust']

        precision_adjust = results['pc_adjust']
        recall_adjust = results['rc_adjust']
        f_score_adjust = results['f1_adjust']
        results['thresh'] = float(thresh)
        results['trt'] = self.train_time_per_epoch
        results['tst'] = self.test_time_per_epoch

        print('=' * 63)
        print(f"Dataset: {self.dataset} | Precision_adjusted: {precision_adjust:.4f} | Recall_adjusted: {recall_adjust:.4f} | f1_score_adjusted: {f_score_adjust:.4f} ")

        detection_result = {
            'dataset': self.dataset,
            'threshold': float(thresh),
            'metrics': results,
            'scores': test_energy.astype(float),
            'pred_labels': pred.astype(int),
            'true_labels': test_labels.astype(int),
        }
        if include_details:
            detection_result['variable_errors'] = variable_errors.astype(float)
        return detection_result

    def test(self, params): # test(self, params) 用于加载最佳模型并根据重构误差和原型匹配误差计算异常分数，再据此输出指标

        model_name = os.path.join(str(self.model_save_path), str(self.dataset) + '_checkpoint.pth')
        self._load_model_checkpoint()
        self.model.eval()
        # 构建模型文件名，如 dataset_checkpoint.pth，并使用 torch.load 加载权重到当前模型；随后设置评估模式 eval()

        print("============================TEST MODE============================")

        criterion = nn.MSELoss(reduction='none')
        gathering_loss = GatheringLoss(reduction='none', memto_framework=True)
        # 定义 criterion 用于计算每个时间步的重构误差
        # gathering_loss 根据论文，该损失通过计算查询向量与固定原型的余弦距离或相似度来衡量互变量关系偏差

        print(f"Dataset: {self.dataset}")

        if self.threshold_setting == 'preset': # 预设阈值 如果 threshold_setting 被设置为 'preset'，则通过训练集和验证集的异常能量确定阈值。具体步骤：
            train_attens_energy = []
            for i, (input_data, labels) in enumerate(self.train_loader):
                input_data = input_data.float().to(self.device)
                output_dict = self.model(input_data, mode='test')

                output = output_dict['out']
                queries = output_dict['queries']
                mem_items = output_dict['mem']

                rec_loss = criterion(input_data, output)
                latent_score = torch.softmax(gathering_loss(queries, mem_items) / self.temperature, dim=-1)
                loss = harmonic_loss_compute(rec_loss, latent_score, self.aggregation)

                cri = loss.detach().cpu().numpy()
                train_attens_energy.append(cri)
                # 将 input_data 转为浮点并送入模型（测试模式 mode='test'）
                # 从模型输出中获取 output（重构序列）、queries（来自互变量分支的查询向量）和 mem_items（固定原型）
                # 计算重构误差 rec_loss
                # 通过 gathering_loss( ) 计算查询向量到原型的距离或相似度并除以温度系数 self.temperature，然后取 softmax 得到权重 latent_score（代表查询向量在原型集合上的匹配分布）
                # 调用 harmonic_loss_compute( ) 得到最终的异常能量 loss
                # 将 cri（当前批次的能量）添加到 train_attens_energy。

            train_attens_energy = np.concatenate(train_attens_energy, axis=0).reshape(-1)
            train_energy = np.array(train_attens_energy)
            # 参数是一个元组或列表，里面放多个形状相同的数组；axis=0 表示沿第 0 维拼接；axis=1 表示沿第 1 维拼接。把数组“重塑形状（reshape）”，-1 表示“自动推算维度”，即“拉平成一维” 
            # 将训练集能量拼接并展平得到一维数组 train_energy

            valid_attens_energy = []
            for i, (input_data, labels) in enumerate(self.vali_loader):
                input_data = input_data.float().to(self.device)
                output_dict = self.model(input_data, mode='test')

                output = output_dict['out']
                queries = output_dict['queries']
                mem_items = output_dict['mem']

                rec_loss = criterion(input_data, output)
                latent_score = torch.softmax(gathering_loss(queries, mem_items) / self.temperature, dim=-1)
                loss = harmonic_loss_compute(rec_loss, latent_score, self.aggregation)

                cri = loss.detach().cpu().numpy()
                valid_attens_energy.append(cri)

            valid_attens_energy = np.concatenate(valid_attens_energy, axis=0).reshape(-1)
            valid_energy = np.array(valid_attens_energy)
            # 对验证集执行类似操作，得到 valid_energy

            combined_energy = np.concatenate([train_energy, valid_energy], axis=0) # 拼接训练和验证能量为 combined_energy
            thresh = np.percentile(combined_energy, 100 - self.anomaly_ratio)
            print("Threshold :", thresh)
            # 根据配置中 anomaly_ratio（异常比例）计算阈值 thresh = np.percentile(combined_energy, 100 - anomaly_ratio)，即将能量排序后取百分位作为阈值。

        test_window_labels = []
        test_window_energy = []
        test_labels = []
        test_attens_energy = []
        start_time = time.time()
        # 初始化用于存储窗口级标签 (test_window_labels)、窗口级能量 (test_window_energy)、逐点标签 (test_labels) 和逐点能量 (test_attens_energy) 的列表

        for i, (input_data, labels) in enumerate(self.test_loader): # 遍历测试集的每个窗口
            input_data = input_data.float().to(self.device)
            output_dict = self.model(input_data, mode='test')

            output = output_dict['out']
            queries = output_dict['queries']
            mem_items = output_dict['mem']

            rec_loss = criterion(input_data, output)
            latent_score = torch.softmax(gathering_loss(queries, mem_items) / self.temperature, dim=-1)
            loss = harmonic_loss_compute(rec_loss, latent_score, self.aggregation)

            cri = loss.detach().cpu().numpy()
            test_attens_energy.append(cri)
            test_labels.append(labels)

            test_window_energy.extend(cri.mean(axis=-1))
            test_window_labels.extend((labels.sum(axis=-1) > 1).numpy().astype(int))
            # test_window_energy 保存每个窗口内能量的平均值，test_window_labels 保存窗口是否包含超过一个异常点（labels.sum(axis=-1) > 1）的标签，用于后续的窗口级评估或点调整

        self.test_time_per_epoch = round(time.time() - start_time, 3)
        # 计算并记录测试阶段耗时 test_time_per_epoch

        test_attens_energy = np.concatenate(test_attens_energy, axis=0).reshape(-1)
        test_labels = np.concatenate(test_labels, axis=0).reshape(-1)
        test_energy = np.array(test_attens_energy)
        test_labels = np.array(test_labels)
        # 将逐窗口能量拼接为一维数组 test_attens_energy，将逐点标签拼接为 test_labels；然后复制到 test_energy 和 test_labels（numpy 数组形式）

        if self.threshold_setting == 'optimal': # 基于 threshold_setting 计算预测结果
            anomaly_ratio = self.anomaly_ratio
            thresh = np.percentile(test_energy, 100 - anomaly_ratio)
            print("Threshold :", thresh)
            pred = (test_energy > thresh).astype(int)
            results = ts_metrics_enhanced(test_labels, point_adjustment(test_labels, test_energy), pred)
            # 如果 threshold_setting == 'optimal'，直接在测试能量上根据异常比例计算阈值，再由 ts_metrics_enhanced 与 point_adjustment 计算结果

        else: # 否则（即 'preset' 情况）
            results = {k: 0.0 for k in metric_list}
            results['thresh'] = 0.0
            # 初始化 results 字典（键来自 metric_list）并将阈值占位

            pred = (test_energy > thresh).astype(int) # 根据预设阈值将 test_energy 二值化得到预测标签 pred
            gt = test_labels.astype(int) # gt 得到真实标签。
            print(f"pred: {pred.shape}, gt: {gt.shape}")
            events = get_events(gt)
            # 打印预测与真实标签形状，并调用 get_events(gt) 将连续的异常点合并成事件区间

            _, _, _, _, _, _, threshold_setting_results = get_point_adjust_scores(gt, pred, test_energy, events)
            # 调用 get_point_adjust_scores() 得到未调整指标及调整后指标（如 pc_adjust）

            results = ts_metrics_enhanced(test_labels, point_adjustment(test_labels, test_energy), pred)
            # 再次调用 ts_metrics_enhanced 计算更全面的指标

            results['pc_adjust'] = threshold_setting_results['pc_adjust']
            results['rc_adjust'] = threshold_setting_results['rc_adjust']
            results['f1_adjust'] = threshold_setting_results['f1_adjust']
            # 从 threshold_setting_results 中提取调整后的精确率、召回率和 F1，并赋值给 results

        precision_adjust, recall_adjust, f_score_adjust = results['pc_adjust'], results['rc_adjust'], results['f1_adjust']
        results['thresh'] = thresh
        results['trt'] = self.train_time_per_epoch
        results['tst'] = self.test_time_per_epoch
        print('=' * 63)
        print(f"Dataset: {self.dataset} | Precision_adjusted: {precision_adjust:.4f} | Recall_adjusted: {recall_adjust:.4f} | f1_score_adjusted: {f_score_adjust:.4f} ")
        # 提取 precision_adjust, recall_adjust, f_score_adjust；更新结果字典中阈值 (thresh)、训练时间 (trt) 和测试时间 (tst)，然后打印最终评估结果

        return results

def get_point_adjust_scores(y_test, pred_labels, pred_scores, true_events):
    # y_test：一维布尔数组，真实标签（0 代表正常，1 代表异常）
    # pred_labels：一维布尔数组，模型的二值化预测结果
    # pred_scores：一维浮点数组，模型输出的连续得分，用于绘制 PR 和 ROC 曲线
    # true_events：由 get_events 函数生成的字典，键为事件编号，值为 (start, end) 元组，表示每个真实异常事件的起止索引
    # 这个函数的目的是计算多种性能指标 论文指出，利用点调整指标可以更好地评估模型在检测连续异常区段方面的表现
    # get_point_adjust_scores 依据论文的点调整原则，对预测标签与真实事件进行统计。其核心思想是：

    # 按照事件区段来统计真阳性和假阴性，避免对连续异常片段的重复计数
    # 使用普通二分类指标和基于分数的曲线指标来全面评估模型的检测性能

    results = {
        "pc": 0.0,
        "rc": 0.0,
        "f1": 0.0,
        "acc_adjust": 0.0,
        "pc_adjust": 0.0,
        "rc_adjust": 0.0,
        "f1_adjust": 0.0,
        "mcc_adjust": 0.0,
        "prc": 0.0,
        "roc": 0.0,
        "apc": 0.0,
    }
    tp = 0
    fn = 0
    # 初始化 results 字典：创建一个包含所有指标初始化值的字典，便于后续填充计算结果
    # 初始化真阳性 tp 和假阴性 fn：这两个变量专用于“点调整”统计，它们分别统计正确预测的异常点数和漏检的异常点数

    for true_event in true_events.keys():       # 遍历每个真实事件：
        true_start, true_end = true_events[true_event]  # 解包该事件的起止索引
        if pred_labels[true_start:true_end].sum() > 0:
            tp += (true_end - true_start)
            # 检查在这一真实异常段内部是否至少有一个时间点被模型预测为异常。根据论文的点调整规则，只要区段中有任意一个预测点标记为异常，
            # 则整个段都算作正确识别。如果至少检测到一个点，则将该段的长度 true_end - true_start 累加到 tp，表示这一整段都被正确预测；
        else:
            fn += (true_end - true_start)
            # 否则将长度累加到 fn，表示这一段完全未被检测到。
            
    fp = np.sum(pred_labels) - np.sum(pred_labels * y_test)
    # np.sum(pred_labels) 统计模型标记为异常的总点数；np.sum(pred_labels * y_test) 是与真实异常重合的部分。两者相减得到误报的异常点数量

    pc, rc, fscore = get_prec_rec_fscore(tp, fp, fn)
    # 根据点调整的 tp、fp、fn 计算精准率（pc）、召回率（rc）和 F1 （fscore），该函数后文将详细解释。点调整的召回率计算公式同样遵循“一个区段全部正确”规则

    tn = len(pred_labels) - (tp + fp + fn)
    # len(pred_labels) 为总样本数，将所有已统计出的 tp + fp + fn 从中减去即可得到未预测为异常且真实为正常的时间点数

    avg_precision = average_precision_score(y_test, pred_scores)
    auc_roc = roc_auc_score(y_test, pred_scores)
    precision, recall, _ = precision_recall_curve(y_test, pred_scores)
    # average_precision_score() 即 APC（Average Precision）
    # roc_auc_score() 是 ROC 曲线面积；
    # precision_recall_curve() 获得一系列阈值下的精准率和召回率曲线，用于计算 PR 曲线面积。

    results['pc'] = round(precision_score(y_test, pred_labels, average='binary'), 4)
    results['rc'] = round(recall_score(y_test, pred_labels, average='binary'), 4)
    results['f1'] = round(f1_score(y_test, pred_labels, average='binary'), 4)
    # 利用 sklearn 内置函数，分别计算基于每个时间点的二分类 Precision、Recall 和 F1，并四舍五入到小数点后四位填入 results['pc']、results['rc'] 和 results['f1']。

    results['f1_adjust'] = round(fscore, 4)
    results['pc_adjust'] = round(pc, 4)
    results['rc_adjust'] = round(rc, 4)
    # results['mcc_adjust'] = round(matthews_correlation_coefficient(tp, tn, fp, fn), 4)
    results['acc_adjust'] = round((tp + tn) / len(y_test), 4)
    # 将pc、rc、fscore 四舍五入后分别赋值到 results['pc_adjust']、results['rc_adjust'] 和 results['f1_adjust']。
    # 随后 results['acc_adjust'] 计算为 (tp + tn) / len(y_test)，表示点调整后的准确率

    results["prc"] = round(auc(recall, precision), 4)
    results["roc"] = round(auc_roc, 4)
    results["apc"] = round(avg_precision, 4)
    # 通过 auc() 计算 PR 曲线下的面积并存入 results["prc"]。此外 results["roc"] 存储 ROC‑AUC，results["apc"] 存储平均精准率。论文中提到，通过综合使用这些指标可以减少单一阈值对评估的偏差

    return fp, fn, tp, pc, rc, fscore, results

def matthews_correlation_coefficient(TP, TN, FP, FN):
    numerator = TP * TN - FP * FN
    denominator = np.sqrt(TP + FP) * np.sqrt(TP + FN) * np.sqrt(TN + FP) * np.sqrt((TN + FN))

    # Avoid division by zero
    if denominator < np.finfo(float).eps:
        return 0.0

    mcc = numerator / denominator
    return mcc
    # 马修斯相关系数（MCC）是一种衡量二分类性能的指标，特别适合处理类别不平衡问题。公式中：
    # TP（真阳性）、TN（真阴性）、FP（假阳性）、FN（假阴性）与传统定义一致
    # 分子 TP * TN - FP * FN 反映正负样本预测的关联
    # 分母为各类总数的几何平均，避免了简单准确率在样本不均衡情况下的偏差；
    # 为避免分母为零导致溢出，代码检查 denominator 是否接近机器精度 np.finfo(float).eps 并返回 0
    # 尽管代码中计算了 MCC，但在 get_point_adjust_scores 中该值被注释掉，可能是作者在实验中未使用此指标。

def get_f_score(pc, rc):
    if pc == 0 and rc == 0:
        f_score = 0
    else:
        f_score = 2 * (pc * rc) / (pc + rc)
    return f_score
    # 该函数根据精准率 pc 和召回率 rc 计算 F1 分数。如果两者都为 0，则直接返回 0；否则按标准公式 F1 = 2 · P · R / (P + R) 计算

def get_prec_rec_fscore(tp, fp, fn):
    if tp == 0:
        precision = 0
        recall = 0
    else:
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
    fscore = get_f_score(precision, recall)
    return precision, recall, fscore
    # 该函数根据真阳性 tp、假阳性 fp 和假阴性 fn 计算精准率、召回率以及 F1 分数。具体步骤：
    # 当 tp==0 时，为避免除零，直接设定 precision=0、recall=0；
    # 否则按定义计算精准率 tp/(tp+fp) 与召回率 tp/(tp+fn)；
    # 调用上面的 get_f_score 计算 F1 分数；
    # 此函数与 get_point_adjust_scores 配合，用于点调整后的统计量计算。

def get_events(y_test, outlier=1, normal=0, breaks=[]):
    '''
    MtsCID 在评估阶段采用的点调整指标要求先将序列划分为连续异常区段
    get_events 正是实现这一划分的辅助函数。它的输出 true_events 被用于 get_point_adjust_scores 对每个区段统计是否检测到异常
    '''
    # 该函数用于从真实标签序列中提取连续的异常片段，生成 事件字典。论文指出，在评估时应该把连续异常区段看作一个整体
    # get_events 的作用正是识别这些区段，返回以 event_id : (start_index, end_index) 形式表示的字典

    events = dict()
    label_prev = normal
    event = 0  # corresponds to no event
    event_start = 0
    # 初始化：events 创建存储结果的字典；
    # label_prev 表示前一个时间点的标签，初始为正常；
    # event = 0 事件编号的起始值，0 表示当前没有进入异常区段；
    # event_start = 0 用于记录某个异常段开始的索引。

    for tim, label in enumerate(y_test):
        if label == outlier:
            if label_prev == normal:
                event += 1
                event_start = tim
            elif tim in breaks:
                # A break point was hit, end current event and start new one
                event_end = tim - 1
                events[event] = (event_start, event_end)
                event += 1
                event_start = tim
        else:
            # event_by_time_true[tim] = 0
            if label_prev == outlier:
                event_end = tim - 1
                events[event] = (event_start, event_end)
        label_prev = label
        # for tim, label in enumerate(y_test) 依次遍历序列的索引和标签。
        # 当遇到异常标签 label == outlier：
        # 如果前一个标签是正常 (label_prev == normal)，说明一个新的异常段开始，于是事件编号加 1，并记录这一段的起始索引 event_start = tim。
        # 如果不是第一次遇到异常但当前时间点在 breaks 列表中，则认为发生了人为的分段（例如跨天或不可连续的情形）。此时应结束前一个异常段：将 (event_start, tim-1) 存入字典，然后开始新的事件并更新 event_start。
        # 当遇到正常标签：
        # 如果前一个标签是异常 (label_prev == outlier)，说明刚结束一个异常段。将 (event_start, tim-1) 记录到事件字典中。
        # 每次迭代结束后更新 label_prev = label

    if label_prev == outlier:
        # event_end = tim - 1  # original code is wrong!
        event_end = tim
        events[event] = (event_start, event_end)
        # 如果最后一个标签仍然是异常 (label_prev == outlier)，则当前事件一直持续到序列末尾。event_end 应设置为最后一位的索引 tim。代码注释指出，此处修正了原始版本将终点减一的错误

    return events
    # 最终返回以事件编号为键、起止索引为值的字典

'''
MtsCID 是一种用于多元时间序列异常检测的半监督方法。论文指出，真实的异常通常以连续的时间段表现，而不是孤立的单个时间点。因此在评估时采用了所谓的“点调整”指标：
如果在真实异常区段内检测到任意一个时间点被模型标记为异常，则整个区段被视为正确识别. 这种度量将连续段看作一个事件，从而更公平地评估序列模型捕捉异常事件的能力
'''

'''
以上函数共同完成了 MtsCID 模型在测试阶段的指标计算。其中 get_events 按论文描述提取连续异常事件；
get_point_adjust_scores 按区段而非单点统计真阳性和假阴性，从而生成点调整后的精准率、召回率、F1 和准确率；其他辅助函数计算 F1 和马修斯相关系数等标准分类指标。
论文强调，通过结合点调整指标和曲线面积等多种评估方法，可以更全面地反映模型对异常区段的检测能力，并减少单一阈值的偏差
'''
