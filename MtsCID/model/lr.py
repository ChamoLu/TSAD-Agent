# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from torch.optim.lr_scheduler import _LRScheduler
# 导入 PyTorch 中的 _LRScheduler 类，所有自定义学习率调度器都需继承此基类。它提供了对优化器参数组的管理和 step() 调度框架。

class PolynomialDecayLR(_LRScheduler): # 定义一个名为 PolynomialDecayLR 的类，继承自 _LRScheduler, 用于实现多项式衰减学习率。
    def __init__(
        self,
        optimizer,
        warmup_updates,
        tot_updates,
        lr,
        end_lr,
        power,
        last_epoch=-1,
        verbose=False,
    ):
        self.warmup_updates = warmup_updates
        self.tot_updates = tot_updates
        self.lr = lr
        self.end_lr = end_lr
        self.power = power
        super(PolynomialDecayLR, self).__init__(optimizer, last_epoch)
        # 可选：如果你希望保留 verbose 变量供后续逻辑使用，可以手动赋值（不加这行通常也不影响运行）
        self.verbose = verbose
        # 构造函数定义了调度器的初始化参数：
        # • optimizer：需要调整学习率的优化器。
        # • warmup_updates：warm‑up 阶段的更新步数，用于线性增大学习率。
        # • tot_updates：总的训练更新步数，决定衰减结束的时间。
        # • lr：峰值学习率，即 warm‑up 阶段结束后的初始值。
        # • end_lr：训练结束时的学习率下限。
        # • power：控制多项式衰减的幂指数，power=1.0 表示线性衰减，power=2.0 表示二次衰减。
        # • last_epoch：从哪一个 epoch 开始计数，默认 −1 代表从 0 开始。
        # • verbose：是否打印调度信息。

    def get_lr(self):
        if self._step_count <= self.warmup_updates:
            self.warmup_factor = self._step_count / float(self.warmup_updates)
            lr = self.warmup_factor * self.lr
            # 当 _step_count（当前更新步数）小于或等于 warmup_updates 时，表示训练处于 warm‑up 阶段。
            # 此时学习率按照线性比例 warmup_factor 从 0 增加至峰值 self.lr。warmup_factor = 当前步数 ÷ warm‑up 总步数，因此第一个步的学习率为 peak_lr / warmup_updates。
            # 若 warmup_updates 为 0，则会产生除 0 警告，这也是默认 warm‑up 设为 0 时需要注意的潜在问题。
        elif self._step_count >= self.tot_updates:
            lr = self.end_lr
            # 若当前步数大于等于总更新步数 tot_updates，说明训练已到达或超过预期迭代次数，此时将学习率固定为最小终止值 end_lr。
            # 这避免后续步骤学习率进一步下降导致训练停滞。
        else:
            warmup = self.warmup_updates
            lr_range = self.lr - self.end_lr
            pct_remaining = 1 - (self._step_count - warmup) / (
                self.tot_updates - warmup
            )
            lr = lr_range * pct_remaining ** (self.power) + self.end_lr
            # 进入该分支说明当前处于正常衰减阶段：
            # 1. warmup 保存 warm‑up 步数。
            # 2. lr_range 计算峰值与终止学习率的差。
            # 3. pct_remaining 计算剩余训练进度的比例：减去已完成的训练比例。具体公式为：1 - (当前步 – warmup) / (总步 – warmup)。此值从 1 随进度下降至 0。 
            # 4. lr 根据多项式衰减公式计算当前学习率：lr = lr_range * (pct_remaining)**power + end_lr。当 power=1.0 时为线性衰减，当 power>1 时曲线更陡峭。
            # 该公式确保在 warm‑up 结束时 pct_remaining≈1，学习率接近峰值；随着训练进度逼近 tot_updates，pct_remaining→0，学习率趋近 end_lr。

        return [lr for group in self.optimizer.param_groups]
        # 返回一个列表，其中每个元素都是上一步计算的学习率。optimizer.param_groups 允许不同参数组使用不同学习率，这里将所有组设置为同样的 lr 值。

    def _get_closed_form_lr(self):
        assert False
        # 重写了 _get_closed_form_lr() 方法并直接抛出断言失败，意味着调度器不支持闭式解析形式。
        # PyTorch 需要实现该方法才能在 scheduler.get_last_lr() 等情况下计算历史学习率；但此处作者选择不实现，表明仅在调用 scheduler.step() 时动态计算学习率。
