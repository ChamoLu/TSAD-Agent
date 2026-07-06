import numpy as np

def point_adjustment(y_true, y_score):
    # 接收真实标签 y_true 和模型输出的异常得分 y_score（一个实数数组，数值越大表示越可能异常）。在 TSAD 领域，调整后的 F1 指标能够更公平地评价模型对异常区段的检测能力
    # 该函数的作用是对每个真实异常区段，将该段内所有得分替换为该段内的最大得分。这样做是为了在计算 PR 或 F1 时保证只要模型在异常段内有一个点得分高，整个段都被视为命中。
    """
    adjust the score for segment detection. i.e., for each ground-truth anomaly segment,
    use the maximum score as the score of all points in that segment. This corresponds to point-adjust f1-score.
    *This function is copied/modified from the source code in [Zhihan Li et al. KDD21]* 

    Args:
    
        y_true (np.array, required): 
            Data label, 0 indicates normal timestamp, and 1 is anomaly.
            
        y_score (np.array, required): 
            Predicted anomaly scores, higher score indicates higher likelihoods to be anomaly.
    Returns:
    
        np.array: 
            Adjusted anomaly scores.
    """
    score = y_score.copy()
    assert len(score) == len(y_true)
    # 使用 .copy() 避免直接修改原始输入
    # 断言检查 score 与 y_true 长度一致
 
    splits = np.where(y_true[1:] != y_true[:-1])[0] + 1 # 似乎没有真实实现效果
    # splits = np.where(np.diff(y_true) != 0)[0] + 1
    # 通过比较相邻元素是否不同来找到标签变化的下标；例如 [0,0,1,1,1,0] 得到 splits=[2,5]。加上 1 表示分割点的索引，方便后续切片。
    # np.diff(y_true): 计算相邻元素之间的差异，返回一个数组，表示每对相邻元素的差值。比如 y_true = [0, 0, 1, 1, 1, 0]，np.diff(y_true) 返回 [0, 1, 0, 0, -1]。表示 0 到 1 的变化、1 到 0 的变化。

    is_anomaly = y_true[0] == 1
    pos = 0
    # is_anomaly 指示当前区段是否为异常段，初值取决于标签序列第一个元素
    # pos 为当前区段的起始位置

    for sp in splits:
        if is_anomaly:
            score[pos:sp] = np.max(score[pos:sp])
        is_anomaly = not is_anomaly
        pos = sp
        # 对于每个分割点 sp：
        # 如果当前区段为异常段 (is_anomaly=True)，则将该区段 [pos:sp] 内的所有得分替换为该区段最大值。这实现了区段内取最大
        # 然后翻转 is_anomaly，代表下一个区段类型与当前相反（因为真实标签是 0/1 相互切换）,标签表示正常（0）和异常（1）交替出现,
        # 每当标签切换时，意味着正常段和异常段的边界发生了变化。因此，在遍历每个区段时，我们需要切换当前区段的类型,更新 pos 为新的区段起点

    sp = len(y_true)
    if is_anomaly:
        score[pos:sp] = np.max(score[pos:sp])
        # 分割循环结束后，pos 指向最后一个区段的起点，sp 设为数组长度
        # 如果最后一个区段是异常段，则同样将其所有得分替换为最大值

    return score
    # 返回经过调整的 score，可用于计算 ROC 曲线、PR 曲线或 threshold-based F1
    # 这种调整策略能保证在真实异常段内只要有一个高得分点，整段的评分都高，减少了“击中但低估”的惩罚
