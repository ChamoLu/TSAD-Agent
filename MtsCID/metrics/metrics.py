# from metrics.f1_score_f1_pa import *
# from metrics.fc_score import *
# from metrics.precision_at_k import *
# from metrics.customizable_f1_score import *
# from metrics.AUC import *
# from metrics.Matthews_correlation_coefficient import *
from metrics.affiliation.generics import convert_vector_to_events
from metrics.affiliation.metrics import pr_from_events
from metrics.vus.metrics import get_range_vus_roc
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score
# convert_vector_to_events: 将二值序列（例如预测标签或真实标签）转换为一系列事件区段(event)，每个事件用起始和结束位置表示。这有助于计算区段级指标。
# pr_from_events: 根据预测事件和真实事件计算 Affiliation Precision 与 Recall。Affiliation 指标以区段为单位度量预测与真实的关联度，它更关注是否正确覆盖了真实异常区段。
# get_range_vus_roc: 计算 R‑AUC 和 VUS 等基于滑动窗口的曲面指标。VUS 指 Volume Under Surface，类似三维版本的 AUC，可以评价在不同决策阈值和窗口大小下检测器的性能。
import numpy as np

def combine_all_evaluation_scores(y_test, pred_labels):
    # combine_all_evaluation_scores 旨在综合计算多种评价指标。参数 y_test 应该是真实标签（0 表示正常，1 表示异常），pred_labels 是模型预测的二值标签。
    
    events_pred = convert_vector_to_events(y_test) 
    events_gt = convert_vector_to_events(pred_labels)
    # 这里从变量名来看存在混淆：events_pred 使用了 y_test，events_gt 使用了 pred_labels。照理应该用 pred_labels 构建预测事件，用 y_test 构建真实事件。
    # 从代码顺序来看似乎把真实标签和预测标签的位置交换了。这可能是代码作者的笔误，也可能是为了将“异常段”(ground truth) 与 “预测段” 调换作用。
    # convert_vector_to_events 将一个形如 [0,0,1,1,1,0,0,...] 的向量转换为事件列表 [{'start':2,'end':5}, ...]。这一操作是计算区段级指标的前提。

    Trange = (0, len(y_test))
    # 用一个元组表示整个序列的时间范围，用于计算 Affiliation 指标时确定评估区间

    affiliation = pr_from_events(events_pred, events_gt, Trange)
    # 根据预测事件列表和真实事件列表计算两个指标：Affiliation Precision 和 Affiliation Recall
    # Affiliation Precision：预测的异常区段中有多少部分与真实异常区段重叠
    # Affiliation Recall：真实异常区段中有多少部分被预测区段覆盖

    aff_p, aff_r = affiliation['Affiliation_Precision'], affiliation['Affiliation_Recall']
    # 从字典 affiliation 中提取 Affiliation_Precision 和 Affiliation_Recall

    aff_f1 = 2 * (aff_p * aff_r) / (aff_p + aff_r)
    # 使用传统的调和平均公式，将精度和召回率结合成 F1 分数。

    pa_accuracy, pa_precision, pa_recall, pa_f_score = get_adjust_F1PA(y_test, pred_labels)
    # 该函数会根据真实标签和预测标签进行 区段调整（point-adjustment），使得当预测命中异常段的某一个点时，该段内其它点也视为命中，然后计算普通的精度、召回和 F1 分数

    vus_results = get_range_vus_roc(y_test, pred_labels, 100) # default slidingWindow = 100
    # get_range_vus_roc 使用滑动窗口来计算范围式曲线下面积 (Range‑AUC) 和体积下面积 (VUS), 参数 100 指定滑动窗口大小
    # 返回的字典通常包含 R_AUC_ROC（范围式 ROC 曲线的面积）、R_AUC_PR（范围式 PR 曲线的面积）、VUS_ROC（基于 ROC 的体积）、VUS_PR（基于 PR 的体积）
    
    score_list_simple = {
                  "Affiliation precision": aff_p,
                  "Affiliation recall": aff_r,
                  "Affiliation f1 score": aff_f1,
                  "R_AUC_ROC": vus_results["R_AUC_ROC"], 
                  "R_AUC_PR": vus_results["R_AUC_PR"],
                  "VUS_ROC": vus_results["VUS_ROC"],
                  "VUS_PR": vus_results["VUS_PR"]
                  }
    # 汇总指标构成一个易于读取的字典。这里没有包含 pa_accuracy、pa_precision 等调整后的 F1PA 指标，说明作者可能只希望输出 Affiliation 和 VUS 类指标。
    return score_list_simple

def get_adjust_F1PA(gt,pred):
    # 该函数用于计算 Point‑Adjust F1（又称 F1‑PA），它调整了预测标签，使得在真实异常区间被击中一次后，该区间内其它点也视为预测命中
    # pred: 预测标签（值为 0 或 1）。这里参数命名为 pred，但在调用时传入的是 y_test，导致实际传递可能是真实标签。这种顺序可能存在 bug
    # gt: 真实标签

    anomaly_state = False
    for i in range(len(gt)):
        if gt[i] == 1 and pred[i] == 1 and not anomaly_state: # 当遇到真实标签为 1 且预测也为 1，并且还未进入 anomaly_state 时，表示预测成功命中了异常段的某个点，此时需要对整个真实异常段进行扩散操作
            anomaly_state = True # 标记当前处于异常段内
            for j in range(i, 0, -1):
                if gt[j] == 0:
                    break
                else:
                    if pred[j] == 0:
                        pred[j] = 1
                        # 向前扩散：从命中点 i 向前回溯，直到遇到真实标签为 0 的位置为止，如果之前某个位置 pred[j]==0（模型没有预测为异常）
                        # 但真实标签是 1，则将预测改为 1，视为命中异常段。这确保整个异常段的前半部分都被标记为预测命中。
            for j in range(i, len(gt)):
                if gt[j] == 0:
                    break
                else:
                    if pred[j] == 0:
                        pred[j] = 1
                        # 向后扩散：类似向前扩散，但向后遍历直到真实标签为 0；将异常段后半部分的预测标签也改为 1
        elif gt[i] == 0:
            anomaly_state = False
            # 如果真实标签为 0，则说明不在异常段，anomaly_state 设为 False
        if anomaly_state:
            pred[i] = 1
            # 在异常段内，即使当前预测为 0，也将其改为 1，保证异常段内所有预测都为 1。这与向前/向后扩散的逻辑结合，使只要某一点预测命中，整个异常段都算命中

    accuracy = accuracy_score(gt, pred)
    precision, recall, f_score, support = precision_recall_fscore_support(gt, pred,average='binary')
    # 计算调整后的准确率 accuracy。使用 precision_recall_fscore_support 计算精度、召回率和 F1 分数（average='binary' 意味着针对二分类问题）。
    # support 返回正类样本数，这里没有用到。
    return accuracy, precision, recall, f_score
'''
该函数本意是实现论文中常用的 point-adjust F1 指标：对于真实异常区段，只要模型在该区段内某个点预测为 1，就认为整段均预测正确。因此会将预测标签在异常段内扩展为全 1。
'''