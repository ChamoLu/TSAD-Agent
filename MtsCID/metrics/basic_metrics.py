import numpy as np


def _as_binary(y):
    return np.asarray(y).astype(int).reshape(-1)


def _safe_div(num, den):
    return 0.0 if den == 0 else float(num) / float(den)


def confusion_counts(y_true, y_pred):
    y_true = _as_binary(y_true)
    y_pred = _as_binary(y_pred)
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    return tp, tn, fp, fn


def precision_score(y_true, y_pred, average='binary', zero_division=0):
    tp, _, fp, _ = confusion_counts(y_true, y_pred)
    if tp + fp == 0:
        return float(zero_division)
    return _safe_div(tp, tp + fp)


def recall_score(y_true, y_pred, average='binary', zero_division=0):
    tp, _, _, fn = confusion_counts(y_true, y_pred)
    if tp + fn == 0:
        return float(zero_division)
    return _safe_div(tp, tp + fn)


def f1_score(y_true, y_pred, average='binary', zero_division=0):
    precision = precision_score(y_true, y_pred, average=average, zero_division=zero_division)
    recall = recall_score(y_true, y_pred, average=average, zero_division=zero_division)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def accuracy_score(y_true, y_pred):
    y_true = _as_binary(y_true)
    y_pred = _as_binary(y_pred)
    if len(y_true) == 0:
        return 0.0
    return float(np.mean(y_true == y_pred))


def precision_recall_fscore_support(y_true, y_pred, average='binary', zero_division=0):
    precision = precision_score(y_true, y_pred, average=average, zero_division=zero_division)
    recall = recall_score(y_true, y_pred, average=average, zero_division=zero_division)
    f1 = f1_score(y_true, y_pred, average=average, zero_division=zero_division)
    support = int(np.sum(_as_binary(y_true) == 1))
    return precision, recall, f1, support


def auc(x, y):
    x = np.asarray(x, dtype=float).reshape(-1)
    y = np.asarray(y, dtype=float).reshape(-1)
    if len(x) < 2:
        return 0.0
    order = np.argsort(x)
    return float(np.trapz(y[order], x[order]))


def roc_curve(y_true, y_score):
    y_true = _as_binary(y_true)
    y_score = np.asarray(y_score, dtype=float).reshape(-1)
    if len(y_true) == 0:
        return np.array([0.0, 1.0]), np.array([0.0, 1.0]), np.array([np.inf])

    thresholds = np.r_[np.inf, np.unique(y_score)[::-1]]
    positives = np.sum(y_true == 1)
    negatives = np.sum(y_true == 0)
    tpr = []
    fpr = []
    for threshold in thresholds:
        pred = (y_score >= threshold).astype(int)
        tp, _, fp, _ = confusion_counts(y_true, pred)
        tpr.append(_safe_div(tp, positives))
        fpr.append(_safe_div(fp, negatives))
    return np.asarray(fpr), np.asarray(tpr), thresholds


def roc_auc_score(y_true, y_score):
    y_true = _as_binary(y_true)
    if len(np.unique(y_true)) < 2:
        return 0.0
    fpr, tpr, _ = roc_curve(y_true, y_score)
    return auc(fpr, tpr)


def precision_recall_curve(y_true=None, probas_pred=None, y_score=None, **kwargs):
    if y_true is None:
        y_true = kwargs.get('label')
    if y_score is None:
        y_score = probas_pred if probas_pred is not None else kwargs.get('score')

    y_true = _as_binary(y_true)
    y_score = np.asarray(y_score, dtype=float).reshape(-1)
    if len(y_true) == 0:
        return np.array([0.0]), np.array([0.0]), np.array([0.0])

    thresholds = np.unique(y_score)[::-1]
    if len(thresholds) == 0:
        thresholds = np.array([0.0])

    precision = []
    recall = []
    for threshold in thresholds:
        pred = (y_score >= threshold).astype(int)
        precision.append(precision_score(y_true, pred, zero_division=1))
        recall.append(recall_score(y_true, pred, zero_division=0))
    return np.asarray(precision), np.asarray(recall), thresholds


def average_precision_score(y_true, y_score):
    y_true = _as_binary(y_true)
    if np.sum(y_true) == 0:
        return 0.0
    precision, recall, _ = precision_recall_curve(y_true=y_true, probas_pred=y_score)
    order = np.argsort(recall)
    return float(np.trapz(precision[order], recall[order]))
