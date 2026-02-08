# impl/metrics.py

from sklearn.metrics import f1_score, roc_auc_score
import numpy as np


def binaryf1(pred, label):
    pred_i = (pred > 0).astype(np.int64)
    label_i = label.reshape(pred.shape[0], -1)
    return f1_score(label_i, pred_i, average="micro")


def microf1(pred, label):
    pred_i = np.argmax(pred, axis=1)
    return f1_score(label, pred_i, average="micro")


def _softmax(x, axis=1):
    x = x - np.max(x, axis=axis, keepdims=True)
    ex = np.exp(x)
    return ex / np.sum(ex, axis=axis, keepdims=True)


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def auroc(pred, label):
    """
    pred: numpy
      - binary: (N,) or (N,1) logits
      - multiclass: (N,C) logits
      - multilabel: (N,K) logits, label (N,K) in {0,1}
    label: numpy
      - binary/multiclass: (N,)
      - multilabel: (N,K)
    """
    pred = np.asarray(pred)
    label = np.asarray(label)

    # flatten trivial last dim
    if pred.ndim == 2 and pred.shape[1] == 1:
        pred = pred.reshape(-1)

    # ----- multilabel -----
    if label.ndim == 2 and label.shape[1] > 1:
        # scores should be probabilities for AUROC; sigmoid is standard
        score = _sigmoid(pred)
        # average can be "macro"/"micro"/"weighted" depending on your convention
        return roc_auc_score(label, score, average="macro")

    # now label is (N,)
    label = label.reshape(-1)

    # ----- binary -----
    # binary can arrive as (N,) logits
    # NOTE: roc_auc_score is fine with logits (ranking-based), but keep consistent.
    uniq = np.unique(label)
    if pred.ndim == 1 and uniq.size <= 2:
        return roc_auc_score(label, pred)

    # ----- multiclass -----
    # pred should be (N,C)
    if pred.ndim != 2:
        raise ValueError(f"unexpected pred shape for multiclass AUROC: {pred.shape}")

    prob = _softmax(pred, axis=1)
    # sklearn requires multi_class to be set for multiclass
    # 'ovr' is the common default; 'ovo' also possible.
    return roc_auc_score(label, prob, multi_class="ovr", average="macro")
