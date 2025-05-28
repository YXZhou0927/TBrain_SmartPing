from sklearn.metrics import roc_auc_score, accuracy_score
import numpy as np

def evaluate_model(y_true_onehot: np.ndarray, y_pred: np.ndarray, method: str = "auc"):
    if method == "auc":
        return roc_auc_score(y_true_onehot, y_pred, average='micro', multi_class='ovr') # if only 2 classes, multi_class='ovr' is ignored
    elif method == "acc":
        return accuracy_score(y_true_onehot, y_pred > 0.5)
    else:
        raise ValueError("Unsupported evaluation method")