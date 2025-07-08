import numpy as np

def RSE(preds, targets):
    return np.sqrt(np.sum((preds - targets) ** 2) / np.sum((targets - targets.mean()) ** 2))

def CORR(preds, targets):
    return np.corrcoef(preds, targets)[0, 1]

def MAE(preds, targets):
    return np.mean(np.abs(preds - targets))

def MAPE(preds, targets):
    return np.mean(np.abs((preds - targets) / targets))

def RMSE(preds, targets):
    return np.sqrt(np.mean((preds - targets) ** 2))

def MSLE(preds, targets):
    return np.mean(np.square(np.log1p(preds) - np.log1p(targets)))

def ACC(preds, targets):
    return np.mean(preds == targets)

def metrics(preds, targets):
    return {
        "RSE": RSE(preds, targets),
        "CORR": CORR(preds, targets),
        "MAE": MAE(preds, targets),
        "MAPE": MAPE(preds, targets),
        "RMSE": RMSE(preds, targets),
        "MSLE": MSLE(preds, targets)
    }