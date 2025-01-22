import numpy as np

def RMSE(pred, true):
    return np.sqrt(np.mean((pred - true) ** 2))

def MAE(pred, true):
    return np.mean(np.abs(pred - true))

def R2(pred, true):
    return 1 - np.mean((pred - true) ** 2) / np.var(true)

def MAPE(pred, true):
    return np.mean(np.abs((pred - true) / true))

def metric(pred, true):
    rmse = RMSE(pred, true)
    mae = MAE(pred, true)
    r_squared = R2(pred, true)
    mape = MAPE(pred, true)

    return rmse, mae, r_squared, mape
