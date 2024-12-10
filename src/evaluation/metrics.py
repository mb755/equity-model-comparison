import numpy as np


def r_squared(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - (ss_res / ss_tot)


def mean_squared_error(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)


def pnl_estimate(y_true, y_pred):
    # e = predicted return, t = true return
    # PnL = sign(e*t)*e^2
    # e*t > 0 => sign = 1, else sign = -1
    sign_factor = np.sign(y_pred * y_true) * np.abs(y_true) - 2 / 3 * np.abs(y_pred)
    return np.mean(sign_factor * ((y_pred * 10_000) ** 2))
