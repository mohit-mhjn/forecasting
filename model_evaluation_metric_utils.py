import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
import logging

logger = logging.getLogger(__name__ + ": ")


def calculate_wmape(actuals, predictions):
    actual_total = np.sum(actuals)
    ape = np.abs(actuals - predictions) / actuals
    weights = actuals / actual_total
    wmape = ape * weights

    return sum(wmape.fillna(0))

def calculate_mape(actuals, predictions):
    # actual_total = np.sum(actuals)
    # ape = np.abs(actuals - predictions) / actuals
    ape = np.abs(actuals - predictions) / np.where(actuals == 0,np.nan,actuals)
    # ape = np.where(np.isinf(ape), np.nan, ape)
    mape = np.mean(ape)

    return mape

def calculate_rmse(actuals, predictions):
    mse = calculate_mse(actuals, predictions)
    rmse = np.sqrt(mse)

    return rmse

def calculate_mse(actuals, predictions):
    deltas = actuals - predictions
    se = np.square(deltas)
    mse = np.mean(se)

    return mse

def calculate_mae(actuals, predictions):
    deltas = actuals - predictions
    ae = np.abs(deltas)
    mae = np.mean(ae)

    return mae

def calculate_r2(actuals, predictions):
    return r2_score(y_pred=predictions, y_true=actuals)
