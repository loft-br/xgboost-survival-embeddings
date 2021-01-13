import numpy as np
import pandas as pd
from xgbse.non_parametric import _get_conditional_probs_from_survival


def extrapolate_constant_risk(survival, final_time, n_windows, lags=-1):
    """
    Extrapolate a survival curve assuming constant risk.

    Args:
        survival (pd.DataFrame): A dataframe of survival probabilities
            for all times (columns), from a time_bins array, for all samples of X (rows).
        final_time (Float): final time for extrapolation
        n_windows (Int): number of time windows to compute from last time window in survival to final_time
        lags (Int): lags to compute constant risk.
            if negative, will use the last "lags" values
            if positive, will remove the first "lags" values
            if 0, will use all values

    Returns:
        pd.DataFrame: survival dataset with appended extrapolated windows
    """

    # calculating conditionals and risk at each time window
    conditionals = _get_conditional_probs_from_survival(survival)
    window_risk = 1 - conditionals

    # calculating window sizes
    time_bins = window_risk.columns.to_series()
    window_sizes = time_bins - time_bins.shift(1).fillna(0)

    # using window sizes to calculate risk per unit time and average risk
    risk_per_unit_time = np.power(window_risk, 1 / window_sizes)
    average_risk = risk_per_unit_time.iloc[:, lags:].mean(axis=1)

    # creating windows for extrapolation
    last_time = survival.columns[-1]
    extrap_windows = np.linspace(last_time, final_time, n_windows) - last_time

    # loop for extrapolated windows
    for delta_t in extrap_windows:

        # running constant risk extrapolation
        extrap_survival = np.power(average_risk, delta_t) * survival.iloc[:, -1]
        extrap_survival = pd.Series(extrap_survival, name=last_time + delta_t)
        survival = pd.concat([survival, extrap_survival], axis=1)

    return survival
