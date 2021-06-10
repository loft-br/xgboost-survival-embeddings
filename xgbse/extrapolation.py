import numpy as np
import pandas as pd
from xgbse.non_parametric import _get_conditional_probs_from_survival
from xgbse.converters import hazard_to_survival


def extrapolate_constant_risk(survival, final_time, intervals, lags=-1):
    """
    Extrapolate a survival curve assuming constant risk.

    Args:
        survival (pd.DataFrame): A dataframe of survival probabilities
            for all times (columns), from a time_bins array, for all samples of X (rows).

        final_time (Float): Final time for extrapolation

        intervals (Int): Time in each interval between last time in survival dataframe and final time

        lags (Int): Lags to compute constant risk.
            if negative, will use the last "lags" values
            if positive, will remove the first "lags" values
            if 0, will use all values

    Returns:
        pd.DataFrame: Survival dataset with appended extrapolated windows
    """

    last_time = survival.columns[-1]
    # creating windows for extrapolation
    # here we sum intervals in times to exclude the last time, that already is in surv dataframe and
    #  to include final time in resulting dataframe
    extrap_windows = np.arange(last_time + intervals, final_time + intervals, intervals)

    # calculating conditionals and hazard at each time window
    hazards = _get_conditional_probs_from_survival(survival)

    # calculating avg hazard for desired lags
    constant_haz = hazards.values[:, lags:].mean(axis=1).reshape(-1, 1)

    # repeat hazard for n_windows required
    constant_haz = np.tile(constant_haz, len(extrap_windows))

    constant_haz = pd.DataFrame(constant_haz, columns=extrap_windows)

    hazards = pd.concat([hazards, constant_haz], axis=1)

    return hazard_to_survival(hazards)
