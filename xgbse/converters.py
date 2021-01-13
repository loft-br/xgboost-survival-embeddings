import numpy as np
import xgboost as xgb


def convert_to_structured(T, E):
    """Converts data in T, E format to structured numpy array
       so we can use integrated_brier_score functions from sksurv

    Args:
        T (np.array): array of times
        E (np.array): array of events

    Returns:
        np.array: structured array containing the boolean event indicator
            as first field, and time of event or time of censoring as second field
    """
    # dtypes for conversion
    default_dtypes = {"names": ("c1", "c2"), "formats": ("bool", "f8")}

    # concat of events and times
    concat = list(zip(E.values, T.values))

    # return structured array
    return np.array(concat, dtype=default_dtypes)


def convert_y(y):
    """convert structured array y into event indicator and time of event

    Args:
        y (structured array(numpy.bool_, numpy.number)): binary event indicator as first field,
            and time of event or time of censoring as second field.

    Returns:
        T ([np.array, pd.Series]): time of events
        E ([np.array, pd.Series]): binary event indicator
    """
    event_field, time_field = y.dtype.names
    return y[event_field], y[time_field]


def convert_data_to_xgb_format(X, y, objective):
    """Convert (X, y) data format to xgb.DMatrix format, either using cox or aft models.

    Args:
        X ([pd.DataFrame, np.array]): features to be used while fitting
            XGBoost model
        y (structured array(numpy.bool_, numpy.number)): binary event indicator as first field,
            and time of event or time of censoring as second field.
        objective (string): one of 'survival:aft' or 'survival:cox'

    Returns:
        xgb.DMatrix: data to train xgb
    """

    E, T = convert_y(y)

    # converting data to xgb format
    if objective == "survival:aft":
        d_matrix = build_xgb_aft_dmatrix(X, T, E)

    elif objective == "survival:cox":
        d_matrix = build_xgb_cox_dmatrix(X, T, E)

    else:
        raise ValueError("Objective not supported. Use survival:cox or survival:aft")

    return d_matrix


# Building XGB Design matrices - AFT and Cox Model
def build_xgb_aft_dmatrix(X, T, E):
    """Builds a XGB DMatrix using specified Data Frame of features (X)
     arrays of times (T) and censors/events (E).

    Args:
        X ([pd.DataFrame, np.array]): Data Frame to be converted to
            XGBDMatrix format.
        T ([np.array, pd.Series]): Array of times.
        E ([np.array, pd.Series]): Array of censors(False) / events(True).

    Returns:
        xgb.DMatrix: A XGB DMatrix is returned including features and target.
    """

    d_matrix = xgb.DMatrix(X)

    y_lower_bound = T
    y_upper_bound = np.where(E, T, np.inf)
    d_matrix.set_float_info("label_lower_bound", y_lower_bound.copy())
    d_matrix.set_float_info("label_upper_bound", y_upper_bound.copy())

    return d_matrix


def build_xgb_cox_dmatrix(X, T, E):
    """Builds a XGB DMatrix using specified Data Frame of features (X)
        arrays of times (T) and censors/events (E).

    Args:
        X ([pd.DataFrame, np.array]): Data Frame to be converted to XGBDMatrix format.
        T ([np.array, pd.Series]): Array of times.
        E ([np.array, pd.Series]): Array of censors(False) / events(True).

    Returns:
        (DMatrix): A XGB DMatrix is returned including features and target.
    """

    target = np.where(E, T, -T)

    return xgb.DMatrix(X, label=target)
