import numpy as np
import pandas as pd
import xgboost as xgb


def _assert_xgb_pre_fitted_model(pre_fitted_xgb_model, X_train):
    """
    Asserts a pre-trained XGBoost model (using objective survival:aft or survival:cox)
    is passed to XGBDebiasedBCE in pre_trained_xgb argument.
    Two tests are performed: assert model XGBoost Booster type and pre-trained model
    predict capability.

    Args:
        pre_fitted_xgb_model (list containing [xgb.core.Booster, dict]): a list with
                [pre-trained XGBoost model, dict of pre-trained model parameters with
                'survival:aft' or 'survival:cox' as objective parameter]
        X_train ([pd.DataFrame, np.ndarray]): training data use to fit BCE posterior model

    Returns:
        pre_fitted_xgb_model (xgb.core.Booster): a list with
                [verified pre-trained XGBoost model, dict of pre-trained model parameters with
                'survival:aft' or 'survival:cox' as objective parameter]
    """

    assert isinstance(
        pre_fitted_xgb_model[0], xgb.core.Booster
    ), """
    Pre-trained model must be an XGBoost trained using either
    survival:aft or survival:cox objective parameters
    """
    assert pre_fitted_xgb_model[1]["objective"] in [
        "survival:aft",
        "survival:cox",
    ], """
    Pre-trained model must be an XGBoost trained using either
    survival:aft or survival:cox objective parameters
    """

    pre_fitted_xgb_model[0].set_param(
        {"objective": pre_fitted_xgb_model[1]["objective"]}
    )

    if isinstance(X_train, pd.DataFrame):
        sample = xgb.DMatrix(X_train.sample(1))
    elif isinstance(X_train, np.ndarray):
        sample = xgb.DMatrix(
            X_train[np.random.randint(X_train.shape[0], size=1), :],
            feature_names=pre_fitted_xgb_model[0].feature_names,
        )
    else:
        raise ValueError("X_train must be either a pd.DataFrame or a np.ndarray")
    try:
        pred = pre_fitted_xgb_model[0].predict(sample)
        assert isinstance(pred, np.ndarray)
    except:
        raise ValueError(
            """
    Pre-trained model must be an XGBoost trained using
    survival:aft or survival:cox parameters
    """
        )

    return pre_fitted_xgb_model
