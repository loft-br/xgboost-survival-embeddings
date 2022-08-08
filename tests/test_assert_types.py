import numpy as np
import pandas as pd
import pytest
import xgboost as xgb

from tests.data import get_data
from xgbse import XGBSEDebiasedBCE, XGBSEStackedWeibull
from xgbse.assert_types import _assert_xgb_pre_fitted_model
from xgbse.converters import convert_data_to_xgb_format


(
    X_train,
    X_test,
    X_valid,
    T_train,
    T_test,
    T_valid,
    E_train,
    E_test,
    E_valid,
    y_train,
    y_test,
    y_valid,
    features,
) = get_data()

y = np.concatenate((y_train, y_test, y_valid))
X = np.concatenate((X_train, X_test, X_valid))
some_data = [13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1]

XGB_AFT_DEFAULT_PARAMS = {
    "objective": "survival:aft",
    "eval_metric": "aft-nloglik",
    "aft_loss_distribution": "normal",
    "aft_loss_distribution_scale": 1,
    "tree_method": "hist",
    "learning_rate": 5e-2,
    "max_depth": 8,
    "booster": "dart",
    "subsample": 0.5,
    "min_child_weight": 50,
    "colsample_bynode": 0.5,
}

XGB_COX_DEFAULT_PARAMS = {
    "objective": "survival:cox",
    "tree_method": "hist",
    "learning_rate": 5e-2,
    "max_depth": 8,
    "booster": "dart",
    "subsample": 0.5,
    "min_child_weight": 50,
    "colsample_bynode": 0.5,
}

OTHER_XGB_PARAMS = {
    "objective": "mse",
}

num_boost_round = 200
validation_data = None
early_stopping_rounds = None
verbose_eval = 0
time_bins = None


def pre_train_xgb(params):

    # pre training XGB Model
    dtrain = convert_data_to_xgb_format(X, y, params["objective"])

    # training XGB
    bst = xgb.train(
        params,
        dtrain,
        num_boost_round=num_boost_round,
        early_stopping_rounds=early_stopping_rounds,
        verbose_eval=verbose_eval,
    )

    return bst


xgb_cox = pre_train_xgb(XGB_COX_DEFAULT_PARAMS)
xgb_aft = pre_train_xgb(XGB_AFT_DEFAULT_PARAMS)


def assert_different_fitted_models(
    preds_with_pre_fitted_model, preds_without_pre_fitted_model
):
    assert ((preds_with_pre_fitted_model != preds_without_pre_fitted_model).all()).all()


@pytest.mark.parametrize(
    "pre_trained_xgb_model,params,data",
    [
        (xgb_cox, XGB_COX_DEFAULT_PARAMS, X_train),
        (xgb_aft, XGB_AFT_DEFAULT_PARAMS, X_train),
        pytest.param(xgb_cox, OTHER_XGB_PARAMS, X_train, marks=pytest.mark.xfail),
        pytest.param(
            xgb_cox, XGB_COX_DEFAULT_PARAMS, some_data, marks=pytest.mark.xfail
        ),
    ],
)
def test_assert_valid_xgb_pre_fitted_model(pre_trained_xgb_model, params, data):
    """
    Test if the pre-trained model is a valid xgb cox or aft model and if the data is a pandas dataframe or numpy array
    """
    assert _assert_xgb_pre_fitted_model([pre_trained_xgb_model, params], data) == [
        pre_trained_xgb_model,
        params,
    ]


@pytest.mark.parametrize("model", [XGBSEDebiasedBCE, XGBSEStackedWeibull])
@pytest.mark.parametrize(
    "pre_trained_xgb_model,params",
    [(xgb_cox, XGB_COX_DEFAULT_PARAMS), (xgb_aft, XGB_AFT_DEFAULT_PARAMS)],
)
def test_xgb_pre_fitted_model(model, pre_trained_xgb_model, params):
    """ "
    Test if the xgbse model using a pre-trained model is working as expected.
    Also test if the output is different from the model without a pre-trained model.
    """

    xgbse_pre_fitted = model()

    xgbse_pre_fitted.fit(
        X_train,
        y_train,
        num_boost_round=1000,
        validation_data=(X_valid, y_valid),
        early_stopping_rounds=10,
        verbose_eval=0,
        pre_fitted_xgb_model=[pre_trained_xgb_model, params],
    )

    xgbse_without_pre_fitted = model()

    xgbse_without_pre_fitted.fit(
        X_train,
        y_train,
        num_boost_round=1000,
        validation_data=(X_valid, y_valid),
        early_stopping_rounds=10,
        verbose_eval=0,
    )

    preds_pre_fitted = xgbse_pre_fitted.predict(X_test)
    preds_without_pre_fitted = xgbse_without_pre_fitted.predict(X_test)

    assert_different_fitted_models(preds_pre_fitted, preds_without_pre_fitted)
