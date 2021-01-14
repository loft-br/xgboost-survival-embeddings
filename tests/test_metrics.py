import pandas as pd
import numpy as np

from xgbse.metrics import concordance_index, approx_brier_score, dist_calibration_score

from xgbse import XGBSEDebiasedBCE

from xgbse.non_parametric import get_time_bins, calculate_kaplan_vectorized
from tests.data import get_data

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


# generating Kaplan Meier for all tests

time_bins = get_time_bins(T_train, E_train, 100)

mean, high, low = calculate_kaplan_vectorized(
    T_train.values.reshape(1, -1), E_train.values.reshape(1, -1), time_bins
)

km_survival = pd.concat([mean] * len(y_train))
km_survival = km_survival.reset_index(drop=True)

# generating xgbse predictions for all tests

xgbse_model = XGBSEDebiasedBCE()

xgbse_model.fit(
    X_train,
    y_train,
    num_boost_round=1000,
    validation_data=(X_valid, y_valid),
    early_stopping_rounds=10,
    verbose_eval=0,
    time_bins=time_bins,
)

preds = xgbse_model.predict(X_test)

# generating dummy predictions

dummy_preds = pd.DataFrame({100: [0.5] * len(y_test)})


# functions to make testing easier
def is_brier_score_return_correct_len():
    return len(approx_brier_score(y_train, km_survival, aggregate=None)) == len(
        km_survival.columns
    )


def is_dist_cal_return_correct_len():
    return len(dist_calibration_score(y_train, km_survival, returns="histogram")) == 10


def is_dist_cal_return_correct_type():
    result = dist_calibration_score(y_train, km_survival, returns="all")
    return type(result) == dict


# testing


def test_concordance_index():

    assert concordance_index(y_train, km_survival) == 0.5
    assert concordance_index(y_test, preds) > 0.5
    assert np.isclose(
        concordance_index(y_test, T_test.values, risk_strategy="precomputed"),
        0,
        atol=0.02,
    )
    assert np.isclose(
        concordance_index(y_test, -T_test.values, risk_strategy="precomputed"),
        1,
        atol=0.02,
    )


def test_approx_brier_score():

    assert approx_brier_score(y_test, preds) < 0.25
    assert approx_brier_score(y_train, km_survival) < 0.2
    assert approx_brier_score(y_test, dummy_preds) == 0.25
    assert is_brier_score_return_correct_len()


def test_dist_calibration_score():

    assert dist_calibration_score(y_train, km_survival) > 0.90
    assert dist_calibration_score(y_train, km_survival, returns="statistic") < 1.0
    assert dist_calibration_score(y_train, km_survival, returns="max_deviation") < 0.01
    assert is_dist_cal_return_correct_len()
    assert is_dist_cal_return_correct_type()
