from xgbse import XGBSEDebiasedBCE
from xgbse.extrapolation import extrapolate_constant_risk
from xgbse.non_parametric import get_time_bins, calculate_kaplan_vectorized
from tests.data import get_data
from tests.test_survival_curves import monotonicity, between_01

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
interval = 10
final_time = max(time_bins) + 1000
n_windows = 100

preds_ext = extrapolate_constant_risk(preds, final_time=final_time, intervals=interval)


def extrapolation_shape():
    return preds.shape[1] + n_windows == preds_ext.shape[1]


def last_col():
    return int(preds_ext.columns[-1]) == final_time


def test_extrapolation():
    assert extrapolation_shape()
    assert last_col()
    assert between_01(preds_ext)
    assert monotonicity(preds_ext)
