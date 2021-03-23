import numpy as np
import pytest
from xgbse.metrics import concordance_index

from tests.data import get_data
from xgbse import (
    XGBSEDebiasedBCE,
    XGBSEKaplanNeighbors,
    XGBSEKaplanTree,
    XGBSEStackedWeibull,
)

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


def monotonicity(survival_curves):
    dx = np.diff(survival_curves)
    return np.all(dx <= 0) or np.all(dx >= 0)


def between_01(survival_curves):
    lower = (0 <= survival_curves).all().all()  # au au
    upper = (survival_curves <= 1).all().all()
    return lower and upper


def assert_survival_curve(xgbse, test, preds, cindex):
    assert cindex > 0.5
    assert monotonicity(preds)
    assert between_01(preds)
    assert preds.shape[0] == test.shape[0]
    assert preds.shape[1] == xgbse.time_bins.shape[0]


@pytest.mark.parametrize(
    "model", [XGBSEDebiasedBCE, XGBSEKaplanNeighbors, XGBSEStackedWeibull]
)
def test_survival_curve(model):
    xgbse = model()

    xgbse.fit(
        X_train,
        y_train,
        num_boost_round=1000,
        validation_data=(X_valid, y_valid),
        early_stopping_rounds=10,
        verbose_eval=0,
    )

    preds = xgbse.predict(X_test)
    cindex = concordance_index(y_test, preds)

    assert_survival_curve(xgbse, X_test, preds, cindex)


@pytest.mark.parametrize(
    "model", [XGBSEDebiasedBCE, XGBSEKaplanNeighbors, XGBSEStackedWeibull]
)
def test_survival_curve_without_early_stopping(model):
    xgbse = model()

    xgbse.fit(
        X_train,
        y_train,
    )

    preds = xgbse.predict(X_test)
    cindex = concordance_index(y_test, preds)

    assert_survival_curve(xgbse, X_test, preds, cindex)


def test_survival_curve_tree():
    xgbse = XGBSEKaplanTree()

    xgbse.fit(X_train, y_train)

    preds = xgbse.predict(X_test)
    cindex = concordance_index(y_test, preds)

    assert_survival_curve(xgbse, X_test, preds, cindex)


@pytest.mark.parametrize("model", [XGBSEKaplanTree, XGBSEKaplanNeighbors])
def test_time_bins(model):
    xgbse = model()

    bins = np.linspace(100, 1000, 6)
    xgbse.fit(X_train, y_train, time_bins=bins)

    preds = xgbse.predict(X_test)
    assert (preds.columns == bins).all()
