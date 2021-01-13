import pytest

from xgbse import (
    XGBSEDebiasedBCE,
    XGBSEKaplanNeighbors,
    XGBSEKaplanTree,
    XGBSEBootstrapEstimator,
)

from xgbse.metrics import concordance_index
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


def is_ci_width_consistent(bootstrap, X):

    mean1, high1, low1 = bootstrap.predict(X, return_ci=True, ci_width=0.683)
    mean2, high2, low2 = bootstrap.predict(X, return_ci=True, ci_width=0.95)

    equal_means = (mean1 == mean2).all().all()
    consistent_highs = (high2 >= high1).all().all()
    consistent_lows = (low2 <= low1).all().all()

    return equal_means & consistent_highs & consistent_lows


@pytest.mark.parametrize(
    "model", [XGBSEDebiasedBCE, XGBSEKaplanNeighbors, XGBSEKaplanTree]
)
def test_ci_width_consistency(model):

    model = model()
    bootstrap = XGBSEBootstrapEstimator(model)
    bootstrap.fit(X_train, y_train)

    assert is_ci_width_consistent(bootstrap, X_test)


def test_accuracy_improvement():

    base_model = XGBSEKaplanTree()
    base_model.fit(X_train, y_train)

    bootstrap = XGBSEBootstrapEstimator(base_model)
    bootstrap.fit(X_train, y_train)

    cind_base = concordance_index(y_test, base_model.predict(X_test))
    cind_boots = concordance_index(y_test, bootstrap.predict(X_test))

    assert cind_boots > cind_base
