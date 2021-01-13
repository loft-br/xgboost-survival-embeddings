import pytest

from tests.data import get_data
from xgbse import XGBSEKaplanNeighbors, XGBSEKaplanTree

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


@pytest.mark.parametrize("model", [XGBSEKaplanNeighbors, XGBSEKaplanTree])
def test_neighbors_confidence_interval(model):
    xgbse = model()

    xgbse.fit(X_train, y_train)

    preds = xgbse.predict(X_test, return_ci=True)

    assert len(preds) == 3
