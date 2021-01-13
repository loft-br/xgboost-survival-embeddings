from sklearn.utils._testing import assert_raises

from tests.data import get_data
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


def test_convert_value_error():
    assert_raises(ValueError, convert_data_to_xgb_format, X_train, y_train, "blablabla")
