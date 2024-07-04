import pytest

from tests.data import get_data
from xgbse._feature_extractors import FeatureExtractor

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


def test_wrong_objective():
    with pytest.raises(ValueError):
        FeatureExtractor(xgb_params={"objective": "reg:squarederror"})


def test_no_objective():
    assert FeatureExtractor(xgb_params={}).xgb_params["objective"] == "survival:aft"


def test_predict_leaves_early_stop():
    xgbse = FeatureExtractor()
    early_stopping_rounds = 10
    xgbse.fit(
        X_train,
        y_train,
        num_boost_round=1000,
        validation_data=(X_valid, y_valid),
        early_stopping_rounds=early_stopping_rounds,
        verbose_eval=0,
    )
    prediction = xgbse.predict_leaves(X_test)
    assert prediction.shape[0] == X_test.shape[0]
    assert (
        xgbse.bst.best_iteration
        <= prediction.shape[1]
        <= xgbse.bst.best_iteration + 1 + early_stopping_rounds
    )


def test_predict_leaves_no_early_stop():
    xgbse = FeatureExtractor()
    xgbse.fit(
        X_train,
        y_train,
        num_boost_round=100,
        validation_data=(X_valid, y_valid),
        early_stopping_rounds=None,
        verbose_eval=0,
    )
    assert xgbse.predict_leaves(X_test).shape == (X_test.shape[0], 100)


def test_predict_hazard_early_stop():
    xgbse = FeatureExtractor()
    xgbse.fit(
        X_train,
        y_train,
        num_boost_round=1000,
        validation_data=(X_valid, y_valid),
        early_stopping_rounds=10,
        verbose_eval=0,
    )
    assert xgbse.predict_hazard(X_test).shape == (X_test.shape[0],)


def test_predict_hazard_no_early_stop():
    xgbse = FeatureExtractor()
    xgbse.fit(
        X_train,
        y_train,
        num_boost_round=100,
        validation_data=(X_valid, y_valid),
        early_stopping_rounds=None,
        verbose_eval=0,
    )
    assert xgbse.predict_hazard(X_test).shape == (X_test.shape[0],)


def test_feature_importances():
    xgbse = FeatureExtractor()
    xgbse.fit(
        X_train,
        y_train,
        num_boost_round=100,
        validation_data=(X_valid, y_valid),
        early_stopping_rounds=None,
        verbose_eval=0,
    )
    assert xgbse.feature_importances_ == xgbse.bst.get_score()


def test_predict_not_fitted():
    xgbse = FeatureExtractor()
    with pytest.raises(ValueError):
        xgbse.predict_leaves(X_test)
    with pytest.raises(ValueError):
        xgbse.predict_hazard(X_test)
