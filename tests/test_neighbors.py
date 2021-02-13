import pytest

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


def assert_neighbors(df, comps, n_neighbors):
    assert comps.shape[1] == n_neighbors  # exact number of comps

    i_am_in = [index not in row.values for index, row in comps.iterrows()]
    assert any(i_am_in)

    # comps are not too different from index (2.5 --> threshold for random)
    assert df.comps_error.median() < 2.5

    # comps are not too different themselves (1 --> threshold for random)
    assert df.varcof.mean() < 1


@pytest.mark.parametrize(
    "model", [XGBSEDebiasedBCE, XGBSEKaplanNeighbors, XGBSEStackedWeibull]
)
def test_model_neighbors_persist_false(model):
    n_neighbors = 30
    test_feature = "pnodes"

    xgbse = model()

    xgbse.fit(
        X_train,
        y_train,
        validation_data=(X_valid, y_valid),
        early_stopping_rounds=10,
        verbose_eval=0,
        persist_train=False,
    )

    df = X_test.copy()
    comps = xgbse.get_neighbors(X_test, index_data=X_test, n_neighbors=n_neighbors)
    df.reset_index(inplace=True)

    df.loc[:, "comps_mean"] = df["index"].apply(
        lambda idx: df.loc[df["index"].isin(comps.loc[idx]), test_feature].mean()
    )

    df.loc[:, "comps_std"] = df["index"].apply(
        lambda idx: df.loc[df["index"].isin(comps.loc[idx]), test_feature].std()
    )

    df.loc[:, "varcof"] = df.comps_std / df.comps_mean
    df.loc[:, "comps_error"] = abs(df[test_feature] - df.comps_mean)

    assert_neighbors(df, comps, n_neighbors)


@pytest.mark.parametrize(
    "model",
    [XGBSEDebiasedBCE, XGBSEKaplanNeighbors, XGBSEKaplanTree, XGBSEStackedWeibull],
)
def test_model_neighbors_persist_true(model):
    n_neighbors = 30
    test_feature = "pnodes"

    xgbse = model()

    xgbse.fit(X_train, y_train, persist_train=True)

    df = X_test.copy()
    comps = xgbse.get_neighbors(X_test, n_neighbors=n_neighbors)
    df.reset_index(inplace=True)

    df.loc[:, "comps_mean"] = df["index"].apply(
        lambda idx: X_train.loc[X_train.index.isin(comps.loc[idx]), test_feature].mean()
    )
    df.loc[:, "comps_std"] = df["index"].apply(
        lambda idx: X_train.loc[X_train.index.isin(comps.loc[idx]), test_feature].std()
    )

    df.loc[:, "varcof"] = df.comps_std / df.comps_mean
    df.loc[:, "comps_error"] = abs(df[test_feature] - df.comps_mean)

    assert_neighbors(df, comps, n_neighbors)
