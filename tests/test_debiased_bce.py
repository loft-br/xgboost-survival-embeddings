import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import train_test_split

from xgbse import XGBSEDebiasedBCE


@pytest.fixture
def sample_data():
    np.random.seed(42)
    n_samples = 1000

    # Generate features
    X = pd.DataFrame(
        {
            "numeric1": np.random.normal(0, 1, n_samples),
            "numeric2": np.random.normal(0, 1, n_samples),
            "categorical1": pd.Categorical(
                np.random.choice(["A", "B", "C"], n_samples)
            ),
            "categorical2": pd.Categorical(
                np.random.choice(["X", "Y", "Z"], n_samples)
            ),
        }
    )

    # Generate survival times and events
    T = np.random.exponential(scale=1, size=n_samples)
    E = np.random.binomial(n=1, p=0.7, size=n_samples)

    y = np.array(list(zip(E, T)), dtype=[("E", bool), ("T", float)])

    return train_test_split(X, y, test_size=0.2, random_state=42)


def test_xgbse_debiased_bce_without_categorical(sample_data):
    X_train, X_test, y_train, y_test = sample_data
    # Use only numeric features
    X_train = X_train[["numeric1", "numeric2"]]
    X_test = X_test[["numeric1", "numeric2"]]

    model = XGBSEDebiasedBCE(n_jobs=-1)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)

    assert isinstance(preds, pd.DataFrame)
    assert preds.shape[0] == X_test.shape[0]
    assert (preds.values >= 0).all() and (preds.values <= 1).all()


def test_xgbse_debiased_bce_with_categorical(sample_data):
    X_train, X_test, y_train, y_test = sample_data

    model = XGBSEDebiasedBCE(n_jobs=-1, enable_categorical=True)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)

    assert isinstance(preds, pd.DataFrame)
    assert preds.shape[0] == X_test.shape[0]
    assert (preds.values >= 0).all() and (preds.values <= 1).all()


def test_xgbse_debiased_bce_feature_importance(sample_data):
    X_train, X_test, y_train, y_test = sample_data

    model = XGBSEDebiasedBCE(n_jobs=-1, enable_categorical=True)
    model.fit(X_train, y_train)

    assert hasattr(model, "feature_importances_")
    assert isinstance(model.feature_importances_, dict)
    assert len(model.feature_importances_) > 0


def test_xgbse_debiased_bce_get_neighbors(sample_data):
    X_train, X_test, y_train, y_test = sample_data

    model = XGBSEDebiasedBCE(n_jobs=-1, enable_categorical=True)
    model.fit(X_train, y_train, persist_train=True)

    neighbors = model.get_neighbors(X_test.iloc[:5], n_neighbors=3)

    assert isinstance(neighbors, pd.DataFrame)
    assert neighbors.shape == (5, 3)


def test_xgbse_debiased_bce_invalid_data(sample_data):
    X_train, X_test, y_train, y_test = sample_data

    model = XGBSEDebiasedBCE(n_jobs=-1)

    with pytest.raises(ValueError):
        # Try to fit with invalid y data
        invalid_y = np.array(
            [(True, -1.0)] * len(y_train), dtype=[("E", bool), ("T", float)]
        )
        model.fit(X_train, invalid_y)


if __name__ == "__main__":
    pytest.main([__file__])
