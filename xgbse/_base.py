import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.base import BaseEstimator
from sklearn.neighbors import BallTree


class XGBSEBaseEstimator(BaseEstimator):
    """
    Base class for all estimators in xgbse. Implements explainability through prototypes.
    """

    def __init__(self):
        self.persist_train = False
        self.index_id = None
        self.tree = None
        self.bst = None

    def get_neighbors(
        self, query_data, index_data=None, query_id=None, index_id=None, n_neighbors=30
    ):
        """
        Search for portotypes (size: n_neighbors) for each unit in a
        dataframe X. If units array is specified, comparables will be returned using
        its identifiers. If not, a dataframe of comparables indexes for each sample
        in X is returned.

        Args:
            query_data (pd.DataFrame): Dataframe of features to be used as input

            query_id ([pd.Series, np.array]): Series or array of identification for each sample of query_data.
                Will be used in set_index if specified.

            index_id ([pd.Series, np.array]): Series or array of identification for each sample of index_id.
                If specified, comparables will be returned using this identifier.

            n_neighbors (int): Number of neighbors/comparables to be considered.

        Returns:
            comps_df (pd.DataFrame): A dataframe of comparables/neighbors for each
            evaluated sample. If units identifier is specified, the output dataframe
            is converted to use units the proper identifier for each sample. The
            reference sample is considered to be the index of the dataframe and
            its comparables are its specific row values.
        """

        if index_data is None and not self.persist_train:
            raise ValueError("please specify the index_data")

        if index_id is None and not self.persist_train:
            index_id = index_data.index.copy()

        if query_id is None:
            query_id = query_data.index.copy()

        if self.persist_train:
            index_id = self.index_id
            index = self.tree
        else:
            index_matrix = xgb.DMatrix(index_data)
            index_leaves = self.bst.predict(
                index_matrix,
                pred_leaf=True,
                iteration_range=(0, self.bst.best_iteration + 1),
            )

            if len(index_leaves.shape) == 1:
                index_leaves = index_leaves.reshape(-1, 1)
            index = BallTree(index_leaves, metric="hamming")

        query_matrix = xgb.DMatrix(query_data)
        query_leaves = self.bst.predict(
            query_matrix,
            pred_leaf=True,
            iteration_range=(0, self.bst.best_iteration + 1),
        )

        if len(query_leaves.shape) == 1:
            query_leaves = query_leaves.reshape(-1, 1)
        compset = index.query(query_leaves, k=n_neighbors + 1, return_distance=False)

        map_to_id = np.vectorize(lambda x: index_id[x])
        comparables = map_to_id(compset)
        comps_df = pd.DataFrame(comparables[:, 1:]).set_index(query_id)
        comps_df.columns = [f"neighbor_{n + 1}" for n in comps_df.columns]

        return comps_df


class DummyLogisticRegression(BaseEstimator):
    """
    Dummy logistic regression to be able to run XGBSEDebiasedBCE in timebuckets with only one class.
    """

    def fit(self, X, y):
        """
        Fits a dummy classifier to keep compatiblity with logistic regression.

        Args:
            X ([pd.DataFrame, np.array]): [not used]

            y (np.array): [targets in a logistic regression]
        """
        unique_class = np.unique(y)
        assert len(unique_class) == 1
        self.returns = unique_class[0]

    def predict(self, X):
        pass

    def predict_proba(self, X):
        y_hat = np.zeros((X.shape[0], 2))
        y_hat[:, self.returns] += 1.0
        return y_hat
