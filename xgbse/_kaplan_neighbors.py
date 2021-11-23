import warnings

import numpy as np
import pandas as pd
import xgboost as xgb
import scipy.stats as st
from sklearn.neighbors import BallTree

from xgbse._base import XGBSEBaseEstimator
from xgbse.converters import convert_data_to_xgb_format, convert_y
from xgbse.non_parametric import (
    calculate_kaplan_vectorized,
    get_time_bins,
    calculate_interval_failures,
)

# at which percentiles will the KM predict
KM_PERCENTILES = np.linspace(0, 1, 11)
DEFAULT_PARAMS = {
    "objective": "survival:aft",
    "eval_metric": "aft-nloglik",
    "aft_loss_distribution": "normal",
    "aft_loss_distribution_scale": 1,
    "tree_method": "hist",
    "learning_rate": 5e-2,
    "max_depth": 8,
    "booster": "dart",
    "subsample": 0.5,
    "min_child_weight": 50,
    "colsample_bynode": 0.5,
}

DEFAULT_PARAMS_TREE = {
    "objective": "survival:cox",
    "eval_metric": "cox-nloglik",
    "tree_method": "exact",
    "max_depth": 100,
    "booster": "dart",
    "subsample": 1.0,
    "min_child_weight": 30,
    "colsample_bynode": 1.0,
}


# class to turn XGB into a kNN with a kaplan meier in the NNs
class XGBSEKaplanNeighbors(XGBSEBaseEstimator):
    """
    Convert xgboost into a nearest neighbor model, where we use hamming distance to define
    similar elements as the ones that co-ocurred the most at the ensemble terminal nodes.

    Then, at each neighbor-set compute survival estimates with the Kaplan-Meier estimator.

    !!! Note
        * We recommend using dart as the booster to prevent any tree
        to dominate variance in the ensemble and break the leaf co-ocurrence similarity logic.

        * This method can be very expensive at scales of hundreds of thousands of samples,
        due to the nearest neighbor search, both on training (construction of search index) and scoring (actual search).

    Read more in [How XGBSE works](https://loft-br.github.io/xgboost-survival-embeddings/how_xgbse_works.html).

    """

    def __init__(self, xgb_params=None, n_neighbors=30, radius=None):
        """
        Args:
            xgb_params (Dict): Parameters for XGBoost model.
                If not passed, the following default parameters will be used:

                ```
                DEFAULT_PARAMS = {
                    "objective": "survival:aft",
                    "eval_metric": "aft-nloglik",
                    "aft_loss_distribution": "normal",
                    "aft_loss_distribution_scale": 1,
                    "tree_method": "hist",
                    "learning_rate": 5e-2,
                    "max_depth": 8,
                    "booster": "dart",
                    "subsample": 0.5,
                    "min_child_weight": 50,
                    "colsample_bynode": 0.5,
                }
                ```

                Check <https://xgboost.readthedocs.io/en/latest/parameter.html> for more options.

            n_neighbors (Int): Number of neighbors for computing KM estimates

            radius (Float): If set, uses a radius around the point for neighbors search
        """
        if xgb_params is None:
            xgb_params = DEFAULT_PARAMS

        self.xgb_params = xgb_params
        self.n_neighbors = n_neighbors
        self.radius = radius
        self.persist_train = False
        self.index_id = None
        self.radius = None
        self.feature_importances_ = None

    def fit(
        self,
        X,
        y,
        num_boost_round=1000,
        validation_data=None,
        early_stopping_rounds=None,
        verbose_eval=0,
        persist_train=True,
        index_id=None,
        time_bins=None,
    ):
        """
        Transform feature space by fitting a XGBoost model and outputting its leaf indices.
        Build search index in the new space to allow nearest neighbor queries at scoring time.

        Args:
            X ([pd.DataFrame, np.array]): Design matrix to fit XGBoost model

            y (structured array(numpy.bool_, numpy.number)): Binary event indicator as first field,
                and time of event or time of censoring as second field.

            num_boost_round (Int): Number of boosting iterations.

            validation_data (Tuple): Validation data in the format of a list of tuples [(X, y)]
                if user desires to use early stopping

            early_stopping_rounds (Int): Activates early stopping.
                Validation metric needs to improve at least once
                in every **early_stopping_rounds** round(s) to continue training.
                See xgboost.train documentation.

            verbose_eval ([Bool, Int]): Level of verbosity. See xgboost.train documentation.

            persist_train (Bool): Whether or not to persist training data to use explainability
                through prototypes

            index_id (pd.Index): User defined index if intended to use explainability
                through prototypes

            time_bins (np.array): Specified time windows to use when making survival predictions

        Returns:
            XGBSEKaplanNeighbors: Fitted instance of XGBSEKaplanNeighbors
        """

        self.E_train, self.T_train = convert_y(y)
        if time_bins is None:
            time_bins = get_time_bins(self.T_train, self.E_train)
        self.time_bins = time_bins

        # converting data to xgb format
        dtrain = convert_data_to_xgb_format(X, y, self.xgb_params["objective"])

        # converting validation data to xgb format
        evals = ()
        if validation_data:
            X_val, y_val = validation_data
            dvalid = convert_data_to_xgb_format(
                X_val, y_val, self.xgb_params["objective"]
            )
            evals = [(dvalid, "validation")]

        # training XGB
        self.bst = xgb.train(
            self.xgb_params,
            dtrain,
            num_boost_round=num_boost_round,
            early_stopping_rounds=early_stopping_rounds,
            evals=evals,
            verbose_eval=verbose_eval,
        )
        self.feature_importances_ = self.bst.get_score()

        # creating nearest neighbor index
        leaves = self.bst.predict(
            dtrain, pred_leaf=True, iteration_range=(0, self.bst.best_iteration + 1)
        )

        self.tree = BallTree(leaves, metric="hamming", leaf_size=40)

        if persist_train:
            self.persist_train = True
            if index_id is None:
                index_id = X.index.copy()
        self.index_id = index_id

        return self

    def predict(
        self,
        X,
        time_bins=None,
        return_ci=False,
        ci_width=0.683,
        return_interval_probs=False,
    ):
        """
        Make queries to nearest neighbor search index build on the transformed XGBoost space.
        Compute a Kaplan-Meier estimator for each neighbor-set. Predict the KM estimators.

        Args:
            X (pd.DataFrame): Dataframe with samples to generate predictions

            time_bins (np.array): Specified time windows to use when making survival predictions

            return_ci (Bool): Whether to return confidence intervals via the Exponential Greenwood formula

            ci_width (Float): Width of confidence interval

            return_interval_probs (Bool): Boolean indicating if interval probabilities are
                supposed to be returned. If False the cumulative survival is returned.


        Returns:
            (pd.DataFrame): A dataframe of survival probabilities
            for all times (columns), from a time_bins array, for all samples of X
            (rows). If return_interval_probs is True, the interval probabilities are returned
            instead of the cumulative survival probabilities.

            upper_ci (np.array): Upper confidence interval for the survival
            probability values

            lower_ci (np.array): Lower confidence interval for the survival
            probability values
        """

        # converting to xgb format
        d_matrix = xgb.DMatrix(X)

        # getting leaves and extracting neighbors
        leaves = self.bst.predict(
            d_matrix, pred_leaf=True, iteration_range=(0, self.bst.best_iteration + 1)
        )

        if self.radius:
            assert self.radius > 0, "Radius must be positive"

            neighs, _ = self.tree.query_radius(
                leaves, r=self.radius, return_distance=True
            )

            number_of_neighbors = np.array([len(neigh) for neigh in neighs])

            if np.argwhere(number_of_neighbors == 1).shape[0] > 0:
                # If there is at least one sample without neighbors apart from itself
                # a warning is raised suggesting a radius increase
                warnings.warn(
                    "Warning: Some samples don't have neighbors apart from itself. Increase the radius",
                    RuntimeWarning,
                )
        else:
            _, neighs = self.tree.query(leaves, k=self.n_neighbors)

        # gathering times and events/censors for neighbor sets
        T_neighs = self.T_train[neighs]
        E_neighs = self.E_train[neighs]

        # vectorized (very fast!) implementation of Kaplan Meier curves
        if time_bins is None:
            time_bins = self.time_bins

        # calculating z-score from width
        z = st.norm.ppf(0.5 + ci_width / 2)

        preds_df, upper_ci, lower_ci = calculate_kaplan_vectorized(
            T_neighs, E_neighs, time_bins, z
        )

        if return_ci and return_interval_probs:
            raise ValueError(
                "Confidence intervals for interval probabilities is not supported. Choose between return_ci and return_interval_probs."
            )

        if return_interval_probs:
            preds_df = calculate_interval_failures(preds_df)
            return preds_df

        if return_ci:
            return preds_df, upper_ci, lower_ci

        return preds_df


def _align_leaf_target(neighs, target):
    # getting times and events for each leaf element
    target_neighs = neighs.apply(lambda x: target[x])

    # converting to vectorized kaplan format
    # filling nas due to different leaf sizes with 0
    target_neighs = (
        pd.concat([pd.DataFrame(e) for e in target_neighs.values], axis=1)
        .T.fillna(0)
        .values
    )

    return target_neighs


# class to turn XGB into a kNN with a kaplan meier in the NNs
class XGBSEKaplanTree(XGBSEBaseEstimator):
    """
    Single tree implementation as a simplification to `XGBSEKaplanNeighbors`.
    Instead of doing nearest neighbor searches, fits a single tree via `xgboost`
    and calculates KM curves at each of its leaves.

    !!! Note
        * It is by far the most efficient implementation, able to scale to millions of examples easily.
        At fit time, the tree is built and all KM curves are pre-calculated,
        so that at scoring time a simple query will suffice to get the model's estimates.

    Read more in [How XGBSE works](https://loft-br.github.io/xgboost-survival-embeddings/how_xgbse_works.html).
    """

    def __init__(
        self,
        xgb_params=None,
    ):
        """
        Args:
            xgb_params (Dict): Parameters for XGBoost model.
                If not passed, the following default parameters will be used:

                ```
                DEFAULT_PARAMS_TREE = {
                    "objective": "survival:cox",
                    "eval_metric": "cox-nloglik",
                    "tree_method": "exact",
                    "max_depth": 100,
                    "booster": "dart",
                    "subsample": 1.0,
                    "min_child_weight": 30,
                    "colsample_bynode": 1.0,
                }
                ```

                Check <https://xgboost.readthedocs.io/en/latest/parameter.html> for more options.
        """
        if xgb_params is None:
            xgb_params = DEFAULT_PARAMS_TREE

        self.xgb_params = xgb_params
        self.persist_train = False
        self.index_id = None
        self.feature_importances_ = None

    def fit(
        self,
        X,
        y,
        persist_train=True,
        index_id=None,
        time_bins=None,
        ci_width=0.683,
        **xgb_kwargs,
    ):
        """
        Fit a single decision tree using xgboost. For each leaf in the tree,
        build a Kaplan-Meier estimator.

        !!! Note
            * Differently from `XGBSEKaplanNeighbors`, in `XGBSEKaplanTree`, the width of
            the confidence interval (`ci_width`) must be specified at fit time.

        Args:

            X ([pd.DataFrame, np.array]): Design matrix to fit XGBoost model

            y (structured array(numpy.bool_, numpy.number)): Binary event indicator as first field,
                and time of event or time of censoring as second field.

            persist_train (Bool): Whether or not to persist training data to use explainability
                through prototypes

            index_id (pd.Index): User defined index if intended to use explainability
                through prototypes

            time_bins (np.array): Specified time windows to use when making survival predictions

            ci_width (Float): Width of confidence interval

        Returns:
            XGBSEKaplanTree: Trained instance of XGBSEKaplanTree
        """

        E_train, T_train = convert_y(y)
        if time_bins is None:
            time_bins = get_time_bins(T_train, E_train)
        self.time_bins = time_bins

        # converting data to xgb format
        dtrain = convert_data_to_xgb_format(X, y, self.xgb_params["objective"])

        # training XGB
        self.bst = xgb.train(self.xgb_params, dtrain, num_boost_round=1, **xgb_kwargs)
        self.feature_importances_ = self.bst.get_score()

        # getting leaves
        leaves = self.bst.predict(
            dtrain, pred_leaf=True, iteration_range=(0, self.bst.best_iteration + 1)
        )

        # organizing elements per leaf
        leaf_neighs = (
            pd.DataFrame({"leaf": leaves})
            .groupby("leaf")
            .apply(lambda x: list(x.index))
        )

        # getting T and E for each leaf
        T_leaves = _align_leaf_target(leaf_neighs, T_train)
        E_leaves = _align_leaf_target(leaf_neighs, E_train)

        # calculating z-score from width
        z = st.norm.ppf(0.5 + ci_width / 2)

        # vectorized (very fast!) implementation of Kaplan Meier curves
        (
            self._train_survival,
            self._train_upper_ci,
            self._train_lower_ci,
        ) = calculate_kaplan_vectorized(T_leaves, E_leaves, time_bins, z)

        # adding leaf indexes
        self._train_survival = self._train_survival.set_index(leaf_neighs.index)
        self._train_upper_ci = self._train_upper_ci.set_index(leaf_neighs.index)
        self._train_lower_ci = self._train_lower_ci.set_index(leaf_neighs.index)

        if persist_train:
            self.persist_train = True
            if index_id is None:
                index_id = X.index.copy()
            self.tree = BallTree(leaves.reshape(-1, 1), metric="hamming", leaf_size=40)
        self.index_id = index_id

        return self

    def predict(self, X, return_ci=False, return_interval_probs=False):
        """
        Run samples through tree until terminal nodes. Predict the Kaplan-Meier
        estimator associated to the leaf node each sample ended into.

        Args:
            X (pd.DataFrame): Data frame with samples to generate predictions

            return_ci (Bool): Whether to return confidence intervals via the Exponential Greenwood formula

            return_interval_probs (Bool): Boolean indicating if interval probabilities are
                supposed to be returned. If False the cumulative survival is returned.


        Returns:
            preds_df (pd.DataFrame): A dataframe of survival probabilities
                for all times (columns), from a time_bins array, for all samples of X
                (rows). If return_interval_probs is True, the interval probabilities are returned
                instead of the cumulative survival probabilities.

            upper_ci (np.array): Upper confidence interval for the survival
                probability values

            lower_ci (np.array): Lower confidence interval for the survival
                probability values
        """

        # converting to xgb format
        d_matrix = xgb.DMatrix(X)

        # getting leaves and extracting neighbors
        leaves = self.bst.predict(
            d_matrix, pred_leaf=True, iteration_range=(0, self.bst.best_iteration + 1)
        )

        # searching for kaplan meier curves in leaves
        preds_df = self._train_survival.loc[leaves].reset_index(drop=True)
        upper_ci = self._train_upper_ci.loc[leaves].reset_index(drop=True)
        lower_ci = self._train_lower_ci.loc[leaves].reset_index(drop=True)

        if return_ci and return_interval_probs:
            raise ValueError(
                "Confidence intervals for interval probabilities is not supported. Choose between return_ci and return_interval_probs."
            )

        if return_interval_probs:
            preds_df = calculate_interval_failures(preds_df)
            return preds_df

        if return_ci:
            return preds_df, upper_ci, lower_ci
        return preds_df
