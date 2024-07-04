import warnings
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import numpy.typing as npt
import pandas as pd
from joblib import Parallel, delayed
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import BallTree
from sklearn.preprocessing import OneHotEncoder

from xgbse._base import DummyLogisticRegression, XGBSEBaseEstimator
from xgbse.converters import convert_y, hazard_to_survival
from xgbse.non_parametric import calculate_interval_failures

KM_PERCENTILES = np.linspace(0, 1, 11)


def _repeat_array(x, n):
    """
    Repeats an array x n times. Resulting array of ((x*n) x 1) shape.

    Args:
        x (np.array): An array to be repeated
        n (Int): Number of times to repeat array x

    Returns:
        (np.array): Array x repeated n times.
    """

    return np.array([x] * n).T


def _build_multi_task_targets(
    E: npt.NDArray[np.bool_], T: npt.NDArray[np.int64], time_bins: npt.NDArray[np.int64]
):
    """
    Builds targets for a multi task survival regression problem.
    This function creates a times array from time 0 to T, where T is the
    event/censor last observed time. If time_bins > T, times greater than the last observed
    time T are considered equal to -1.

    Args:
        E ([np.array, pd.Series]): Array of censors(0)/events(1).

        T ([np.array, pd.Series]): Array of times.
        time_bins ([np.array]): Specified time bins to split targets.

    Returns:
        targets (pd.Series): A Series with multi task targets (for data existent
             just up to time T=t, all times over t are considered equal to -1).
        time_bins (np.array): Time bins to be used for multi task survival analysis.
    """

    events = _repeat_array(E, len(time_bins))
    times = _repeat_array(T, len(time_bins)) < time_bins

    targets = times.astype(int)
    shifted_array = np.roll(targets, 1)
    shifted_array[:, 0] = 0
    shifted_array = shifted_array + targets
    shifted_array[shifted_array == 2] = -1

    shifted_array[np.logical_not(events) & times] = -1

    return shifted_array, time_bins


# class to fit a BCE on the leaves of a XGB
class XGBSEDebiasedBCE(XGBSEBaseEstimator):
    """
    Train a set of logistic regressions on top of leaf embeddings produced by XGBoost,
    each predicting survival at different user-defined discrete time windows.
    The classifiers remove individuals as they are censored,
     with targets that are indicatorsof surviving at each window.

    !!! Note
        * Training and scoring of logistic regression models is efficient,
        being performed in parallel through joblib, so the model can scale to
        hundreds of thousands or millions of samples.
        * However, if many windows are used and data is large, training of
        logistic regression models may become a bottleneck, taking more time
        than training of the underlying XGBoost model.

    Read more in [How XGBSE works](https://loft-br.github.io/xgboost-survival-embeddings/how_xgbse_works.html).

    """

    def __init__(
        self,
        xgb_params: Optional[Dict[str, Any]] = None,
        lr_params: Dict[str, Any] = {},
        n_jobs: int = 1,
        enable_categorical: bool = False,
    ):
        """
        Args:
            xgb_params (Dict, None): Parameters for XGBoost model.
                If None, will use XGBoost defaults and set objective as `survival:aft`.
                Check <https://xgboost.readthedocs.io/en/latest/parameter.html> for options.

            lr_params (Dict, None): Parameters for LogisticRegression model.
                If None, will use scikit-learn default parameters.

            n_jobs (int): Number of jobs used for parallel training of logistic regressions.

            enable_categorical (bool): Enable categorical feature support on xgboost model
        """
        super().__init__(xgb_params=xgb_params, enable_categorical=enable_categorical)
        self.lr_params = lr_params
        self.n_jobs = n_jobs

    def fit(
        self,
        X,
        y,
        time_bins: Optional[Sequence] = None,
        validation_data: Optional[List[Tuple[Any, Any]]] = None,
        num_boost_round: int = 10,
        early_stopping_rounds: Optional[int] = None,
        verbose_eval: int = 0,
        persist_train: bool = False,
        index_id=None,
    ):
        """
        Transform feature space by fitting a XGBoost model and returning its leafs.
        Leaves are transformed and considered as dummy variables to fit multiple
        logistic regression models to each evaluated time bin.

        Args:
            X ([pd.DataFrame, np.array]): Features to be used while fitting XGBoost model

            y (structured array(numpy.bool_, numpy.number)): Binary event indicator as first field,
                and time of event or time of censoring as second field.

            time_bins (np.array): Specified time windows to use when making survival predictions

            validation_data (List[Tuple]): Validation data in the format of a list of tuples [(X, y)]
                if user desires to use early stopping

            num_boost_round (Int): Number of boosting iterations.

            early_stopping_rounds (Int): Activates early stopping.
                Validation metric needs to improve at least once
                in every **early_stopping_rounds** round(s) to continue training.
                See xgboost.train documentation.

            verbose_eval ([Bool, Int]): Level of verbosity. See xgboost.train documentation.

            persist_train (Bool): Whether or not to persist training data to use explainability
                through prototypes

            index_id (pd.Index): User defined index if intended to use explainability
                through prototypes

        Returns:
            XGBSEDebiasedBCE: Trained XGBSEDebiasedBCE instance
        """
        self.fit_feature_extractor(
            X,
            y,
            time_bins=time_bins,
            validation_data=validation_data,
            num_boost_round=num_boost_round,
            early_stopping_rounds=early_stopping_rounds,
            verbose_eval=verbose_eval,
        )

        E_train, T_train = convert_y(y)
        # predicting and encoding leaves
        self.encoder = OneHotEncoder()
        leaves = self.feature_extractor.predict_leaves(X)
        leaves_encoded = self.encoder.fit_transform(leaves)

        # convert targets for using with logistic regression
        self.targets, self.time_bins = _build_multi_task_targets(
            E_train, T_train, self.feature_extractor.time_bins
        )

        # fitting LR for several targets
        self.lr_estimators_ = self._fit_all_lr(leaves_encoded, self.targets)

        if persist_train:
            self.persist_train = True
            if index_id is None:
                index_id = X.index.copy()
            self.tree = BallTree(leaves, metric="hamming")

        self.index_id = index_id

        return self

    def _fit_one_lr(self, leaves_encoded, target):
        """
        Fits a single logistic regression to predict survival probability
        at a certain time bin as target. Encoded leaves are used as features.

        Args:
            leaves_encoded (np.array): A tensor of one hot encoded leaves.

            target (np.array): An array of time targets for a specific

        Returns:
            lr (sklearn.linear_model.LogisticRegression): A fitted Logistic
            Regression model. This model outputs calibrated survival probabilities
            on a time T.
        """

        # masking
        mask = target != -1

        # by default we use a logistic regression
        classifier = LogisticRegression(**self.lr_params)

        if len(target[mask]) == 0:
            # If there's no observation in a time bucket we raise an error
            raise ValueError("Error: No observations in a time bucket")
        elif len(np.unique(target[mask])) == 1:
            # If there's only one class in a time bucket
            # we create a dummy classifier that predicts that class and send a warning
            warnings.warn(
                "Warning: Only one class found in a time bucket", RuntimeWarning
            )
            classifier = DummyLogisticRegression()

        classifier.fit(leaves_encoded[mask, :], target[mask])
        return classifier

    def _fit_all_lr(self, leaves_encoded, targets):
        """
        Fits multiple Logistic Regressions to predict survival probability
        for a list of time bins as target. Encoded leaves are used as features.

        Args:
            leaves_encoded (np.array): A tensor of one hot encoded leaves.

            targets (np.array): An array of time targets for a specific time bin.

        Returns:
            lr_estimators (List): A list of fitted Logistic Regression models.
                These models output calibrated survival probabilities for all times
                in pre specified time bins.
        """

        with Parallel(n_jobs=self.n_jobs) as parallel:
            lr_estimators = parallel(
                delayed(self._fit_one_lr)(leaves_encoded, targets[:, i])
                for i in range(targets.shape[1])
            )

        return lr_estimators

    def _predict_from_lr_list(self, lr_estimators, leaves_encoded, time_bins):
        """
        Predicts survival probabilities from a list of multiple fitted
        Logistic Regressions models. Encoded leaves are used as features.

        Args:
            lr_estimators (List): A list of fitted Logistic Regression models.
            These models output calibrated survival probabilities for all times
            in pre specified time bins.

            leaves_encoded (np.array): A tensor of one hot encoded leaves.

            time_bins (np.array): Specified time bins to split targets.

        Returns:
            preds (pd.DataFrame): A dataframe of estimated survival probabilities
                for all times (columns), from the time_bins array, for all samples
                (rows).
        """

        with Parallel(n_jobs=self.n_jobs) as parallel:
            preds = parallel(
                delayed(m.predict_proba)(leaves_encoded) for m in lr_estimators
            )

        # organizing interval predictions from LRs
        preds = np.array(preds)[:, :, 1].T
        preds = pd.DataFrame(preds, columns=time_bins)

        # converting these interval predictions
        # to cumulative survival curve
        return hazard_to_survival(preds)

    def predict(self, X: pd.DataFrame, return_interval_probs: bool = False):
        """
        Predicts survival probabilities using XGBoost + Logistic Regression pipeline.

        Args:
            X (pd.DataFrame): Dataframe of features to be used as input for the
                XGBoost model.

            return_interval_probs (Bool): Boolean indicating if interval probabilities
                are to be returned. If False the cumulative survival is returned.
                Default is False.

        Returns:
            pd.DataFrame: A dataframe of survival probabilities
            for all times (columns), from a time_bins array, for all samples of X
            (rows). If return_interval_probs is True, the interval probabilities are
            returned instead of the cumulative survival probabilities.
        """

        # getting leaves and extracting neighbors
        leaves = self.feature_extractor.predict_leaves(X)

        leaves_encoded = self.encoder.transform(leaves)

        # predicting from logistic regression artifacts

        preds_df = self._predict_from_lr_list(
            self.lr_estimators_, leaves_encoded, self.time_bins
        )

        if return_interval_probs:
            preds_df = calculate_interval_failures(preds_df)

        return preds_df
