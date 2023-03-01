from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from lifelines import WeibullAFTFitter
from sklearn.neighbors import BallTree

from xgbse._base import XGBSEBaseEstimator
from xgbse._feature_extractors import FeatureExtractor
from xgbse.converters import convert_y
from xgbse.non_parametric import calculate_interval_failures

KM_PERCENTILES = np.linspace(0, 1, 11)


class XGBSEStackedWeibull(XGBSEBaseEstimator):
    """
    Perform stacking of a XGBoost survival model with a Weibull AFT parametric model.
    The XGBoost fits the data and then predicts a value that is interpreted as a risk metric.
    This risk metric is fed to the Weibull regression which uses it as its only independent variable.

    Thus, we can get the benefit of XGBoost discrimination power alongside the Weibull AFT
    statistical rigor (e.g. calibrated survival curves).

    !!! Note
        * As we're stacking XGBoost with a single, one-variable parametric model
        (as opposed to `XGBSEDebiasedBCE`), the model can be much faster (especially in training).
        * We also have better extrapolation capabilities, as opposed to the cure fraction
        problem in `XGBSEKaplanNeighbors` and `XGBSEKaplanTree`.
        * However, we also have stronger assumptions about the shape of the survival curve.

    Read more in [How XGBSE works](https://loft-br.github.io/xgboost-survival-embeddings/how_xgbse_works.html).

    """

    def __init__(
        self,
        xgb_params: Optional[Dict[str, Any]] = None,
        weibull_params: Optional[Dict[str, Any]] = {},
        enable_categorical: bool = False,
    ):
        """
        Args:
            xgb_params (Dict, None): Parameters for XGBoost model.
                If None, will use XGBoost defaults and set objective as `survival:aft`.
                Check <https://xgboost.readthedocs.io/en/latest/parameter.html> for options.

            weibull_params (Dict): Parameters for Weibull Regerssion model.
                If not passed, will use the default parameters as shown in the Lifelines documentation.
                Check <https://lifelines.readthedocs.io/en/latest/fitters/regression/WeibullAFTFitter.html>
                for more options.

            enable_categorical (bool): Enable categorical feature support on xgboost model

        """
        self.feature_extractor = FeatureExtractor(
            xgb_params=xgb_params, enable_categorical=enable_categorical
        )
        self.xgb_params = self.feature_extractor.xgb_params
        self.weibull_params = weibull_params

        self.persist_train = False
        self.feature_importances_ = None

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
        Fit XGBoost model to predict a value that is interpreted as a risk metric.
        Fit Weibull Regression model using risk metric as only independent variable.

        Args:
            X ([pd.DataFrame, np.array]): Features to be used while fitting XGBoost model

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
            XGBSEStackedWeibull: Trained XGBSEStackedWeibull instance
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

        # predicting hazard ratio from XGBoost
        train_risk = self.feature_extractor.predict_hazard(X)

        # replacing 0 by minimum positive value in df
        # so Weibull can be fitted
        min_positive_value = T_train[T_train > 0].min()
        T_train = np.clip(T_train, min_positive_value, None)

        # creating df to use lifelines API
        weibull_train_df = pd.DataFrame(
            {"risk": train_risk, "duration": T_train, "event": E_train}
        )

        # fitting weibull aft
        self.weibull_aft = WeibullAFTFitter(**self.weibull_params)
        self.weibull_aft.fit(weibull_train_df, "duration", "event", ancillary=True)

        if persist_train:
            self.persist_train = True
            if index_id is None:
                index_id = X.index.copy()

            index_leaves = self.feature_extractor.predict_leaves(X)
            self.tree = BallTree(index_leaves, metric="hamming")

        self.index_id = index_id

        return self

    def predict(self, X, return_interval_probs=False):
        """
        Predicts survival probabilities using the XGBoost + Weibull AFT stacking pipeline.

        Args:
            X (pd.DataFrame): Dataframe of features to be used as input for the
                XGBoost model.

            return_interval_probs (Bool): Boolean indicating if interval probabilities are
                supposed to be returned. If False the cumulative survival is returned.
                Default is False.

        Returns:
            pd.DataFrame: A dataframe of survival probabilities
            for all times (columns), from a time_bins array, for all samples of X
            (rows). If return_interval_probs is True, the interval probabilities are returned
            instead of the cumulative survival probabilities.
        """
        risk = self.feature_extractor.predict_hazard(X)
        weibull_score_df = pd.DataFrame({"risk": risk})

        preds_df = self.weibull_aft.predict_survival_function(
            weibull_score_df, self.time_bins
        ).T

        if return_interval_probs:
            preds_df = calculate_interval_failures(preds_df)

        return preds_df
