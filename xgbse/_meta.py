import pandas as pd
from copy import deepcopy
from sklearn.utils import resample
from sklearn.base import BaseEstimator


class XGBSEBootstrapEstimator(BaseEstimator):

    """
    Bootstrap meta-estimator for XGBSE models:

    *  allows for confidence interval estimation for `XGBSEDebiasedBCE` and `XGBSEStackedWeibull`
    *  provides variance stabilization for all models, specially for `XGBSEKaplanTree`

    Performs simple bootstrap with sample size equal to training set size.

    """

    def __init__(self, base_estimator, n_estimators=10, random_state=42):
        """
        Args:
            base_estimator (XGBSEBaseEstimator): Base estimator for bootstrap procedure
            n_estimators (int): Number of estimators to fit in bootstrap procedure
            random_state (int): Random state for resampling function
        """
        self.base_estimator = base_estimator
        self.n_estimators = n_estimators
        self.random_state = random_state

    def fit(self, X, y, **kwargs):

        """
        Fit several (base) estimators and store them.

        Args:
            X ([pd.DataFrame, np.array]): Features to be used while fitting
                XGBoost model

            y (structured array(numpy.bool_, numpy.number)): Binary event indicator as first field,
                and time of event or time of censoring as second field.

            **kwargs : Keyword arguments to be passed to .fit() method of base_estimator

        Returns:
            XGBSEBootstrapEstimator: Trained instance of XGBSEBootstrapEstimator

        """

        # initializing list of estimators
        self.estimators_ = []

        # loop for n_estimators
        for i in range(self.n_estimators):

            X_sample, y_sample = resample(X, y, random_state=i + self.random_state)

            trained_model = self.base_estimator.fit(X_sample, y_sample, **kwargs)

            self.estimators_.append(deepcopy(trained_model))

        return self

    def predict(self, X, return_ci=False, ci_width=0.683, return_interval_probs=False):

        """
        Predicts survival as given by the base estimator. A survival function, its upper and lower
        confidence intervals can be returned for each sample of the dataframe X.

        Args:
            X (pd.DataFrame): data frame with samples to generate predictions

            return_ci (Bool): whether to include confidence intervals

            ci_width (Float): width of confidence interval

        Returns:
            ([(pd.DataFrame, np.array, np.array), pd.DataFrame]):
            preds_df: A dataframe of survival probabilities
            for all times (columns), from a time_bins array, for all samples of X
            (rows). If return_interval_probs is True, the interval probabilities are returned
            instead of the cumulative survival probabilities.

            upper_ci: Upper confidence interval for the survival
                probability values

            lower_ci: Lower confidence interval for the survival
                probability values
        """

        preds_list = []

        for estimator in self.estimators_:

            temp_preds = estimator.predict(
                X, return_interval_probs=return_interval_probs
            )
            preds_list.append(temp_preds)

        agg_preds = pd.concat(preds_list)

        preds_df = agg_preds.groupby(level=0).mean()

        if return_ci:

            low_p = 0.5 - ci_width / 2
            high_p = 0.5 + ci_width / 2

            lower_ci = agg_preds.groupby(level=0).quantile(low_p)
            upper_ci = agg_preds.groupby(level=0).quantile(high_p)

            return preds_df, upper_ci, lower_ci

        return preds_df
