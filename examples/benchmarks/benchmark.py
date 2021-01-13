import time

import numpy as np
import pandas as pd


from xgbse.metrics import concordance_index, approx_brier_score, dist_calibration_score
from xgbse.converters import (
    convert_data_to_xgb_format,
    convert_to_structured,
)

# setting seed
np.random.seed(42)


def dataframe_to_xy(dataf, event_column, time_column):

    e = dataf.loc[:, event_column]
    t = dataf.loc[:, time_column]
    return dataf.drop([event_column, time_column], axis=1), convert_to_structured(t, e)


class BenchmarkBase:
    def __init__(
        self,
        model,
        train_dataset,
        valid_dataset,
        test_dataset,
        event_column,
        time_column,
        time_bins,
        name,
    ):
        self.model = model
        self.train_dataset = train_dataset.dropna()
        self.valid_dataset = valid_dataset.dropna()
        self.test_dataset = test_dataset.dropna()
        self.event_column = event_column
        self.time_column = time_column
        self.time_bins = time_bins
        self.name = name

        self.X_train, self.y_train = dataframe_to_xy(
            self.train_dataset, event_column, time_column
        )
        self.X_test, self.y_test = dataframe_to_xy(
            self.test_dataset, event_column, time_column
        )

        self.survival_predictions = None
        self.hazard_predictions = None
        self.training_time = None
        self.inference_time = None

    def train(self):
        pass

    def validate(self):
        pass

    def predict(self):
        pass

    def test(self):

        self.predict()
        try:
            c_index = concordance_index(
                self.y_test, self.hazard_predictions, risk_strategy="precomputed"
            )
        except:
            c_index = np.nan
        try:
            ibs = approx_brier_score(self.y_test, self.survival_predictions)
            dcal_pval = dist_calibration_score(self.y_test, self.survival_predictions)
            dcal_max_dev = dist_calibration_score(
                self.y_test, self.survival_predictions, returns="max_deviation"
            )
        except:
            ibs = np.nan
            dcal_pval = np.nan
            dcal_max_dev = np.nan

        return {
            "model": self.name,
            "c-index": c_index,
            "ibs": ibs,
            "dcal_pval": dcal_pval,
            "dcal_max_dev": dcal_max_dev,
            "training_time": self.training_time,
            "inference_time": self.inference_time,
        }


class BenchmarkLifelines(BenchmarkBase):
    def __init__(
        self,
        model,
        train_dataset,
        valid_dataset,
        test_dataset,
        event_column,
        time_column,
        time_bins,
        name,
    ):
        super().__init__(
            model,
            train_dataset,
            valid_dataset,
            test_dataset,
            event_column,
            time_column,
            time_bins,
            name,
        )

    def train(self):
        start = time.time()
        self.model.fit(
            self.train_dataset,
            duration_col=self.time_column,
            event_col=self.event_column,
        )
        self.training_time = time.time() - start

    def predict(self):
        self.hazard_predictions = -self.model.predict_expectation(self.X_test)

        start = time.time()
        self.survival_predictions = self.model.predict_survival_function(
            self.X_test, times=self.time_bins
        ).T
        self.inference_time = time.time() - start


class BenchmarkXGBoost(BenchmarkBase):
    def __init__(
        self,
        model,
        train_dataset,
        valid_dataset,
        test_dataset,
        event_column,
        time_column,
        time_bins,
        name,
        objective,
    ):
        super().__init__(
            model,
            train_dataset,
            valid_dataset,
            test_dataset,
            event_column,
            time_column,
            time_bins,
            name,
        )
        self.objective = objective
        self.dtest = convert_data_to_xgb_format(
            self.X_test, self.y_test, self.objective
        )
        self.dtrain = convert_data_to_xgb_format(
            self.X_train, self.y_train, self.objective
        )

    def train(self):

        start = time.time()
        params = {"objective": self.objective}
        self.model = self.model.train(params, self.dtrain)
        self.training_time = time.time() - start
        return self

    def hyperopt(self):
        pass

    def predict(self):
        start = time.time()
        if self.objective == "survival:aft":
            self.hazard_predictions = pd.Series(-self.model.predict(self.dtest))
        else:
            self.hazard_predictions = pd.Series(self.model.predict(self.dtest))
        self.inference_time = time.time() - start


class BenchmarkXGBSE(BenchmarkBase):
    def __init__(
        self,
        model,
        train_dataset,
        valid_dataset,
        test_dataset,
        event_column,
        time_column,
        time_bins,
        name,
        objective,
    ):
        super().__init__(
            model,
            train_dataset,
            valid_dataset,
            test_dataset,
            event_column,
            time_column,
            time_bins,
            name,
        )
        self.objective = objective

    def train(self):

        start = time.time()
        self.model.fit(self.X_train, self.y_train, time_bins=self.time_bins)
        self.training_time = time.time() - start
        return self

    def hyperopt(self):
        pass

    def predict(self):
        start = time.time()
        self.survival_predictions = self.model.predict(self.X_test)
        self.hazard_predictions = -self.survival_predictions.mean(axis=1)
        self.inference_time = time.time() - start


class BenchmarkPysurvival(BenchmarkBase):
    def __init__(
        self,
        model,
        train_dataset,
        valid_dataset,
        test_dataset,
        event_column,
        time_column,
        time_bins,
        name,
    ):
        super().__init__(
            model,
            train_dataset,
            valid_dataset,
            test_dataset,
            event_column,
            time_column,
            time_bins,
            name,
        )

    def train(self):

        T = self.train_dataset[self.time_column]
        E = self.train_dataset[self.event_column]

        start = time.time()
        self.model.fit(self.X_train, T, E)
        self.training_time = time.time() - start
        return self

    def hyperopt(self):
        pass

    def predict(self):
        start = time.time()
        self.survival_predictions = np.empty((len(self.test_dataset,)))
        self.hazard_predictions = np.empty((len(self.test_dataset,)))
        self.survival_predictions = np.column_stack(
            [self.model.predict_survival(self.X_test, t=t) for t in self.time_bins]
        )
        self.survival_predictions = pd.DataFrame(
            self.survival_predictions, columns=self.time_bins
        )
        self.hazard_predictions = self.model.predict_risk(self.X_test)

        self.inference_time = time.time() - start
