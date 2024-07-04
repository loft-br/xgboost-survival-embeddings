from typing import Any, Dict, List, Optional, Tuple

import numpy.typing as npt
import xgboost as xgb

from xgbse.converters import convert_data_to_xgb_format, convert_y
from xgbse.non_parametric import get_time_bins


class FeatureExtractor:
    def __init__(
        self,
        xgb_params: Optional[Dict[str, Any]] = None,
        enable_categorical: bool = False,
    ):
        """
        Args:
        xgb_params (Dict, None): Parameters for XGBoost model.
            If None, will use XGBoost defaults and set objective as `survival:aft`.
            Check <https://xgboost.readthedocs.io/en/latest/parameter.html> for options.

        """
        if not xgb_params:
            xgb_params = {}
        xgb_params = check_xgboost_parameters(xgb_params, enable_categorical)

        self.xgb_params = xgb_params
        self.persist_train = False
        self.feature_importances_ = None
        self.enable_categorical = enable_categorical

    def fit(
        self,
        X,
        y,
        time_bins: Optional[npt.ArrayLike] = None,
        validation_data: Optional[List[Tuple[Any, Any]]] = None,
        num_boost_round: int = 10,
        early_stopping_rounds: Optional[int] = None,
        verbose_eval: int = 0,
    ):
        """
                Transform feature space by fitting a XGBoost model and returning its leaf indices.
                Leaves are transformed and considered as dummy variables to fit multiple logistic
                regression models to each evaluated time bin.
        //
                Args:
                    X ([pd.DataFrame, np.array]): Features to be used while fitting XGBoost model

                    y (structured array(numpy.bool_, numpy.number)): Binary event indicator as first field,
                        and time of event or time of censoring as second field.

                    time_bins (np.array): Specified time windows to use when making survival predictions

                    validation_data (Tuple): Validation data in the format of a list of tuples [(X, y)]
                        if user desires to use early stopping

                    num_boost_round (Int): Number of boosting iterations, defaults to 10

                    early_stopping_rounds (Int): Activates early stopping.
                        Validation metric needs to improve at least once
                        in every **early_stopping_rounds** round(s) to continue training.
                        See xgboost.train documentation.

                    persist_train (Bool): Whether or not to persist training data to use explainability
                        through prototypes

                    index_id (pd.Index): User defined index if intended to use explainability
                        through prototypes


                    verbose_eval ([Bool, Int]): Level of verbosity. See xgboost.train documentation.

                Returns:
                    XGBSEDebiasedBCE: Trained XGBSEDebiasedBCE instance
        """

        E_train, T_train = convert_y(y)
        if time_bins is None:
            time_bins = get_time_bins(T_train, E_train)
        self.time_bins = time_bins

        # converting data to xgb format
        dtrain = convert_data_to_xgb_format(
            X, y, self.xgb_params["objective"], self.enable_categorical
        )

        # converting validation data to xgb format
        evals = ()
        if validation_data:
            X_val, y_val = validation_data
            dvalid = convert_data_to_xgb_format(
                X_val, y_val, self.xgb_params["objective"], self.enable_categorical
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

    def predict_leaves(self, X):
        """
        Predict leaf indices of XGBoost model.

        Args:
            X (pd.DataFrame, np.array): Features to be used while predicting leaf indices

        Returns:
            np.array: Leaf indices of XGBoost model
        """
        if not hasattr(self, "bst"):
            raise ValueError("XGBoost model not fitted yet.")

        dmatrix = xgb.DMatrix(X, enable_categorical=self.enable_categorical)
        return self.bst.predict(dmatrix, pred_leaf=True)

    def predict_hazard(self, X):
        if not hasattr(self, "bst"):
            raise ValueError("XGBoost model not fitted yet.")

        return self.bst.predict(
            xgb.DMatrix(X, enable_categorical=self.enable_categorical)
        )


def check_xgboost_parameters(
    xgb_params: Dict[str, Any], enable_categorical: bool
) -> Dict[str, Any]:
    """Check if XGBoost objective parameter is valid.

    Args:
        xgb_params (Dict): Parameters for XGBoost model.

    Returns:
        xgb_params (Dict): Parameters for XGBoost model.

    Raises:
        ValueError: If XGBoost parameters are not valid for survival analysis.
    """
    if enable_categorical:
        if "tree_method" not in xgb_params:
            xgb_params["tree_method"] = "hist"
        if xgb_params["tree_method"] not in ("hist", "gpu_hist"):
            raise ValueError(
                "XGBoost tree_method must be either 'hist' or 'gpu_hist' for categorical features"
            )

    if "objective" not in xgb_params:
        xgb_params["objective"] = "survival:aft"
    if xgb_params["objective"] not in ("survival:aft", "survival:cox"):
        raise ValueError(
            "XGBoost objective must be either 'survival:aft' or 'survival:cox'"
        )

    return xgb_params
