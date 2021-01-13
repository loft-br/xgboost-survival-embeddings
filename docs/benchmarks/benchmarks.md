### **Benchmarks**

In the examples folder you'll find benchmarks comparing `xgbse` to other survival analysis methods. We show 6 metrics (see [9] for details):

* `c-index`: concordance index. Equivalent to AUC with censored data.
* `dcal_max_dev`: maximum decile deviation from calibrated distribution. 
* `dcal_pval`: p-value from chi-square test checking for D-Calibration. If larger than 0.05 then the model is D-Calibrated.
* `ibs`: approximate integrated brier score, the average brier score across all time windows.
* `inference_time`: time to perform inference.
* `training_time`: time to perform training.

We executed all methods with default parameters, except for `num_boosting_rounds`, which was set to `10` for vanilla XGBoost, and `1000` for `xgbse`, as vanilla XGBoost was overfitting with the same setting as `xgbse`. No early stopping was used for XGBoost. We show results below for five datasets. 

#### [FLCHAIN](https://github.com/vincentarelbundock/Rdatasets)

| model                       |   c-index |   dcal_max_dev |   dcal_pval |     ibs |   inference_time |   training_time |
|:----------------------------|----------:|---------------:|------------:|--------:|-----------------:|----------------:|
| Weibull AFT                 |     0.789 |          0.013 |       0.849 |   0.099 |            0.01  |           0.84  |
| Cox-PH                      |     0.788 |          0.011 |       0.971 |   0.099 |            0.007 |           1.192 |
| XGBSE - Debiased BCE        |     0.784 |          0.03  |       0.036 |   0.101 |            0.47  |          46.155 |
| XGB - AFT                   |     0.782 |        nan     |     nan     | nan     |            0.001 |           0.065 |
| XGBSE - Bootstrap Trees     |     0.781 |          0.009 |       0.985 |   0.1   |            0.425 |          17.498 |
| XGB - Cox                   |     0.779 |        nan     |     nan     | nan     |            0.001 |           0.085 |
| XGBSE - Kaplan Neighbors    |     0.769 |          0.02  |       0.732 |   0.103 |            5.807 |          31.366 |
| XGBSE - Kaplan Tree         |     0.768 |          0.011 |       0.929 |   0.103 |            0.003 |           0.212 |
| Conditional Survival Forest |     0.761 |          0.03  |       0.031 |   0.106 |          109.553 |           1.103 |

#### [METABRIC](https://github.com/jaredleekatzman/DeepSurv/tree/master/experiments/data)

| model                       |   c-index |   dcal_max_dev |   dcal_pval |     ibs |   inference_time |   training_time |
|:----------------------------|----------:|---------------:|------------:|--------:|-----------------:|----------------:|
| XGBSE - Debiased BCE        |     0.632 |          0.032 |       0.381 |   0.157 |            0.369 |          14.198 |
| XGBSE - Kaplan Neighbors    |     0.627 |          0.024 |       0.525 |   0.156 |            0.791 |          25.679 |
| XGBSE - Bootstrap Trees     |     0.624 |          0.024 |       0.563 |   0.155 |            0.466 |          11.956 |
| Conditional Survival Forest |     0.623 |          0.032 |       0.289 |   0.152 |           31.874 |           0.201 |
| Weibull AFT                 |     0.622 |          0.024 |       0.667 |   0.154 |            0.008 |           0.39  |
| Cox-PH                      |     0.622 |          0.026 |       0.567 |   0.154 |            0.004 |           0.217 |
| XGB - Cox                   |     0.619 |        nan     |     nan     | nan     |            0.001 |           0.023 |
| XGB - AFT                   |     0.61  |        nan     |     nan     | nan     |            0.001 |           0.024 |
| XGBSE - Kaplan Tree         |     0.59  |          0.036 |       0.18  |   0.165 |            0.014 |           0.114 |

#### [RRNLNPH](https://github.com/havakv/pycox/blob/master/pycox/simulations/relative_risk.py)

| model                       |   c-index |   dcal_max_dev |   dcal_pval |     ibs |   inference_time |   training_time |
|:----------------------------|----------:|---------------:|------------:|--------:|-----------------:|----------------:|
| XGBSE - Debiased BCE        |     0.827 |          0.044 |           0 |   0.098 |            0.982 |         128.98  |
| XGBSE - Bootstrap Trees     |     0.826 |          0.035 |           0 |   0.097 |            0.419 |          40.945 |
| XGB - Cox                   |     0.826 |        nan     |         nan | nan     |            0.001 |           0.056 |
| XGB - AFT                   |     0.825 |        nan     |         nan | nan     |            0.001 |           0.051 |
| XGBSE - Kaplan Neighbors    |     0.823 |          0.034 |           0 |   0.1   |           66.891 |         112.969 |
| XGBSE - Kaplan Tree         |     0.821 |          0.044 |           0 |   0.101 |            0.005 |           0.506 |
| Conditional Survival Forest |     0.811 |          0.067 |           0 |   0.113 |         5786.26  |          17.024 |
| Weibull AFT                 |     0.787 |          0.057 |           0 |   0.136 |            0.008 |           0.262 |
| Cox-PH                      |     0.787 |          0.055 |           0 |   0.135 |            0.011 |           2.362 |

#### [SAC3](https://github.com/havakv/pycox/blob/master/pycox/simulations/discrete_logit_hazard.py)

| model                       |   c-index |   dcal_max_dev |   dcal_pval |     ibs |   inference_time |   training_time |
|:----------------------------|----------:|---------------:|------------:|--------:|-----------------:|----------------:|
| XGBSE - Debiased BCE        |     0.699 |          0.038 |           0 |   0.162 |            4.067 |         689.045 |
| Cox-PH                      |     0.682 |          0.035 |           0 |   0.165 |            0.037 |           2.138 |
| Weibull AFT                 |     0.682 |          0.039 |           0 |   0.165 |            0.037 |           2.057 |
| XGBSE - Bootstrap Trees     |     0.677 |          0.043 |           0 |   0.168 |            3.134 |         220.772 |
| XGB - AFT                   |     0.671 |        nan     |         nan | nan     |            0.003 |           0.735 |
| XGB - Cox                   |     0.67  |        nan     |         nan | nan     |            0.002 |           0.913 |
| XGBSE - Kaplan Neighbors    |     0.631 |          0.038 |           0 |   0.186 |         1318.84  |         531.032 |
| XGBSE - Kaplan Tree         |     0.631 |          0.034 |           0 |   0.191 |            0.069 |           2.717 |
| Conditional Survival Forest |     0.622 |          0.044 |           0 |   0.187 |          691.939 |         837.069 |

#### [SUPPORT](https://github.com/jaredleekatzman/DeepSurv/tree/master/experiments/data)

| model                       |   c-index |   dcal_max_dev |   dcal_pval |     ibs |   inference_time |   training_time |
|:----------------------------|----------:|---------------:|------------:|--------:|-----------------:|----------------:|
| XGB - Cox                   |     0.612 |        nan     |         nan | nan     |            0.001 |           0.137 |
| XGB - AFT                   |     0.612 |        nan     |         nan | nan     |            0.001 |           0.05  |
| XGBSE - Bootstrap Trees     |     0.607 |          0.103 |           0 |   0.188 |            0.524 |          19.814 |
| XGBSE - Debiased BCE        |     0.607 |          0.119 |           0 |   0.19  |            1.221 |          62.017 |
| XGBSE - Kaplan Tree         |     0.598 |          0.097 |           0 |   0.203 |            0.005 |           0.194 |
| Conditional Survival Forest |     0.595 |          0.166 |           0 |   0.195 |          115.486 |           2.626 |
| Cox-PH                      |     0.578 |          0.16  |           0 |   0.201 |            0.01  |           0.519 |
| XGBSE - Kaplan Neighbors    |     0.578 |          0.11  |           0 |   0.202 |            8.933 |          49.236 |
| Weibull AFT                 |     0.576 |          0.138 |           0 |   0.201 |            0.009 |           0.624 |

General comments:

* `XGBSEDebiasedBCE` show the most promising results, being the best method in 3 datasets and competitive in other 2. 
* Other `xgbse` methods show good results too. In particular `XGBSEKaplanTree` with `XGBSEBootstrapEstimator` shows promising results, pointing to a direction for further research.
* Linear methods such as the **Weibull AFT** and **Cox-PH** from `lifelines` are surprisingly strong, specially for datasets with a small number of samples.
* `xgbse` methods show competitive results to vanilla `xgboost` as measured by C-index, while showing good results for "survival curve metrics". Thus, we can use `xgbse` as a calibrated replacement to vanilla `xgboost`.
* **Conditional Survival Forest** does not show consistent results. Also, it shows very high inference time, higher than training, which is unusual. It is the less efficient method of the group, followed by `XGBSEKaplanNeighbors`.
* `xgbse` takes longer to fit than vanilla `xgboost`. Specially for `XGBSEDebiasedBCE`, we have to build N logistic regressions where N is the number of time windows we'll predict. In all cases we used N = 30.
