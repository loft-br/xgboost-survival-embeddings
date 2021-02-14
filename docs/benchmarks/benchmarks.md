---
hide:
  - navigation
---

### Metrics

In the examples folder you'll find benchmarks comparing `xgbse` to other survival analysis methods. We show 6 metrics (see [9] for details):

* `c-index`: concordance index. Equivalent to AUC with censored data.
* `dcal_max_dev`: maximum decile deviation from calibrated distribution.
* `dcal_pval`: p-value from chi-square test checking for D-Calibration. If larger than 0.05 then the model is D-Calibrated.
* `ibs`: approximate integrated brier score, the average brier score across all time windows.
* `inference_time`: time to perform inference, recorded on a 2018 MacBook Pro.
* `training_time`: time to perform training, recorded on a 2018 MacBook Pro.

We executed all methods with default parameters. For vanilla XGBoost and `xgbse`, early stopping was used, with `num_boosting_rounds=1000`, and  `early_stopping_rounds=10`. We show results below for five datasets.

### Results

#### [FLCHAIN](https://github.com/vincentarelbundock/Rdatasets)

| model                    |   c-index |   dcal_max_dev |   dcal_pval |     ibs |   inference_time |   training_time |
|:-------------------------|----------:|---------------:|------------:|--------:|-----------------:|----------------:|
| Weibull AFT              |     0.789 |          0.013 |       0.849 |   0.099 |            0.006 |           0.537 |
| Cox-PH                   |     0.788 |          0.011 |       0.971 |   0.099 |            0.005 |           0.942 |
| XGBSE - Debiased BCE     |     0.784 |          0.037 |       0     |   0.117 |            0.233 |           3.062 |
| XGBSE - Bootstrap Trees  |     0.781 |          0.009 |       0.985 |   0.1   |            0.382 |          15.351 |
| XGBSE - Kaplan Neighbors |     0.777 |          0.013 |       0.918 |   0.102 |            0.543 |           0.479 |
| XGBSE - Stacked Weibull  |     0.776 |          0.008 |       0.994 |   0.103 |            0.011 |           0.719 |
| XGB - Cox                |     0.775 |        nan     |     nan     | nan     |            0.001 |           0.054 |
| XGB - AFT                |     0.772 |        nan     |     nan     | nan     |            0.001 |           0.106 |
| XGBSE - Kaplan Tree      |     0.768 |          0.011 |       0.929 |   0.103 |            0.003 |           0.167 |

#### [METABRIC](https://github.com/jaredleekatzman/DeepSurv/tree/master/experiments/data)

| model                    |   c-index |   dcal_max_dev |   dcal_pval |     ibs |   inference_time |   training_time |
|:-------------------------|----------:|---------------:|------------:|--------:|-----------------:|----------------:|
| XGBSE - Stacked Weibull  |     0.63  |          0.045 |       0.146 |   0.162 |            0.01  |           0.525 |
| XGBSE - Debiased BCE     |     0.627 |          0.033 |       0.128 |   0.165 |            0.09  |           3.165 |
| XGBSE - Bootstrap Trees  |     0.624 |          0.024 |       0.563 |   0.155 |            0.301 |           6.165 |
| Weibull AFT              |     0.622 |          0.024 |       0.667 |   0.154 |            0.005 |           0.284 |
| Cox-PH                   |     0.622 |          0.026 |       0.567 |   0.154 |            0.004 |           0.244 |
| XGB - Cox                |     0.617 |        nan     |     nan     | nan     |            0.001 |           0.096 |
| XGBSE - Kaplan Neighbors |     0.605 |          0.023 |       0.588 |   0.163 |            0.111 |           0.154 |
| XGB - AFT                |     0.6   |        nan     |     nan     | nan     |            0.001 |           0.044 |
| XGBSE - Kaplan Tree      |     0.59  |          0.036 |       0.18  |   0.165 |            0.002 |           0.05  |

#### [RRNLNPH](https://github.com/havakv/pycox/blob/master/pycox/simulations/relative_risk.py)

| model                    |   c-index |   dcal_max_dev |   dcal_pval |     ibs |   inference_time |   training_time |
|:-------------------------|----------:|---------------:|------------:|--------:|-----------------:|----------------:|
| XGBSE - Stacked Weibull  |     0.826 |          0.05  |           0 |   0.113 |            0.019 |           2.255 |
| XGBSE - Bootstrap Trees  |     0.826 |          0.035 |           0 |   0.097 |            0.534 |          44.736 |
| XGBSE - Kaplan Neighbors |     0.824 |          0.038 |           0 |   0.1   |           15.662 |           1.504 |
| XGBSE - Debiased BCE     |     0.824 |          0.068 |           0 |   0.108 |            0.285 |           4.562 |
| XGB - Cox                |     0.824 |        nan     |         nan | nan     |            0.002 |           0.375 |
| XGB - AFT                |     0.823 |        nan     |         nan | nan     |            0.001 |           0.243 |
| XGBSE - Kaplan Tree      |     0.821 |          0.044 |           0 |   0.101 |            0.006 |           0.49  |
| Weibull AFT              |     0.787 |          0.057 |           0 |   0.136 |            0.01  |           0.326 |
| Cox-PH                   |     0.787 |          0.055 |           0 |   0.135 |            0.021 |           2.267 |

#### [SAC3](https://github.com/havakv/pycox/blob/master/pycox/simulations/discrete_logit_hazard.py)

| model                    |   c-index |   dcal_max_dev |   dcal_pval |     ibs |   inference_time |   training_time |
|:-------------------------|----------:|---------------:|------------:|--------:|-----------------:|----------------:|
| XGBSE - Stacked Weibull  |     0.697 |          0.04  |           0 |   0.171 |            0.153 |          32.469 |
| XGB - AFT                |     0.691 |        nan     |         nan | nan     |            0.004 |           7.413 |
| XGBSE - Debiased BCE     |     0.69  |          0.045 |           0 |   0.169 |            1.141 |          47.814 |
| XGB - Cox                |     0.686 |        nan     |         nan | nan     |            0.002 |           4.885 |
| Cox-PH                   |     0.682 |          0.035 |           0 |   0.165 |            0.039 |           1.84  |
| Weibull AFT              |     0.682 |          0.039 |           0 |   0.165 |            0.043 |           2.307 |
| XGBSE - Bootstrap Trees  |     0.677 |          0.043 |           0 |   0.168 |            3.134 |         164.173 |
| XGBSE - Kaplan Neighbors |     0.666 |          0.037 |           0 |   0.175 |          416.382 |          36.759 |
| XGBSE - Kaplan Tree      |     0.631 |          0.034 |           0 |   0.191 |            0.036 |           1.478 |

#### [SUPPORT](https://github.com/jaredleekatzman/DeepSurv/tree/master/experiments/data)

| model                    |   c-index |   dcal_max_dev |   dcal_pval |     ibs |   inference_time |   training_time |
|:-------------------------|----------:|---------------:|------------:|--------:|-----------------:|----------------:|
| XGBSE - Stacked Weibull  |     0.621 |          0.092 |           0 |   0.198 |            0.013 |           0.954 |
| XGBSE - Debiased BCE     |     0.617 |          0.139 |           0 |   0.188 |            0.272 |           3.852 |
| XGB - Cox                |     0.61  |        nan     |         nan | nan     |            0.001 |           0.069 |
| XGB - AFT                |     0.609 |        nan     |         nan | nan     |            0.001 |           0.137 |
| XGBSE - Bootstrap Trees  |     0.607 |          0.103 |           0 |   0.188 |            0.371 |          18.202 |
| XGBSE - Kaplan Neighbors |     0.601 |          0.099 |           0 |   0.197 |            1.488 |           0.752 |
| XGBSE - Kaplan Tree      |     0.598 |          0.097 |           0 |   0.203 |            0.004 |           0.149 |
| Cox-PH                   |     0.578 |          0.16  |           0 |   0.201 |            0.006 |           0.465 |
| Weibull AFT              |     0.576 |          0.138 |           0 |   0.201 |            0.007 |           0.461 |

### Analysis

* `XGBSEDebiasedBCE` and `XGBSEStackedWeibull` show the most promising results, being in the top three methods 4 out of 5 times.
* Other `xgbse` methods show good results too. In particular `XGBSEKaplanTree` with `XGBSEBootstrapEstimator` shows promising results, pointing to a direction for further research.
* Linear methods such as the **Weibull AFT** and **Cox-PH** from `lifelines` are surprisingly strong, specially for datasets with a small number of samples.
* `xgbse` methods show competitive results to vanilla `xgboost` as measured by C-index, while showing good results for "survival curve metrics". Thus, we can use `xgbse` as a calibrated replacement to vanilla `xgboost`.
* `xgbse` takes longer to fit than vanilla `xgboost`. Specially for `XGBSEDebiasedBCE`, we have to build N logistic regressions where N is the number of time windows we'll predict. In all cases we used N = 30. `XGBSEStackedWeibull` is the most efficient method, behind `XGBSEKaplanTree`.
