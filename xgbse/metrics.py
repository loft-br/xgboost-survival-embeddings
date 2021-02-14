import numpy as np
import pandas as pd
from bisect import bisect_right
from scipy.stats import chisquare
from .non_parametric import calculate_kaplan_vectorized
from .converters import convert_y

# epsilon to prevent division by zero
EPS = 1e-6


def concordance_index(y_true, survival, risk_strategy="mean", which_window=None):
    """
    Compute the C-index for a structured array of ground truth times and events
    and a predicted survival curve using different strategies for estimating risk from it.

    !!! Note
        * Computation of the C-index is $\\mathcal{O}(n^2)$.

    Args:
        y_true (structured array(numpy.bool_, numpy.number)): Binary event indicator as first field,
            and time of event or time of censoring as second field.

        survival ([pd.DataFrame, np.array]): A dataframe of survival probabilities
            for all times (columns), from a time_bins array, for all samples of X (rows).
            If risk_strategy is 'precomputed', is an array with representing risks for each sample.

        risk_strategy (string):
            Strategy to compute risks from the survival curve. For a given sample:

            * `mean` averages probabilities across all times

            * `window`: lets user choose on of the time windows available (by which_window argument)
                and uses probabilities of this specific window

            * `midpoint`: selects the most central window of index int(survival.columns.shape[0]/2)
                and uses probabilities of this specific window

            * `precomputed`: assumes user has already calculated risk.
                The survival argument is assumed to contain an array of risks instead

        which_window (object): Which window to use when risk_strategy is 'window'. Should be one
            of the columns of the dataframe. Will raise ValueError if column is not present

    Returns:
        Float: Concordance index for y_true and survival
    """

    # choosing risk calculation strategy

    if risk_strategy == "mean":
        risks = 1 - survival.mean(axis=1)

    elif risk_strategy == "window":
        if which_window is None:
            raise ValueError(
                "Need to set which window to use via the which_window parameter"
            )
        risks = 1 - survival[which_window]

    elif risk_strategy == "midpoint":
        midpoint = int(survival.columns.shape[0] / 2)
        midpoint_col = survival.columns[midpoint]
        risks = 1 - survival[midpoint_col]

    elif risk_strategy == "precomputed":
        risks = survival

    else:
        raise ValueError(
            f"Chosen risk computing strategy of {risk_strategy} is not available."
        )

    # organizing event, time and risk data
    events, times = convert_y(y_true)
    events = events.astype(bool)

    cind_df = pd.DataFrame({"t": times, "e": events, "r": risks})

    count_pairs = 0
    concordant_pairs = 0
    tied_pairs = 0

    # running loop for each uncensored sample,
    # as by https://arxiv.org/pdf/1811.11347.pdf
    for _, row in cind_df.query("e == True").iterrows():

        # getting all censored and uncensored samples
        # after current row
        samples_after_i = cind_df.query(f"""{row['t']} < t""")

        # counting total, concordant and tied pairs
        count_pairs += samples_after_i.shape[0]
        concordant_pairs += (samples_after_i["r"] < row["r"]).sum()
        tied_pairs += (samples_after_i["r"] == row["r"]).sum()

    return (concordant_pairs + tied_pairs / 2) / count_pairs


def _match_times_to_windows(times, windows):

    """
    Match a list of event or censoring times to the corresponding
    time window on the survival dataframe.
    """

    matches = np.array([bisect_right(windows, e) for e in times])
    matches = np.clip(matches, 0, len(windows) - 1)
    return windows[matches]


def approx_brier_score(y_true, survival, aggregate="mean"):
    """
    Estimate brier score for all survival time windows. Aggregate scores for an approximate
    integrated brier score estimate.

    Args:
        y_true (structured array(numpy.bool_, numpy.number)): B inary event indicator as first field,
            and time of event or time of censoring as second field.

        survival ([pd.DataFrame, np.array]): A dataframe of survival probabilities
            for all times (columns), from a time_bins array, for all samples of X (rows).
            If risk_strategy is 'precomputed', is an array with representing risks for each sample.

        aggregate ([string, None]): How to aggregate brier scores from different time windows:

            * `mean` takes simple average

            * `None` returns full list of brier scores for each time window

    Returns:
        [Float, np.array]:
            single value if aggregate is 'mean'
            np.array if aggregate is None
    """
    events, times = convert_y(y_true)
    events = events.astype(bool)

    # calculating censoring distribution
    censoring_dist, _, _ = calculate_kaplan_vectorized(
        times.reshape(1, -1), ~events.reshape(1, -1), survival.columns
    )

    # initializing scoring df
    scoring_df = pd.DataFrame({"e": events, "t": times}, index=survival.index)

    # adding censoring distribution survival at event
    event_time_windows = _match_times_to_windows(times, survival.columns)
    scoring_df["cens_at_event"] = censoring_dist[event_time_windows].iloc[0].values

    # list of window results
    window_results = []

    # loop for all suvival time windows
    for window in survival.columns:

        # adding window info to scoring df
        scoring_df = scoring_df.assign(surv_at_window=survival[window]).assign(
            cens_at_window=censoring_dist[window].values[0]
        )

        # calculating censored brier score first term
        # as by formula on B4.3 of https://arxiv.org/pdf/1811.11347.pdf
        first_term = (
            (scoring_df["t"] <= window).astype(int)
            * (scoring_df["e"])
            * (scoring_df["surv_at_window"]) ** 2
            / (scoring_df["cens_at_event"])
        )

        # calculating censored brier score second term
        # as by formula on B4.3 of https://arxiv.org/pdf/1811.11347.pdf
        second_term = (
            (scoring_df["t"] > window).astype(int)
            * (1 - scoring_df["surv_at_window"]) ** 2
            / (scoring_df["cens_at_window"])
        )

        # adding and taking average
        result = (first_term + second_term).sum() / scoring_df.shape[0]
        window_results.append(result)

    if aggregate == "mean":
        return np.array(window_results).mean()
    elif aggregate is None:
        return np.array(window_results)
    else:
        raise ValueError(
            f"Chosen aggregating strategy of {aggregate} is not available."
        )


def dist_calibration_score(y_true, survival, n_bins=10, returns="pval"):
    """
    Estimate D-Calibration for the survival predictions.

    Args:
        y_true (structured array(numpy.bool_, numpy.number)): Binary event indicator as first field,
            and time of event or time of censoring as second field.

        survival ([pd.DataFrame, np.array]): A dataframe of survival probabilities
            for all times (columns), from a time_bins array, for all samples of X (rows).
            If risk_strategy is 'precomputed', is an array with representing risks for each sample.

        n_bins (Int): Number of bins to equally divide the [0, 1] interval

        returns (string):
            What information to return from the function:

            * `statistic` returns the chi squared test statistic

            * `pval` returns the chi squared test p value

            * `max_deviation` returns the maximum percentage deviation from the expected value,
            calculated as `abs(expected_percentage - real_percentage)`,
            where `expected_percentage = 1.0/n_bins`

            * `histogram` returns the full calibration histogram per bin

            * `all` returns all of the above in a dictionary

    Returns:
        [Float, np.array, Dict]:
        * Single value if returns is in `['statistic','pval','max_deviation']``
        * np.array if returns is 'histogram'
        * dict if returns is 'all'
    """

    # calculating bins
    bins = np.round(np.linspace(0, 1, n_bins + 1), 2)

    events, times = convert_y(y_true)
    events = events.astype(bool)

    # mapping event and censoring times to survival windows
    event_time_windows = _match_times_to_windows(times, survival.columns)
    survival_at_ti = np.array(
        [survival.iloc[i][event_time_windows[i]] for i in range(len(survival))]
    )
    survival_at_ti = np.clip(survival_at_ti, EPS, None)

    # creating data frame to calculate uncensored and censored counts
    scoring_df = pd.DataFrame(
        {
            "survival_at_ti": survival_at_ti,
            "t": times,
            "e": events,
            "bin": pd.cut(survival_at_ti, bins, include_lowest=True),
            "cens_spill_term": 1 / (n_bins * survival_at_ti),
        }
    )

    # computing uncensored counts:
    # sum the number of events per bin
    count_uncens = scoring_df.query("e == True").groupby("bin").size()

    # computing censored counts at bin of censoring
    # formula (A) as by page 49 of
    # https://arxiv.org/pdf/1811.11347.pdf
    count_cens = (
        scoring_df.query("e == False")
        .groupby("bin")
        .apply(lambda x: (1 - np.clip(x.name.left, 0, 1) / x["survival_at_ti"]).sum())
    )

    # computing censored counts at bins after censoring
    # effect of 'blurring'
    # formula (B) as by page 49 of
    # https://arxiv.org/pdf/1811.11347.pdf
    count_cens_spill = (
        scoring_df.query("e == False")
        .groupby("bin")["cens_spill_term"]
        .sum()
        .iloc[::-1]
        .shift()
        .fillna(0)
        .cumsum()
        .iloc[::-1]
    )

    final_bin_counts = count_uncens + count_cens + count_cens_spill

    if returns == "statistic":
        result = chisquare(final_bin_counts)
        return result.statistic

    elif returns == "pval":
        result = chisquare(final_bin_counts)
        return result.pvalue

    elif returns == "max_deviation":
        proportions = final_bin_counts / final_bin_counts.sum()
        return np.abs(proportions - 0.1).max()

    elif returns == "histogram":
        return final_bin_counts

    elif returns == "all":
        result = chisquare(final_bin_counts)
        proportions = final_bin_counts / final_bin_counts.mean()
        max_deviation = np.abs(proportions - 1).max()
        return {
            "statistic": result.statistic,
            "pval": result.pvalue,
            "max_deviation": max_deviation,
            "histogram": final_bin_counts,
        }
    else:
        raise ValueError(f"Chosen return of {returns} is not available.")
