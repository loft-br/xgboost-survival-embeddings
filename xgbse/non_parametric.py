# basic imports
import numpy as np
import pandas as pd

# epsilon to prevent division by zero
EPS = 1e-6


def get_time_bins(T, E, size=12):
    """
    Method to automatically define time bins
    """

    lower_bound = max(T[E == 0].min(), T[E == 1].min()) + 1
    upper_bound = min(T[E == 0].max(), T[E == 1].max()) - 1

    return np.linspace(lower_bound, upper_bound, size, dtype=int)


def sort_and_add_zeros(x, ind):
    """
    Sorts an specified array x according to a reference index ind

    Args:
        x (np.array): Array to be sorted according to index specified in ind
        ind (np.array): Index to be used as reference to sort array x

    Returns:
        np.array: Array x sorted according to ind indexes
    """

    x = np.take_along_axis(x, ind, axis=1)
    # Check with concatenate
    x = np.c_[np.zeros(x.shape[0]), x]

    return x


# Utils to calculate survival intervals
def calculate_exp_boundary(survival_func, V, z):
    """
    Creates confidence intervals using the Exponential Greenwood formula.
    Available at: https://www.math.wustl.edu/~sawyer/handouts/greenwood.pdf

    Args:
        survival_func ([np.array, pd.Series]): Survival function estimates
        V ([np.array, pd.Series]): Exponential Greenwood variance component
        z (Int): Normal quantile to be used - depends on the confidence level

    Returns:
        pd.DataFrame: Confidence intervals
    """

    C = np.log(-np.log(survival_func)) + z * np.sqrt(V)
    C_exp = np.exp(-np.exp(C))

    return pd.DataFrame(C_exp).fillna(method="bfill").fillna(method="ffill").values


def sample_time_bins(surv_array, T_neighs, time_bins):
    """
    Sample survival curve at specified points in time to get a survival df

    Args:
        surv_array (np.array): Survival array to be sampled from

        T_neighs (np.array): Array of observed times

        time_bins (List): Specified time bins to retrieve survival estimates

    Returns:
        pd.DataFrame: DataFrame with survival for each specified time
        bin
    """

    surv_df = []

    for t in time_bins:
        survival_at_t = (surv_array + (T_neighs > t)).min(axis=1)
        surv_df.append(survival_at_t)

    surv_df = pd.DataFrame(surv_df, index=time_bins).T
    return surv_df


def sort_times_and_events(T, E):
    """
    Collects and sorts times and events from neighbors set

    Args:
        T (np.array): matrix of times (will compute one kaplan meier for each row)

        E (np.array): matrix of events (will compute one kaplan meier for each row)
    Returns:
        (np.array, np.array): matrix of times, sorted by most recent, matrix of events. sorted by most recent
    """

    # getting sorted indices for times along each neighbor-set
    argsort_ind = np.argsort(T, axis=1)

    # reordering times and events according to sorting and adding t=0
    T_sorted = sort_and_add_zeros(T, argsort_ind)
    E_sorted = sort_and_add_zeros(E, argsort_ind)

    return T_sorted, E_sorted


def calculate_survival_func(E_sorted):
    """
    Calculates the survival function for a given set of neighbors

    Args:
        E_sorted (np.array): A time-sorted array (row-wise) of events/censors

    Returns:
        np.array: The survival function evaluated for each row
    """

    # max number of elements in leafs
    # TODO: allow numpy work with variable size arrays
    n_samples = E_sorted.shape[1] - 1

    # number of elements at risk
    at_risk = np.r_[n_samples, np.arange(n_samples, 0, -1)]

    # product argument for surivial
    survival_prod_arg = 1 - (E_sorted / at_risk)

    return np.cumprod(survival_prod_arg, axis=1)


def calculate_confidence_intervals(E_sorted, survival_func, z):
    """
    Calculates confidence intervals based on the Exponential Greenwood
    formula. Available at: https://www.math.wustl.edu/~sawyer/handouts/greenwood.pdf

    Args:
        E_sorted (np.array): A time-sorted array (row-wise) of events/censors

        survival_func (np.array): Survival function array to be used as a
        baseline to calculate confidence interval

    Returns:
        (np.array, np.array): Upper confidence interval (95%), Lower condidence interval (95%)
    """

    # guarantee that survival_func
    # is strictly positive and not exactly equal to 1
    # for numerical purposess
    survival_func = np.clip(survival_func, EPS, 1 - EPS)

    # max number of elements in leafs
    # TODO: allow numpy work with variable size arrays
    n_samples = E_sorted.shape[1] - 1

    # number of elements at risk
    at_risk = np.r_[n_samples, np.arange(n_samples, 0, -1)]

    # also calculating confidence intervals
    numerator = E_sorted.astype(float)
    denominator = at_risk * (at_risk - E_sorted)
    ci_prod_arg = numerator / np.clip(denominator, EPS, None)

    # exponential greenwood variance component
    numerator = np.cumsum(ci_prod_arg, axis=1)
    denominator = np.power(np.log(survival_func), 2)
    V = numerator / np.clip(denominator, EPS, None)

    # calculating upper and lower confidence intervals (one standard deviation)
    upper_ci = calculate_exp_boundary(survival_func, V, z)
    lower_ci = calculate_exp_boundary(survival_func, V, -z)

    return upper_ci, lower_ci


def calculate_kaplan_vectorized(T, E, time_bins, z=1.0):
    """
    Predicts a Kaplan Meier estimator in a vectorized manner, including
    its upper and lower confidence intervals based on the Exponential
    Greenwod formula. See _calculate_survival_func for theoretical reference

    Args:
        T (np.array): matrix of times (will compute one kaplan meier for each row)

        E (np.array): matrix of events (will compute one kaplan meier for each row)

        time_bins (List): Specified time bins to retrieve survival estimates

    Returns:
        (np.array, np.array, np.array): survival values at specified time bins,
            upper confidence interval, lower confidence interval

    """

    # sorting times and events
    T_sorted, E_sorted = sort_times_and_events(T, E)
    E_sorted = E_sorted.astype(int)

    # calculating survival functions
    survival_func = calculate_survival_func(E_sorted)

    # calculating confidence intervals
    upper_ci, lower_ci = calculate_confidence_intervals(E_sorted, survival_func, z)

    # only returning time bins asked by user
    survival_func = sample_time_bins(survival_func, T_sorted, time_bins)
    upper_ci = sample_time_bins(upper_ci, T_sorted, time_bins)
    lower_ci = sample_time_bins(lower_ci, T_sorted, time_bins)

    return survival_func, upper_ci, lower_ci


def _get_conditional_probs_from_survival(surv):
    """
    Return conditional failure probabilities (for each time interval) from survival curve.
    P(T < t+1 | T > t): probability of failure up to time t+1 conditional on individual
    survival up to time t.

    Args:
        surv (pd.DataFrame): dataframe of survival estimates, as .predict() methods return

    Returns:
        pd.DataFrame: conditional failurer probability of event
            specifically at time bucket
    """

    conditional_preds = 1 - (surv / surv.shift(1, axis=1).fillna(1))
    conditional_preds = conditional_preds.fillna(0)

    return conditional_preds


def _get_point_probs_from_survival(conditional_preds):
    """
    Transform conditional failure probabilities into point probabilities
    (at each interval) from survival curve.
    P(t < T < t+1): point probability of failure between time t and t+1.

    Args:
        conditional_preds (pd.DataFrame): dataframe of conditional failure probability -
        output of _get_conditional_probs_from_survival function

    Returns:
        pd.DataFrame: probability of event at all specified time buckets
    """

    sample = conditional_preds.reset_index(drop=True)

    # list of event probabilities summing up to 1
    event = []

    # event in time interval 0
    v_0 = 1 * sample[0]
    event.append(v_0)

    # Looping over other time intervals
    for i in range(1, len(sample)):
        v_i = (1 - sum(event)) * sample[i]
        event.append(v_i)

    return pd.Series(event)


def calculate_interval_failures(surv):
    """
    Return point probabilities (at each interval) from survival curve.
    P(t < T < t+1): point probability of failure between time t and t+1.

    Args:
        surv (pd.DataFrame): dataframe of (1 - cumulative survival estimates),
        complementary of .predict() methods return

    Returns:
        pd.DataFrame: probability of event at all specified time buckets
    """

    interval_preds = _get_conditional_probs_from_survival(surv).apply(
        _get_point_probs_from_survival, axis=1
    )
    interval_preds.columns = surv.columns

    return interval_preds
