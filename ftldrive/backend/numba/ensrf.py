import numpy as np
from numba import njit, prange


def serial_ensrf(state, obs_value, obs_error, obs_idx, inflation=None,
                 localization=None):
    """Serial ensemble square root filter.

    Parameters
    ----------
    state : array_like
        2D (m x n) state ensemble where m is the state vector size and n
        is ensemble size. Must not contain NaNs.
    obs_value : array_like
        1D (p) array of observations to be assimilated.
    obs_error : array_like
        1D (p) array of error variances for each observation.
    obs_idx : array_like
        1D (p) array containing indexes indicating which state vector element
        corresponds to a given observation.
    inflation : array_like
        1D (m) inflation factor applied to background state deviations before
        the background error covariance is calculated and before any
        observations are assimilated. A value of 1 leaves the covariance
        unchanged.
    localization : array_like
        2D (p x m) or 1D (p) array of values multiplied against the background error
        covariance matrix to limit the influence of each observation on the
        state. Values of 1 will leave the covariance unchanged.

    Returns
    -------
    updated_state : ndarray
        2D (m x n) updated state vector ensemble.

    References
    ----------
    Compo, G. P., Whitaker, J. S., & Sardeshmukh, P. D. (2006). Feasibility of
        a 100-Year Reanalysis Using Only Surface Pressure Data. Bulletin of the
        American Meteorological Society, 87(2), 175–190.
        https://doi.org/10.1175/BAMS-87-2-175
    Whitaker, J. S., Compo, G. P., Wei, X., & Hamill, T. M. (2004). Reanalysis
        without Radiosondes Using Ensemble Data Assimilation. Monthly Weather
        Review, 132(5), 1190–1200.
        https://doi.org/10.1175/1520-0493(2004)132<1190:RWRUED>2.0.CO;2
    """
    state = np.atleast_2d(state)
    obs_value = np.atleast_1d(obs_value)
    obs_error = np.atleast_1d(obs_error)
    obs_idx = np.atleast_1d(obs_idx)

    obs, error, idx = np.broadcast_arrays(obs_value, obs_error, obs_idx)
    # Below line guarantee contiguous in memory. Might be too bold if we're expecting our users to handle this.
    # obs, error, idx = [np.array(a) for a in np.broadcast_arrays(obs_value, obs_error, obs_idx)]

    # Ghetto check that all our sizes are good.
    m, n = state.shape
    p = len(obs_value)

    if inflation is not None:
        inflation = np.atleast_1d(inflation)
    else:
        inflation = np.ones(m)

    if localization is not None:
        localization = np.atleast_2d(localization)
    else:
        localization = np.ones(p)

    assert p == len(error), 'obs_value and obs_error need to have the same first dim size'
    assert p == len(idx), 'obs_value and obs_idx need to have the same first dim size'
    assert inflation.shape[0] == m, 'state and inflation need to have the same first dim size'
    assert localization.shape == (p, m) or localization.shape == (p,), 'state and inflation need to have the same first dim size'

    updated_state = obs_assimilation_loop(state, obs, error, idx, inflation, localization)

    return updated_state


@njit
def obs_assimilation_loop(state, obs, obs_error, obs_idx, inflation,
                          localization):
    """Serial ensemble square root filter observation assimilation loop.

    This function is intended to run fast, with little or no error checking.

    Parameters
    ----------
    state : array_like
        2D (m x n) state ensemble where m is the state vector size and n
        is ensemble size. Must not contain NaNs.
    obs : array_like
        1D (p) array of observations to be assimilated.
    obs_error : array_like
        1D (p) array of error variances for each observation.
    obs_idx : array_like
        1D (p) array containing indexes indicating which state vector element
        corresponds to a given observation.
    inflation : array_like
        1D (m) inflation factor applied to background state deviations before
        the background error covariance is calculated and before any
        observations are assimilated. A value of 1 leaves the covariance
        unchanged.
    localization : array_like
        2D (p x m) or 1D (p) array of values multiplied against the background
        error covariance matrix to limit the influence of each observation on
        the state. Values of 1 will leave the covariance unchanged.

    Returns
    -------
    updated_state : ndarray
        2D (m x n) updated state vector ensemble.
    """
    p = len(obs)

    updated_state = update(xb=state, yb=state[obs_idx[0]], y0=obs[0],
                        r=obs_error[0], infl=inflation, loc=localization[0])

    if p == 1:
        return updated_state

    for i in range(1, p):
        updated_state[...] = update(xb=updated_state,
                                    yb=updated_state[obs_idx[i]],
                                    y0=obs[i], r=obs_error[i],
                                    loc=localization[i])
    return updated_state


@njit
def update(xb, yb, y0, r, infl=1, loc=1):
    """Serial ensemble square root filter update step.

    Parameters
    ----------
    xb : array_like
        2D (m x n) background state vector ensemble where m is the state vector
        size and n is ensemble size. Must not contain NaNs.
    yb : array_like
        n-length ensemble of estimates for the observation.
    y0 : scalar
        Observation.
    r : scalar
        Observation error (variance).
    infl : array_like
        1D (m) inflation factor applied to background state deviations before
        the background error covariance is calculated and before any
        observations are assimilated. A value of 1 leaves the covariance
        unchanged.
    loc : array_like
        1D (m) covariance localization weights for background error covariance.
        A value of 1 leaves the covariance unchanged.

    Returns
    -------
    xa : ndarray
        2D (m x n) updated state vector ensemble.

    References
    ----------
    Compo, G. P., Whitaker, J. S., & Sardeshmukh, P. D. (2006). Feasibility of
        a 100-Year Reanalysis Using Only Surface Pressure Data. Bulletin of the
        American Meteorological Society, 87(2), 175–190.
        https://doi.org/10.1175/BAMS-87-2-175
    Whitaker, J. S., Compo, G. P., Wei, X., & Hamill, T. M. (2004). Reanalysis
        without Radiosondes Using Ensemble Data Assimilation. Monthly Weather
        Review, 132(5), 1190–1200.
        https://doi.org/10.1175/1520-0493(2004)132<1190:RWRUED>2.0.CO;2
    """
    # Background state mean and deviation.
    xb_bar = ensemble_mean(xb)  # (m x 1)
    xb_prime = xb - xb_bar  # (m x n)

    # Apply background deviation inflation.
    xb_prime *= infl

    # Obs estimate mean and deviation.
    yb_bar = np.mean(yb)  # (scalar)
    yb_prime = yb - yb_bar  # (m)

    # Obs estimate variance and covariance with the rest of the background state
    # (AKA background error covariance).
    yb_prime_var = sample_var_1d(yb_prime, ddof=1)  # (scalar)
    xb_prime_yb_prime_cov = sample_cov_2d1d(xb_prime, yb_prime, ddof=1)  # (m)

    # Apply covariance  localization weights.
    xb_prime_yb_prime_cov *= loc

    # Assemble kalman gains
    k = kalman_gain(xbye_cov=xb_prime_yb_prime_cov, yb_prime_var=yb_prime_var,
                    r=r)
    k_tilde = modified_kalman_gain(yb_prime_var=yb_prime_var, r=r, k=k)

    # Analysis state mean and deviations.
    xa_prime = analysis_deviation(xb_prime, k_tilde, yb_prime)
    xa_bar = analysis_mean(xb_bar, k, y0, yb_bar)

    return xa_bar + xa_prime


@njit
def sample_var_1d(x_prime, ddof=1):
    """Variance given a 1D array of deviations, given degrees of freedom.

    Returns
    -------
    scalar
    """
    n = len(x_prime)
    return np.sum(x_prime ** 2) / (n - ddof)


@njit(parallel=True)
def sample_cov_2d1d(x_prime, y_prime, ddof=1):
    """Covariance given arrays of deviations and degrees of freedom.

    Parameters
    ----------
    x_prime : ndarray
        2D (m x n) array.
    y_prime : ndarray
        1D (n) array.
    ddof : int, optional
        Degrees freedom used for (n - ddof) as denominator. Default is 1.

    Returns
    -------
    out : ndarray
        1D (m) array of covariances.
    """
    m = x_prime.shape[0]
    out = np.zeros(m)

    for i in prange(m):
        out[i] += np.sum(x_prime[i, :] * y_prime) / (m - ddof)

    return out


@njit(parallel=True)
def ensemble_mean(x):
    """Calculate the mean across ensembles for a state vector.

    This function is needed so that we can release the GIL while calculating an
    axis-targetting mean.

    Parameters
    ----------
    x : ndarray
        2D (m x n) state vector ensemble, where m the state vector size and n is
        ensemble size. Must not contain NaNs.

    Returns
    -------
    x_bar : ndarray
        2D (m x 1) state vector mean.
    """
    m = x.shape[0]
    x_bar = np.zeros((m, 1))
    n = x.shape[1]

    for i in prange(m):
        x_bar[i, 0] = np.sum(x[i, :]) / n

    return x_bar


@njit
def kalman_gain(xbye_cov, yb_prime_var, r):
    """Kalman gain (K) for sequential ensemble square root filter.

    Parameters
    ----------
    xbye_cov : array_like
        1D (m) array sample covariance between the background state deviations
        (xb_prime) and observation estimate deviations(ye_prime). Often noted as PbHt.
    yb_prime_var : scalar
        Sample variance of observation estimate deviations (yb_prime). Often noted as HPbHt.
    r : scalar
        Observation error variance.

    Returns
    -------
    k : ndarray
        2D (m x 1) array kalman gain.

    References
    ----------
    Compo, G. P., Whitaker, J. S., & Sardeshmukh, P. D. (2006). Feasibility of
        a 100-Year Reanalysis Using Only Surface Pressure Data. Bulletin of the
        American Meteorological Society, 87(2), 175–190.
        https://doi.org/10.1175/BAMS-87-2-175
    Whitaker, J. S., Compo, G. P., Wei, X., & Hamill, T. M. (2004). Reanalysis
        without Radiosondes Using Ensemble Data Assimilation. Monthly Weather
        Review, 132(5), 1190–1200.
        https://doi.org/10.1175/1520-0493(2004)132<1190:RWRUED>2.0.CO;2
    """
    m = len(xbye_cov)
    k = np.ones((m, 1))
    k[:, 0] = np.divide(xbye_cov, yb_prime_var + r)
    return k


@njit
def modified_kalman_gain(yb_prime_var, r, k):
    """Modified kalman gain (~K) for sequential ensemble square root filter.

    Parameters
    ----------
    yb_prime_var: scalar
        Sample variance of observation estimate deviations (yb_prime). Often noted as HPbHt.
    r: scalar
        Observation error variance.
    k: array_like
       2D (m x 1) Kalman gain (K).

    Returns
    -------
    k_tilde : ndarray
        2D (m x 1) array modified kalman gain.

    References
    ----------
    Compo, G. P., Whitaker, J. S., & Sardeshmukh, P. D. (2006). Feasibility of
        a 100-Year Reanalysis Using Only Surface Pressure Data. Bulletin of the
        American Meteorological Society, 87(2), 175–190.
        https://doi.org/10.1175/BAMS-87-2-175
    Whitaker, J. S., Compo, G. P., Wei, X., & Hamill, T. M. (2004). Reanalysis
        without Radiosondes Using Ensemble Data Assimilation. Monthly Weather
        Review, 132(5), 1190–1200.
        https://doi.org/10.1175/1520-0493(2004)132<1190:RWRUED>2.0.CO;2
    """
    return np.divide(1, 1 + np.sqrt(np.divide(r, yb_prime_var + r))) * k


@njit
def analysis_mean(xb_bar, k, y0, yb_bar):
    """Update step for analysis mean (xabar).

    Parameters
    ----------
    xb_bar : array_like
        1D (m) background state ensemble mean.
    k : array_like
        2D (m x 1) kalman gain (K).
    y0 : scalar
        Observation to be assimilated.
    yb_bar : scalar
        Ensemble mean estimate for observation.

    Returns
    -------
    xa_bar : ndarray
        1D (m) ensemble mean for the updated state.

    References
    ----------
    Compo, G. P., Whitaker, J. S., & Sardeshmukh, P. D. (2006). Feasibility of
        a 100-Year Reanalysis Using Only Surface Pressure Data. Bulletin of the
        American Meteorological Society, 87(2), 175–190.
        https://doi.org/10.1175/BAMS-87-2-175
    Whitaker, J. S., Compo, G. P., Wei, X., & Hamill, T. M. (2004). Reanalysis
        without Radiosondes Using Ensemble Data Assimilation. Monthly Weather
        Review, 132(5), 1190–1200.
        https://doi.org/10.1175/1520-0493(2004)132<1190:RWRUED>2.0.CO;2
    """
    return xb_bar + k * (y0 - yb_bar)


@njit
def analysis_deviation(xb_prime, k_tilde, yb_prime):
    """Update step for analysis deviation (xa_prime).

    Parameters
    ----------
    xb_prime : array_like
        1D (m) background state ensemble deviation.
    k_tilde: array_like
        2D (m x 1) modified kalman gain.
    yb_prime : array_like
        1D (n) Ensemble deviation for observation estimate.

    Returns
    -------
    xa_prime : ndarray
        2D (m x n) ensemble deviation for the updated state.

    References
    ----------
    Compo, G. P., Whitaker, J. S., & Sardeshmukh, P. D. (2006). Feasibility of
        a 100-Year Reanalysis Using Only Surface Pressure Data. Bulletin of the
        American Meteorological Society, 87(2), 175–190.
        https://doi.org/10.1175/BAMS-87-2-175
    Whitaker, J. S., Compo, G. P., Wei, X., & Hamill, T. M. (2004). Reanalysis
        without Radiosondes Using Ensemble Data Assimilation. Monthly Weather
        Review, 132(5), 1190–1200.
        https://doi.org/10.1175/1520-0493(2004)132<1190:RWRUED>2.0.CO;2
    """
    return xb_prime - k_tilde * yb_prime
