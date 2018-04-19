from numba import jit
import numpy as np


@jit
def sequential_ekf(state, obs_value, obs_error, obs_idx, inflation=None,
                   localization=None):
    state = np.atleast_2d(state)
    obs_value = np.atleast_1d(obs_value)
    obs_error = np.atleast_1d(obs_error)
    obs_idx = np.atleast_1d(obs_idx)

    # Might be faster to guarentee contiguous memory with `[np.array(a) for a in np.broadcast_arrays(x, y)]`
    obs, error, idx = np.broadcast_arrays(obs_value, obs_error, obs_idx)

    n = len(obs_value)

    for i in range(n):
        obs_estimate = np.ravel(state[idx[i]])

        # If localization is None. Best to ask forgiveness.
        loc = None

        if localization is not None:
            loc = localization[i]

        state[...] = update_state(prior_ensemble=state, obs=obs[i],
                                  obs_estimate=obs_estimate, obs_error=error[i],
                                  inflation=inflation, localization=loc)
    return state


@jit
def update_state(prior_ensemble, obs, obs_estimate, obs_error,
                 localization=None, inflation=None):
    # Get ensemble size from passed array: prior_ensemble has dims [state vect.,ens. members]
    n_ensemble = prior_ensemble.shape[-1]

    # ensemble mean background and perturbations
    prior_mean = np.mean(prior_ensemble, axis=1)
    prior_anomaly = prior_ensemble - prior_mean[:, None]  # "None" means replicate in this dimension

    # ensemble mean and variance of the background estimate of the proxy
    obs_estimate_mean = np.mean(obs_estimate)
    obs_estimate_var = np.var(obs_estimate)

    obs_estimate_anomaly = obs_estimate - obs_estimate_mean
    innovation = obs - obs_estimate_mean

    # innovation variance (denominator of serial Kalman gain)
    gain_denom = obs_estimate_var + obs_error
    # numerator of serial Kalman gain (cov(x,Hx))
    gain_numer = prior_anomaly @ np.transpose(obs_estimate_anomaly) / (n_ensemble - 1)

    # Option to inflation the covariances by a certain factor
    if inflation is not None:
        gain_numer = inflation * gain_numer
    # Option to localize the gain
    if localization is not None:
        gain_numer = gain_numer * localization

    # Kalman gain
    gain = gain_numer / gain_denom
    # update ensemble mean
    update_mean = prior_mean + gain * innovation

    # update the ensemble members using the square-root approach
    beta = 1 / (1 + np.sqrt(obs_error / (obs_estimate_var + obs_error)))
    gain = beta * gain
    obs_estimate_anomaly = np.atleast_2d(obs_estimate_anomaly)
    gain = np.atleast_2d(gain)
    update_anomaly = prior_anomaly - np.transpose(gain) @ obs_estimate_anomaly

    # full state
    # import pdb;pdb.set_trace()
    posterior_ensemble = update_mean[:, None] + update_anomaly

    # Return the full state,
    return posterior_ensemble
