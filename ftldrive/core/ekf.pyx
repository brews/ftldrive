import numpy as np
cimport numpy as cnp
cimport cython


def sequential_ekf(state, obs_value, obs_error, obs_idx, inflation=None, 
                   localization=None):
    state = np.atleast_2d(state).astype('double')
    obs_value = np.atleast_1d(obs_value).astype('double')
    obs_error = np.atleast_1d(obs_error).astype('double')
    obs_idx = np.atleast_1d(obs_idx).astype('long')

    # TODO(brew): Need test that non-state arrays are of equal length.

    # Guarantee contiguous in memory.
    obs, error, idx = [np.array(a) for a in np.broadcast_arrays(obs_value, obs_error, obs_idx)]

    if inflation is not None:
        inflation = np.atleast_1d(inflation).astype('double')

    if localization is not None:
        localization = np.atleast_2d(localization).astype('double')

    state = fast_obs_assimilation(state, obs, error, idx, inflation, localization)

    return np.asarray(state)


# TODO(brews): Go through and declare key arrays as contiguous (https://cython.readthedocs.io/en/latest/src/userguide/numpy_tutorial.html)
# @cython.boundscheck(False)
# @cython.wraparound(False)
cdef double[:,:] fast_obs_assimilation(double[:,:] state, double[:] obs_value, 
    double[:] obs_error, long[:] obs_idx, double[:] inflation=None,
    double[:,:] localization=None):

    cdef long n = len(obs_value)
    cdef double[:] obs_estimate

    cdef double[:] this_localization = None

    if localization is None:
        print('boo')

    for i in range(n):
        obs_estimate = np.ravel(state[obs_idx[i]])
        if localization is not None:
            this_localization = localization[i]
        state[...] = sequential_filter(prior_ensemble=state, obs=obs_value[i],
                                       obs_estimate=obs_estimate, 
                                       obs_error=obs_error[i], 
                                       inflation=inflation,
                                       localization=this_localization)
    return state


# @cython.boundscheck(False)
# @cython.wraparound(False)
cdef double[:,:] sequential_filter(double[:,:] prior_ensemble, double obs, 
    double[:] obs_estimate, double obs_error, double[:] inflation=None,
    double[:] localization=None):

    # Get ensemble size from passed array: prior_ensemble has dims [state vect.,ens. members]
    cdef long n_ensemble = np.shape(prior_ensemble)[1]

    # ensemble mean background and perturbations
    cdef double[:] prior_mean = np.mean(prior_ensemble, axis=1)
    cdef double[:,:] prior_anomaly = np.subtract(prior_ensemble, 
                                                 prior_mean[:, None])

    # ensemble mean and variance of the background estimate of the proxy 
    cdef double obs_estimate_mean = np.mean(obs_estimate)
    cdef double obs_estimate_var = np.var(obs_estimate)

    cdef double[:] obs_estimate_anomaly = np.subtract(obs_estimate, obs_estimate_mean)
    cdef double innovation = obs - obs_estimate_mean
    
    # innovation variance (denominator of serial Kalman gain)
    cdef double gain_denom = obs_estimate_var + obs_error
    # numerator of serial Kalman gain (cov(x,Hx))
    cdef double[:] gain_numer = np.dot(prior_anomaly, np.transpose(obs_estimate_anomaly)) / (n_ensemble - 1)

    # Optional covariance inflation
    if inflation is not None:
        gain_numer = np.multiply(inflation, gain_numer)

    # Optional covariance localization
    if localization is not None:
        gain_numer = np.multiply(gain_numer, localization)

    # Kalman gain
    cdef double[:] gain = np.divide(gain_numer, gain_denom)
    # update ensemble mean
    cdef double[:] update_mean = prior_mean + np.multiply(gain, innovation)

    # update the ensemble members using the square-root approach
    cdef double alpha = 1 / (1 + np.sqrt(obs_error / (obs_estimate_var + obs_error)))
    gain = np.multiply(alpha, gain)
    cdef double[:,:] update_anomaly = prior_anomaly - np.dot(gain[:, None], obs_estimate_anomaly[None])

    # full state
    cdef double[:,:] posterior_ensemble = np.add(update_mean[:, None], update_anomaly)

    return posterior_ensemble

