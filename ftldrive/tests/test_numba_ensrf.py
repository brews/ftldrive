import pytest
import numpy as np
from ftldrive.backend.numba.ensrf import (ensemble_mean, sample_var_1d,
    sample_cov_2d1d, kalman_gain, modified_kalman_gain, analysis_deviation,
    analysis_mean, update, obs_assimilation_loop, serial_ensrf, inflate_state_variance)


def test_ensemble_mean():
    victim = np.arange(15).reshape(5, 3)
    goal = np.array([[1.0, 4.0, 7.0, 10.0, 13.0]]).T
    actual = ensemble_mean(victim)
    np.testing.assert_allclose(actual, goal, atol=1e-15, rtol=0)


def test_inflate_state_variance():
    state = np.arange(15).reshape(5, 3)
    inflation = np.ones((state.shape[0], 1)) * 2.0
    goal = np.array([[-1., 1., 3.],
                     [2., 4., 6.],
                     [5., 7., 9.],
                     [8., 10., 12.],
                     [11., 13., 15.]])
    actual = inflate_state_variance(x=state, infl=inflation)
    np.testing.assert_allclose(actual, goal, atol=1e-15, rtol=0)


def test_sample_var_1d():
    x_prime = np.array([-1, 0, 1])
    goal = 1.0
    actual = sample_var_1d(x_prime, ddof=1)
    np.testing.assert_allclose(actual, goal, atol=1e-15, rtol=0)


def test_sample_cov_2d1d():
    y_prime = np.array([-1, 0, 1])
    x = np.arange(15).reshape((5, 3))
    x_prime = x - x.mean(axis=1)[:, None]
    goal = np.array([0.5, 0.5, 0.5, 0.5, 0.5])
    actual = sample_cov_2d1d(x_prime=x_prime, y_prime=y_prime, ddof=1)
    np.testing.assert_allclose(actual, goal, atol=1e-15, rtol=0)


def test_kalman_gain():
    xb_yb_cov = np.array([0.5, 0.5, 0.5, 0.5, 0.5])
    yb_var = 1.0
    r = 0.25
    actual = kalman_gain(xbye_cov=xb_yb_cov, yb_prime_var=yb_var, r=r)
    goal = np.array([[0.4, 0.4, 0.4, 0.4, 0.4]]).T
    np.testing.assert_allclose(actual, goal, atol=1e-15, rtol=0)


def test_modified_kalman_gain():
    yb_var = 1.0
    r = 0.25
    k = np.array([[0.4, 0.4, 0.4, 0.4, 0.4]]).T
    actual = modified_kalman_gain(yb_prime_var=yb_var, r=r, k=k)
    goal = np.array([[0.2763932, 0.2763932, 0.2763932, 0.2763932, 0.2763932]]).T
    np.testing.assert_allclose(actual, goal, atol=1e-6, rtol=0)


def test_analysis_deviation():
    y_prime = np.array([-1, 0, 1])
    x = np.arange(15).reshape((5, 3))
    x_prime = x - x.mean(axis=1)[:, None]
    k_tilde = np.array([[0.2763932, 0.2763932, 0.2763932, 0.2763932, 0.2763932]]).T
    actual = analysis_deviation(xb_prime=x_prime, k_tilde=k_tilde,
                                yb_prime=y_prime)
    goal = np.array([[-0.7236068, 0, 0.7236068],
                     [-0.7236068, 0, 0.7236068],
                     [-0.7236068, 0, 0.7236068],
                     [-0.7236068, 0, 0.7236068],
                     [-0.7236068, 0, 0.7236068]])
    np.testing.assert_allclose(actual, goal, atol=1e-7, rtol=0)


def test_analysis_mean():
    x = np.arange(15).reshape((5, 3))
    xb_bar = ensemble_mean(x)
    y0 = 6.5
    yb_bar = 7.0
    k = np.array([[0.4, 0.4, 0.4, 0.4, 0.4]]).T
    actual = analysis_mean(xb_bar=xb_bar, k=k, y0=y0, yb_bar=yb_bar)
    goal = np.array([[0.8, 3.8, 6.8, 9.8, 12.8]]).T
    np.testing.assert_allclose(actual, goal, atol=1e-7, rtol=0)


def test_update():
    xb = np.arange(15).reshape(5, 3)
    yb = xb[2, :]
    y0 = 6.5
    r = 0.25
    loc = 1
    actual = update(xb=xb, yb=yb, y0=y0, r=r, loc=loc)
    goal = np.array([[0.0763932, 0.8, 1.5236068],
                     [3.0763932, 3.8, 4.5236068],
                     [6.0763932, 6.8, 7.5236068],
                     [9.0763932, 9.8, 10.5236068],
                     [12.0763932, 12.8, 13.5236068]])
    np.testing.assert_allclose(actual, goal, atol=1e-7, rtol=0)


def test_obs_assimilation_loop_1obs():
    state = np.arange(15).reshape(5, 3)
    obs = np.array([6.5])
    obs_error = np.array([0.25])
    obs_idx = np.array([2, ])
    inflation = np.ones((state.shape[0], 1))
    localization = np.ones((len(obs), state.shape[0]))
    actual = obs_assimilation_loop(state=state, obs=obs, obs_error=obs_error,
                                   obs_idx=obs_idx, inflation=inflation,
                                   localization=localization)
    goal = np.array([[0.0763932, 0.8, 1.5236068],
                     [3.0763932, 3.8, 4.5236068],
                     [6.0763932, 6.8, 7.5236068],
                     [9.0763932, 9.8, 10.5236068],
                     [12.0763932, 12.8, 13.5236068]])
    np.testing.assert_allclose(actual, goal, atol=1e-7, rtol=0)


def test_obs_assimilation_loop_1obs2():
    state = np.array([[0.0763932, 0.8, 1.5236068],
                     [3.0763932, 3.8, 4.5236068],
                     [6.0763932, 6.8, 7.5236068],
                     [9.0763932, 9.8, 10.5236068],
                     [12.0763932, 12.8, 13.5236068]])
    obs = np.array([0.4])
    obs_error = np.array([0.35])
    obs_idx = np.array([0, ])
    inflation = np.ones((state.shape[0], 1))
    localization = np.ones((len(obs), state.shape[0]))
    actual = obs_assimilation_loop(state=state, obs=obs, obs_error=obs_error,
                                   obs_idx=obs_idx, inflation=inflation,
                                   localization=localization)
    goal = np.array([[0.08931723, 0.68012758, 1.27093793],
                     [3.08931723, 3.68012758, 4.27093793],
                     [6.08931723, 6.68012758, 7.27093793],
                     [9.08931723, 9.68012758, 10.27093793],
                     [12.08931723, 12.68012758, 13.27093793]])
    np.testing.assert_allclose(actual, goal, atol=1e-7, rtol=0)


def test_obs_assimilation_loop_2obs():
    state = np.arange(15).reshape(5, 3)
    obs = np.array([6.5, 0.4])
    obs_error = np.array([0.25, 0.35])
    obs_idx = np.array([2, 0])
    inflation = np.ones((state.shape[0], 1))
    localization = np.ones((len(obs)))
    actual = obs_assimilation_loop(state=state, obs=obs, obs_error=obs_error,
                                   obs_idx=obs_idx, inflation=inflation,
                                   localization=localization)
    goal = np.array([[0.08931723, 0.68012758, 1.27093793],
                     [3.08931723, 3.68012758, 4.27093793],
                     [6.08931723, 6.68012758, 7.27093793],
                     [9.08931723, 9.68012758, 10.27093793],
                     [12.08931723, 12.68012758, 13.27093793]])
    np.testing.assert_allclose(actual, goal, atol=1e-7, rtol=0)


def test_serial_ensrf():
    state = np.arange(15).reshape(5, 3)
    obs = np.array([6.5, 0.4])
    obs_error = np.array([0.25, 0.35])
    obs_idx = np.array([2, 0])
    goal = np.array([[0.08931723, 0.68012758, 1.27093793],
                     [3.08931723, 3.68012758, 4.27093793],
                     [6.08931723, 6.68012758, 7.27093793],
                     [9.08931723, 9.68012758, 10.27093793],
                     [12.08931723, 12.68012758, 13.27093793]])
    actual = serial_ensrf(state=state, obs_value=obs, obs_error=obs_error,
                          obs_idx=obs_idx, inflation=None, localization=None)
    np.testing.assert_allclose(actual, goal, atol=1e-7, rtol=0)


def test_serial_ensrf_inflation_1obs():
    state = np.arange(15).reshape(5, 3)
    m = state.shape[0]
    obs = np.array([0.4])
    obs_error = np.array([0.35])
    obs_idx = np.array([0])
    inflation = np.ones((m, 1)) * 2.0
    goal = np.array([[-0.55951638, 0.72413793, 2.00779225],
                     [2.44048362, 3.72413793, 5.00779225],
                     [5.44048362, 6.72413793, 8.00779225],
                     [8.44048362, 9.72413793, 11.00779225],
                     [11.44048362, 12.72413793, 14.00779225]])
    actual = serial_ensrf(state=state, obs_value=obs, obs_error=obs_error,
                          obs_idx=obs_idx, inflation=inflation, localization=None)
    np.testing.assert_allclose(actual, goal, atol=1e-7, rtol=0)


def test_serial_ensrf_inflation_2obs():
    state = np.arange(15).reshape(5, 3)
    m = state.shape[0]
    obs = np.array([6.5, 0.4])
    obs_error = np.array([0.25, 0.35])
    obs_idx = np.array([2, 0])
    inflation = np.ones((m, 1)) * 2.0
    goal = np.goal = np.array([[-0.27229131, 0.61605256, 1.50439643],
                               [2.72770869, 3.61605256, 4.50439643],
                               [5.72770869, 6.61605256, 7.50439643],
                               [8.72770869, 9.61605256, 10.50439643],
                               [11.72770869, 12.61605256, 13.50439643]])
    actual = serial_ensrf(state=state, obs_value=obs, obs_error=obs_error,
                          obs_idx=obs_idx, inflation=inflation, localization=None)
    np.testing.assert_allclose(actual, goal, atol=1e-7, rtol=0)


def test_serial_ensrf_localization():
    state = np.arange(15).reshape(5, 3)
    obs = np.array([6.5, 0.4])
    obs_error = np.array([0.25, 0.35])
    obs_idx = np.array([2, 0])
    localization = np.array([[0, 0.5, 1.0, 0.5, 0],
                             [1.0, 0.5, 0, 0, 0]])
    goal = np.array([[0.02319024, 0.77777778, 1.53236532],
                     [3.04818931, 3.80424407, 4.56029882],
                     [6.0763932, 6.8, 7.5236068],
                     [9.0381966, 9.9, 10.7618034],
                     [12.0, 13., 14.0]])
    actual = serial_ensrf(state=state, obs_value=obs, obs_error=obs_error,
                          obs_idx=obs_idx, inflation=None,
                          localization=localization)
    np.testing.assert_allclose(actual, goal, atol=1e-7, rtol=0)


def test_serial_ensrf_localization_inflation():
    state = np.arange(15).reshape(5, 3)
    m = state.shape[0]
    obs = np.array([6.5, 0.4])
    obs_error = np.array([0.25, 0.35])
    obs_idx = np.array([2, 0])
    inflation = np.ones((m, 1)) * 2.0
    localization = np.array([[0, 0.5, 1.0, 0.5, 0],
                             [1.0, 0.5, 0, 0, 0]])
    goal = np.array([[-0.55951638, 0.72413793, 2.00779225],
                     [2.43962061, 3.77054137, 5.10146213],
                     [5.52217026, 6.76470588, 8.00724151],
                     [8.26108513, 9.88235294, 11.50362075],
                     [11.0, 13.0, 15.0]])
    actual = serial_ensrf(state=state, obs_value=obs, obs_error=obs_error,
                          obs_idx=obs_idx, inflation=inflation,
                          localization=localization)
    np.testing.assert_allclose(actual, goal, atol=1e-7, rtol=0)
