import pytest
import numpy as np
from ftldrive.backend.numba.ensrf import (ensemble_mean, kalman_gain,
    modified_kalman_gain, analysis_deviation, analysis_mean, update,
    obs_assimilation_loop, serial_ensrf, inflate_state_variance)


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
    goal = np.array([[  0.1527864,   0.6,   1.0472136],
                     [  3.1527864,   3.6,   4.0472136],
                     [  6.1527864,   6.6,   7.0472136],
                     [  9.1527864,   9.6,  10.0472136],
                     [ 12.1527864,  12.6,  13.0472136]])
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
    goal = np.array([[  0.1527864,   0.6,   1.0472136],
                     [  3.1527864,   3.6,   4.0472136],
                     [  6.1527864,   6.6,   7.0472136],
                     [  9.1527864,   9.6,  10.0472136],
                     [ 12.1527864,  12.6,  13.0472136]])
    np.testing.assert_allclose(actual, goal, atol=1e-7, rtol=0)


def test_obs_assimilation_loop_1obs2():
    state = np.array([[  0.1527864,   0.6,   1.0472136],
                     [  3.1527864,   3.6,   4.0472136],
                     [  6.1527864,   6.6,   7.0472136],
                     [  9.1527864,   9.6,  10.0472136],
                     [ 12.1527864,  12.6,  13.0472136]])
    obs = np.array([0.4])
    obs_error = np.array([0.35])
    obs_idx = np.array([0, ])
    inflation = np.ones((state.shape[0], 1))
    localization = np.ones((len(obs), state.shape[0]))
    actual = obs_assimilation_loop(state=state, obs=obs, obs_error=obs_error,
                                   obs_idx=obs_idx, inflation=inflation,
                                   localization=localization)
    goal = np.array([[  0.17051969,   0.52727273,   0.88402576],
                    [  3.17051969,   3.52727273,   3.88402576],
                    [  6.17051969,   6.52727273,   6.88402576],
                    [  9.17051969,   9.52727273,   9.88402576],
                    [ 12.17051969,  12.52727273,  12.88402576]])
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
    goal = np.array([[  0.17051969,   0.52727273,   0.88402576],
                    [  3.17051969,   3.52727273,   3.88402576],
                    [  6.17051969,   6.52727273,   6.88402576],
                    [  9.17051969,   9.52727273,   9.88402576],
                    [ 12.17051969,  12.52727273,  12.88402576]])
    np.testing.assert_allclose(actual, goal, atol=1e-7, rtol=0)


def test_serial_ensrf():
    state = np.arange(15).reshape(5, 3)
    obs = np.array([6.5, 0.4])
    obs_error = np.array([0.25, 0.35])
    obs_idx = np.array([2, 0])
    goal = np.array([[  0.17051969,   0.52727273,   0.88402576],
                    [  3.17051969,   3.52727273,   3.88402576],
                    [  6.17051969,   6.52727273,   6.88402576],
                    [  9.17051969,   9.52727273,   9.88402576],
                    [ 12.17051969,  12.52727273,  12.88402576]])
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
    goal = np.array([[ -0.11903277,   0.44827586,   1.01558449],
                     [  2.88096723,   3.44827586,   4.01558449],
                     [  5.88096723,   6.44827586,   7.01558449],
                     [  8.88096723,   9.44827586,  10.01558449],
                     [ 11.88096723,  12.44827586,  13.01558449]])
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
    goal = np.array([[  0.10228226,   0.47738693,   0.85249161],
                     [  3.10228226,   3.47738693,   3.85249161],
                     [  6.10228226,   6.47738693,   6.85249161],
                     [  9.10228226,   9.47738693,   9.85249161],
                     [ 12.10228226,  12.47738693,  12.85249161]])
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
    goal = np.array([[  0.04638048,   0.55555556,   1.06473063],
                     [  3.09317382,   3.63919849,   4.18522316],
                     [  6.1527864 ,   6.6       ,   7.0472136 ],
                     [  9.0763932 ,   9.8       ,  10.5236068 ],
                     [ 12.        ,  13.        ,  14.        ]])
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
    goal = np.array([[ -0.11903277,   0.44827586,   1.01558449],
                     [  2.79582855,   3.59332166,   4.39081477],
                     [  6.04434051,   6.52941176,   7.01448301],
                     [  8.52217026,   9.76470588,  11.00724151],
                     [ 11.        ,  13.        ,  15.        ]])
    actual = serial_ensrf(state=state, obs_value=obs, obs_error=obs_error,
                          obs_idx=obs_idx, inflation=inflation,
                          localization=localization)
    np.testing.assert_allclose(actual, goal, atol=1e-7, rtol=0)
