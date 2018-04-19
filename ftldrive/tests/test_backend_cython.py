import pytest
import numpy as np
from ftldrive.oldcore import sequential_ekf


@pytest.fixture()
def simple_bstate():
    bstate = np.ones((9, 6))  # 6 is ensemble size
    bstate[4, :] += 0.5
    noise = np.arange(-0.3, 0.3, 0.1)
    bstate += noise
    return bstate


@pytest.mark.parametrize('in_kwargs,goal', [
    ({'obs_value': 1.35, 'obs_error': 0.05, 'obs_idx': 4},
     {'mean': np.array([0.90578947, 0.90578947, 0.90578947,
                        0.90578947, 1.40578947, 0.90578947,
                        0.90578947, 0.90578947, 0.90578947]),
      'var': np.array([0.01656691, 0.01656691, 0.01656691,
                       0.01656691, 0.01656691, 0.01656691,
                       0.01656691, 0.01656691, 0.01656691])}),

    ({'obs_value': 1.35, 'obs_error': 0.05, 'obs_idx': 4, 'inflation': 1.5},
     {'mean': np.array([0.88368421, 0.88368421, 0.88368421,
                        0.88368421, 1.38368421, 0.88368421,
                        0.88368421, 0.88368421, 0.88368421]),
      'var': np.array([0.01159445, 0.01159445, 0.01159445,
                       0.01159445, 0.01159445, 0.01159445,
                       0.01159445, 0.01159445, 0.01159445])}),

    ({'obs_value': 1.35, 'obs_error': 0.05, 'obs_idx': 4, 'localization': np.ones(9) * 0.5},
     {'mean': np.array([0.92789474, 0.92789474, 0.92789474,
                        0.92789474, 1.42789474, 0.92789474,
                        0.92789474, 0.92789474, 0.92789474]),
      'var': np.array([0.02242432, 0.02242432, 0.02242432,
                       0.02242432, 0.02242432, 0.02242432,
                       0.02242432, 0.02242432, 0.02242432])}),

    ({'obs_value': [1.34, 0.95], 'obs_error': [0.02, 0.05], 'obs_idx': [4, 7]},
     {'mean': np.array([0.88646139, 0.88646139, 0.88646139,
                        0.88646139, 1.38646139, 0.88646139,
                        0.88646139, 0.88646139, 0.88646139]),
      'var': np.array([0.00757937, 0.00757937, 0.00757937,
                       0.00757937, 0.00757937, 0.00757937,
                       0.00757937, 0.00757937, 0.00757937])}),

])
def test_sequential_ekf(simple_bstate, in_kwargs, goal):
    ustate = sequential_ekf(simple_bstate.copy(), backend='python', **in_kwargs)

    np.testing.assert_allclose(ustate.mean(axis=1), goal['mean'], atol=1e-3, rtol=0)
    np.testing.assert_allclose(ustate.var(axis=1), goal['var'], atol=1e-3, rtol=0)
