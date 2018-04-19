from ftldrive.backend.python import sequential_ekf as sequential_ekf_python
from ftldrive.backend.cython import sequential_ekf as sequential_ekf_cython
from ftldrive.backend.numba import sequential_ekf as sequential_ekf_numba


BACKEND_DISPATCH = {}


def backend_registry(key):
    def decorator(func):
        BACKEND_DISPATCH[key] = func
        return func
    return decorator


@backend_registry('python')
def _python_backend(*args, **kwargs):
    return sequential_ekf_python(*args, **kwargs)


@backend_registry('cython')
def _cython_backend(*args, **kwargs):
    return sequential_ekf_cython(*args, **kwargs)


@backend_registry('numba')
def _cython_backend(*args, **kwargs):
    return sequential_ekf_numba(*args, **kwargs)


def sequential_ekf(*args, backend='python', **kwargs):
    return BACKEND_DISPATCH[backend](*args, **kwargs)
