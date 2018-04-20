from ftldrive.backend.python import serial_ensrf as serial_ensrf_python
from ftldrive.backend.cython import serial_ensrf as serial_ensrf_cython
from ftldrive.backend.numba import serial_ensrf as serial_ensrf_numba


BACKEND_DISPATCH = {}


def backend_registry(key):
    def decorator(func):
        BACKEND_DISPATCH[key] = func
        return func
    return decorator


@backend_registry('python')
def _python_backend(*args, **kwargs):
    return serial_ensrf_python(*args, **kwargs)


@backend_registry('cython')
def _cython_backend(*args, **kwargs):
    return serial_ensrf_cython(*args, **kwargs)


@backend_registry('numba')
def _cython_backend(*args, **kwargs):
    return serial_ensrf_numba(*args, **kwargs)


def serial_ensrf(*args, backend='python', **kwargs):
    """Serial processing ensemble square root filter

    This function dispatches to the appropriate backend function.
    """
    return BACKEND_DISPATCH[backend](*args, **kwargs)
