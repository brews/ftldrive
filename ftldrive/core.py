from ftldrive.backend.numba import serial_ensrf as serial_ensrf_numba


BACKEND_DISPATCH = {}


def backend_registry(key):
    def decorator(func):
        BACKEND_DISPATCH[key] = func
        return func
    return decorator


@backend_registry('numba')
def _numba_backend(*args, **kwargs):
    return serial_ensrf_numba(*args, **kwargs)


def serial_ensrf(*args, backend='numba', **kwargs):
    """Serial processing ensemble square root filter

    This function dispatches to the appropriate backend function.
    """
    return BACKEND_DISPATCH[backend](*args, **kwargs)
