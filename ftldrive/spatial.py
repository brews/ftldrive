import numpy as np


def haversine_distance(latlon1, latlon2, sphere_radius=6378.137):
    """haversine distance between two sequences of (lat, lon) points

    Parameters
    ----------
    latlon1 : sequence of tuples
        (latitude, longitude) for one set of points.
    latlon2 : sequence of tuples
        A sequence of (latitude, longitude) for another set of points.
    sphere_radius: float
        Radius of sphere we are calculating distances on. Default is 6378.137,
        Earth radius in km.

    Returns
    -------
    dists : 2d array
        An mxn array of Earth haversine distances [1]_ between points in
        latlon1 and latlon2. Units is based on `sphere_radius` used.

    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/Haversine_formula

    """
    latlon1 = np.atleast_2d(latlon1)
    latlon2 = np.atleast_2d(latlon2)

    n = latlon1.shape[0]
    m = latlon2.shape[0]

    paired = np.hstack((np.kron(latlon1, np.ones((m, 1))),
                        np.kron(np.ones((n, 1)), latlon2)))
    latdif = np.deg2rad(paired[:, 0] - paired[:, 2])
    londif = np.deg2rad(paired[:, 1] - paired[:, 3])

    a = (np.sin(latdif / 2)**2 + np.cos(np.deg2rad(paired[:, 0])) 
         * np.cos(np.deg2rad(paired[:, 2])) * np.sin(londif / 2)**2)
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

    return sphere_radius * c


def chordal_distance(latlon1, latlon2, sphere_radius=6378.137):
    """Chordal distance between two sequences of (lat, lon) points

    Parameters
    ----------
    latlon1 : sequence of tuples
        (latitude, longitude) for one set of points.
    latlon2 : sequence of tuples
        A sequence of (latitude, longitude) for another set of points.
    sphere_radius: float
        Radius of sphere we are calculating distances on. Default is 6378.137,
        Earth radius in km.

    Returns
    -------
    dists : 2d array
        An mxn array of Earth chordal distances [1]_ (km) between points in
        latlon1 and latlon2.

    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/Chord_(geometry)

    """
    latlon1 = np.atleast_2d(latlon1)
    latlon2 = np.atleast_2d(latlon2)

    n = latlon1.shape[0]
    m = latlon2.shape[0]

    paired = np.hstack((np.kron(latlon1, np.ones((m, 1))),
                        np.kron(np.ones((n, 1)), latlon2)))

    latdif = np.deg2rad(paired[:, 0] - paired[:, 2])
    londif = np.deg2rad(paired[:, 1] - paired[:, 3])

    a = np.sin(latdif / 2) ** 2
    b = np.cos(np.deg2rad(paired[:, 0]))
    c = np.cos(np.deg2rad(paired[:, 2]))
    d = np.sin(np.abs(londif) / 2) ** 2

    half_angles = np.arcsin(np.sqrt(a + b * c * d))

    dists = 2 * sphere_radius * np.sin(half_angles)

    return dists.reshape(m, n)


def gasparicohn_localization(dists, local_radius):
    """Gaspari-Cohn distance-weights

    Parameters
    ----------
    dists : ndarray
        Distances from a given point.
    local_radius : float
        Distance at which correlation drops to zero.

    Returns
    -------
    weights : ndarray
        Correlations weighted on distance.


    References
    ----------
    Hamill, T. M., Whitaker, J. S., & Snyder, C. (2001). Distance-Dependent Filtering of Background Error Covariance Estimates in an Ensemble Kalman Filter. Monthly Weather Review, 129(11), 2776–2790. https://doi.org/10.1175/1520-0493(2001)129<2776:DDFOBE>2.0.CO;2

    Gaspari Gregory, & Cohn Stephen E. (2006). Construction of correlation functions in two and three dimensions. Quarterly Journal of the Royal Meteorological Society, 125(554), 723–757. https://doi.org/10.1002/qj.49712555417

    """
    weights = np.ones(shape=dists.shape, dtype=np.float64)
    hlr = 0.5 * local_radius # work with half the localization radius
    r = dists / hlr

    weights = np.piecewise(r, condlist=[dists <= hlr, dists > hlr, dists > 2 * hlr],
                 funclist=[_gasparicohn_near, _gasparicohn_far, lambda x: 0])

    # prevent negative values: calc. above may produce tiny negative
    weights[weights < 0] = 0

    return weights


def _gasparicohn_near(x):
    """Support funciton for `gasparicohn_localization`
    """
    return (((-0.25 * x + 0.5) * x + 0.625) * x - (5 / 3)) * (x**2) + 1


def _gasparicohn_far(x):
    """Support function for `gasparicohn_localization`
    """
    return ((((x / 12 - 0.5) * x + 0.625) * x + 5 / 3) * x - 5) * x + 4 - 2 / (3 * x)
