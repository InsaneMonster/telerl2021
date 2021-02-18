#!/usr/bin/env python3
# filename "common_method.py"

# Global methods
# Contains general methods used everywhere
import numpy as np
from scipy.stats import rice, norm
import phy.common.common_dict as dic


# Fading channel
def fading(typ: str, dim: tuple = (1,),
           shape: float = 6, seed: int = None) -> np.ndarray:
    """Create a sampled fading channel from a given distribution and given
    dimension.

    Parameters
    __________
    typ : str in dic.channel_types,
        type of the fading channel to be used.
    dim : tuple,
        dimension of the resulting array.
    shape : float [dB],
        shape parameters used in the rice distribution modeling the power of
        the LOS respect to the NLOS rays.
    seed : int,
        seed used in the random number generator to provide the same arrays if
        the same is used.
    """
    if typ not in dic.channel_types:
        raise ValueError(f'Type can only be in {dic.channel_types}')
    elif typ == 'AWGN':
        return np.ones(dim)
    elif typ == "Rayleigh":
        vec = norm.rvs(size=2 * np.prod(dim), random_state=seed)
        return (vec[0:2] + 1j * vec[1:2]).reshape(dim)
    elif typ == "Rice":
        return rice.rvs(10 ** (shape / 10), size=np.prod(dim),
                        random_state=seed).reshape(dim)
    elif typ == "Shadowing":
        return norm.rvs(scale=10 ** (shape / 10), random_state=seed)


# Physical noise
def thermal_noise_power(bandwidth, t0=293):
    """Compute the noise power [W] according to bandwidth and ambient temperature.

    Parameters
    ----------
    bandwidth : float
                receiver bandwidth
    t0 : float
         ambient temperature

    Returns
    -------
    out : float
        power of the noise in W
    """
    return 1.38 * 1e-23 * bandwidth * t0  # [W]


# Custom distributions
def circ_uniform(n: int, r_outer: float, r_inner: float = 0,
                 rng: np.random.RandomState = None):
    """Generate n points uniform distributed on an annular region. The outputs
    is given in polar coordinates.

    Parameters
    ----------
    n : int,
        number of points.
    r_outer : float,
        outer radius of the annular region.
    r_inner : float,
        inner radius of the annular region.

    Returns
    -------
    rho : np.ndarray,
        distance of each point from center of the annular region.
    phi : np.ndarray,
        azimuth angle of each point.
    """
    if rng is None:
        rho = np.sqrt((r_outer ** 2 - r_inner ** 2) *
                      np.random.rand(n, 1) + r_inner ** 2)
        phi = 2 * np.pi * np.random.rand(n, 1)
    else:
        rho = np.sqrt((r_outer ** 2 - r_inner ** 2) *
                      rng.rand(n, 1) + r_inner ** 2)
        phi = 2 * np.pi * rng.rand(n, 1)
    return rho, phi


def dbm2watt(dBm):
    """Simply converts dBm to Watt"""
    return 10 ** (dBm / 10 - 3)


def watt2dbm(watt):
    """Simply converts Watt to dBm"""
    return 10 * np.log10(watt * 1e3)


def array2cs(arr: np.array):
    with np.printoptions(precision=2, suppress=True, floatmode="fixed"):
        out: str = ""
        for i in range(arr.size):
            out += np.array2string(arr[i]) + ","
    return out


def missing(ref, new):
    # Store all elements of ref array in a hash table
    s = dict()
    for i in range(len(new)):
        s[new[i]] = 1
    # Return all elements of first array that are not present in hash table
    out = []
    n = 0
    for i in range(len(ref)):
        if ref[i] not in s.keys():
            out.append(ref[i] - n)
            n = n + 1
    return out


def randint_sum_equal_to(sum_value: int, n: int, lower: (int, list) = 0, upper: (int,list) = None):
    """Returns a vector of length n which sum is exactly sum_value and values are random
    integers from lower(i) to upper(i).

    The solution is provided by John McClane and Peter O. from StackOverflow: https://stackoverflow.com/questions/61393463/is-there-an-efficient-way-to-generate-n-random-integers-in-a-range-that-have-a-g
    """
    # Control on input
    if isinstance(lower, (list, np.ndarray)):
        assert len(lower) == n
    else:
        lower = lower * np.ones(n)
    if isinstance(upper, (list, np.ndarray)):
        assert len(upper) == n
    elif upper is None:
        upper = sum_value * np.ones(n)
    else:
        upper = upper * np.ones(n)
    # Trivial solutions
    if np.sum(upper) < sum_value:
        raise ValueError('No solution can be found: sum(upper_bound) < sum_value')
    elif np.sum(lower) > sum_value:
        raise ValueError('No solution can be found: sum(lower_bound) > sum_value')
    elif np.sum(upper) == sum_value:
        return upper
    elif np.sum(lower) == sum_value:
        return lower
    # Setup phase
    # I generate the table t(y,x) storing the relative probability that the sum of y numbers
    # (in the appropriate range) is equal x.
    t = np.zeros((n + 1, sum_value))
    t[0, 0] = 1
    for i in np.arange(1, n + 1):
        # Build the k indexes which are taken for each j following k from 0 to min(u(i-1)-l(i-1), j).
        # This can be obtained creating a repetition matrix of from t[i] multiplied by the triangular matrix
        # tri_mask and then sum each row
        tri_mask = np.tri(sum_value, k=0) - np.tri(sum_value, k=-(upper[i-1] - lower[i-1]))
        t[i] = np.sum(np.repeat(t[i-1][np.newaxis], sum_value, 0)*tri_mask, axis=1)
    # Sampling phase
    values = np.zeros(n)
    s = (sum_value - np.sum(lower)).astype(int)
    for i in np.arange(n)[::-1]:
        # The basic algorithm is the one commented:
        # v = np.round(np.random.rand() * t[i+1, s])
        # r = lower[i]
        # v -= t[i, s]
        # while (v >= 0) and (s > 0):
        #     s -= 1
        #     v -= t[i, s]
        #     r += 1
        # values[i] = r
        # ---------------------------------------------------- #
        # To speed up the convergence I use some numpy tricks.
        # The idea is the same of the Setup phase:
        # - I build a repeat matrix of t[i, s:1];
        # - I take only the lower triangular part, multiplying by a np.tri(s)
        # - I sum over rows, so each element of sum_t contains the cumulative sum of t[i, s - k]
        # - I subtract v - sum_t and count the element greater of equal zero,
        #   which are used to set the output and update s
        v = np.round(np.random.rand() * t[i+1, s])
        values[i] = lower[i]
        sum_t = np.sum(np.repeat(t[i, np.arange(1, s + 1)[::-1]][np.newaxis], s, 0) * np.tri(s), axis=1)
        vt_difference_nonzero = np.sum(np.repeat(v, s) - sum_t >= 0)
        values[i] += vt_difference_nonzero
        s -= vt_difference_nonzero
    return values.astype(int)


def randint_sum_equal_to2(sum_value: int, n: int, lower: int = 0):
    """Returns a vector of a determined size which sum is exactly sum_value and values are random
    integers from lower to sum_value - size*lower.
    THIS IS NOT REALLY RANDOM, BUT IT IS FASTER THAN THE PREVIOUS
    """
    # Trivial solutions
    if np.sum(lower) > sum_value:
        raise ValueError('No solution can be found: sum(lower_bound) > sum_value')
    elif np.sum(lower) == sum_value:
        return lower
    upper = sum_value - n * lower
    values = np.random.random_sample(n)
    values = np.round(values / np.sum(values) * upper) + lower
    delta = sum_value - np.sum(values)
    if delta < 0:
        for i in range(int(abs(delta))):
            values[np.argmax(values).astype(int)] -= 1
    elif delta > 0:
        for i in range(int(abs(delta))):
            values[np.argmin(values).astype(int)] += 1
    return values.astype(int)


def reduce(vec) -> list:
    b = []
    for el in vec:
        if el not in b:
            b.append(el)
    return b
