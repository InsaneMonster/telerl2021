#!/usr/bin/env python3
# file: waterfilling.py

import numpy as np
import warnings
from phy.scenario.resources import Resources
from phy.common.common_method import dbm2watt


def time_freq_ra(chan: np.ndarray,
                 user: list,
                 RB: Resources = None) -> list:
    """Heuristic max rate algorithm with min rate requirements in time freq.

    Parameters
    ----------
    chan : numpy.ndarray
        array of channel gain needed; chan.shape must be (F, K) where F is the
        number of frequency resource available and K is the number of users.
    user : list
        list of instance of User to be allocated; user.shape must be (K,).
    RB : Resources
        resources available for the allocation.

    Returns
    _______
    out : list(np.ndarray, np.ndarray)
        tuple of power coefficients p and allocation variable x:
        - p.shape and x.shape are (T, F, K) where p[t, f, k] is the power
            coefficient (float) of time mini-slot t, frequency f and user k;
        - x[t, f, k] is the allocation variable (bool) of time mini-slot t,
            frequency f and user k;
        It is assumed that the max_power available (at BS or UL) is a
        constraint for each time mini-slot, and the min rate requirements is
        averaged for each mini-time slot.
    """
    # Control on input
    assert isinstance(chan, np.ndarray), 'chan must be a ndarray'
    if chan.shape[3:] > (1, 1) or chan.shape[3:] == ():
        raise ValueError('This function is meant to be used for SISO '
                         'communication only')
    if RB is None:
        T = 1
    else:
        T = RB.slot
    # Dimensions
    F = chan.shape[0]
    K = len(user)
    # Collect variables -------------------------------------------------------
    att = np.array([(n.ord, n.useful.ord, n.useful.noise.linear)
                    if n.dir == 'UL' else (n.useful.ord, n.ord, n.noise.linear)
                    for n in user])
    t = np.array(att[:, 0], dtype=int)
    r = np.array(att[:, 1], dtype=int)
    h = np.real(np.conj(chan[:, t, r]) * chan[:, t, r]).reshape(F, K) \
        / np.repeat(att[:, 2][np.newaxis], F, axis=0)
    # User data
    w = []
    UL = []
    DL = []
    BS = []
    r_min = []
    for n in user:
        r_min.append(n.QoS[0])
        w.append(n.weight)
        if n.dir == 'UL':
            UL.append(n)
        elif n.type == 'UE' and n.dir == 'DL':
            if not np.isin(BS, n.useful).any():
                BS.append(n.useful)
                DL.append([x for x in BS[-1].useful if x.dir == 'DL'])
    # Bounds
    pow_max = np.array([dbm2watt(n.max_pow) for n in np.hstack((UL, BS))])
    # Minimum rate requirements
    data_min = np.array(r_min) * T
    # Weight
    weight = np.repeat(np.array(w)[np.newaxis], F, axis=0)
    # Partitioning of the user having the same power constraint
    P = len(UL) + len(BS)
    part = np.zeros((P, F, K), dtype=bool)
    # Each uplink user has a different power constraint
    part[np.arange(len(UL)), :, np.isin(user, UL)] = 1
    # Each group of downlink user associated to a BS has a single power constraint
    for x in range(len(BS)):
        part[x + len(UL), :, np.isin(user, DL[x])] = 1
    # Init RA -----------------------------------------------------------------
    # Resource allocation is developed for eMBB users which are allocated on entire time slot.
    x = np.zeros((T, F, K), dtype=bool)
    p = np.zeros((T, F, K))
    r = np.zeros((T, F, K))

    feasible = False
    for t in range(T):
        # control minimum rate
        if (np.sum(r, axis=(0, 1)) >= data_min).all():
            h_test = h
            feasible = True
        else:
            xk = np.sum(r, axis=(0, 1)) < data_min
            h_test = np.zeros(h.shape)
            h_test[:, xk] = h[:, xk]
        x[t, np.arange(F), np.argmax(h_test, axis=1)] = 1
        # power allocation
        for i in range(P):
            if (part[i] * x[t]).any():
                p[t, part[i] * x[t]] = geo_wf(weight[part[i] * x[t]],
                                              h[part[i] * x[t]],
                                              pow_max[i])
        r[t] = weight * np.log2(1 + h * p[t])
    max_iter = 50
    it = 1
    while ~feasible:
        strong = np.argmax(np.sum(r, axis=(0, 1)))
        weak = np.argmin(np.sum(r, axis=(0, 1)))
        t, f, _ = np.where(r == np.min(r[:, :, strong][x[:, :, strong]]))
        t = t[0]
        f = f[0]
        x[t, f, strong] = 0
        x[t, f, weak] = 1
        p[t] = p[t] * x[t]
        for i in range(P):
            if (part[i] * x[t]).any():
                p[t, part[i] * x[t]] = geo_wf(weight[part[i] * x[t]],
                                              h[part[i] * x[t]],
                                              pow_max[i])
        r[t] = weight * np.log2(1 + h * p[t])
        if (np.sum(r, axis=(0, 1)) >= data_min).all():
            break
        elif it >= max_iter:
            warnings.warn(f'Unfeasible solution after {max_iter} iterations')
            break
        else:
            it += 1
    return [r, p, x]


def geo_wf(weight: np.ndarray, chan: np.ndarray, bound: float) -> np.ndarray:
    """Weighted geometric water filling algorithm.

    Parameters
    __________
    weight : np.ndarray 1-D
        user specific weight for the sum rate; weight.shape must be equal to
        chan.shape.
    chan :  np.ndarray 1-D
        channel gains.
    bound : float
        maximum power available (water level).

    Returns
    _______
    p : numpy.ndarray 1-D
        power allocated for each channel; p.shape = chan.shape
    """
    if chan.shape != weight.shape:
        raise ValueError('chan.shape and weight.shape must be equal')
    # Init
    K = chan.shape[0]
    p = np.zeros(K)
    # Ordering channels
    order = np.flip(np.argsort(weight * chan))
    d = 1 / weight[order] / chan[order]
    # The loop commented is replaced by numpy operations ----------------------
    # P_i = np.zeros(len(chan))
    # for i in range(len(chan)):
    #     P_i[i] = bound - np.sum((d[i] - d[:i]) * weight[:i])
    # -------------------------------------------------------------------------
    D = np.repeat(d * weight[np.newaxis], K, axis=0)[::-1]
    w = np.repeat(weight[np.newaxis], K, axis=0)[::-1]
    D[np.triu_indices(K)] = 0
    w[np.triu_indices(K)] = 0
    P_i = bound - d * np.sum(w, axis=1) + np.sum(D, axis=1)
    opt = max(np.flatnonzero(P_i > 0))
    p[:opt + 1] = (P_i[opt] / sum(weight[:opt + 1]) +
                   d[opt] - d[:opt + 1]) * weight[:opt + 1]
    return p[order.argsort()]
