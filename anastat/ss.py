from math import ceil, exp
import typing as tp

import matplotlib.pyplot as plt
import scipy.stats as stats
import numpy as np
import math

import pingouin as pg 

norm = stats.norm

Tail = tp.Literal['one-sided', 'two-sided']


def Z(alpha: float, beta: float|None = None, *, kind: Tail = 'two-sided') -> float:
    assert kind in {'two-sided', 'one-sided'}, ":kind: can only be 'one-sided' or 'two-sided'"
    assert 0 < alpha < 1, "alpha has to be between 0 and 1"
    if beta is not None:
        assert 0 < beta < 1, "beta has to be between 0 and 1"
    Z = norm.ppf(1-alpha/ 2 if kind == 'two-sided' else 1)
    if beta:
        Z += norm.ppf(1-beta)
    return Z


def aucroc(auc: float, alpha: float=0.05, error: float=0.05) -> int:
    a = stats.norm.ppf(auc) * 1.414
    v_auc = (0.0099*exp(-a**2/2)) * (6*a**2+16)
    z = Z(alpha)
    n = (z**2*v_auc)/error**2
    return ceil(n)


def one_proportion(p, alpha=0.05, error=0.05, *, kind: Tail = 'two-sided'):
    z = Z(alpha, kind=kind)
    q = 1-p
    n = (z*p*q)/error**2
    return ceil(n)


def one_proportion_finite(p, N, alpha=0.05, error=0.05, *, kind: Tail = 'two-sided'):
    z = Z(alpha, kind=kind)
    q = 1-p
    num = N * z**2 * p * q
    den = error**2 * (N-1) + z**2 * p * q
    n = num/den
    return ceil(n)


def one_proportion_test(p, alpha=0.05, beta=0.2, error=0.05, *, kind: Tail = 'two-sided'):
    z = Z(alpha, beta, kind=kind)
    q = 1-p
    n = (z**2 * p * q)/error**2
    return ceil(n)


def one_proportion_test_bad(p, alpha=0.05, beta=0.2, error=0.05, *, kind: Tail = 'two-sided'):
    z = Z(alpha, beta, kind=kind)
    q = 1-p
    n = (z*p*q)/error**2 # w/o squared z
    return ceil(n)


def two_proportions(p1, p2, alpha=0.05, beta=0.2, *, kind: Tail = 'two-sided'):
    z = Z(alpha, beta, kind=kind)
    num = (p1*(1-p1))+(p2*(1-p2))
    den = (p1-p2)**2
    n = num/den * z
    return ceil(n)
    

def one_mean(s, error, alpha=0.05, beta=0.2, *, kind: Tail = 'two-sided'):
    z = Z(alpha, beta, kind=kind)
    n = (z*s/error)**2
    return ceil(n)


def one_mean_test(s, error, alpha=0.05, *, kind: Tail = 'two-sided'):
    z = Z(alpha, kind=kind)
    n = (z*s/error)**2
    return ceil(n)  


def one_mean_test(s, error, alpha=0.05, beta=0.2, *, kind: Tail = 'two-sided'):
    z = Z(alpha, beta, kind=kind)
    n = (z*s/error)**2
    return ceil(n)


def correlation(r, alpha=0.05, beta=0.2, *, kind: Tail = 'two-sided'):
    num = Z(alpha, beta, kind=kind)
    den = np.log((1+r)/(1-r))/2
    n = (num/den)**2 + 3
    return ceil(n)


def survival_single_arm(s_alt, s_null, alpha=0.05, beta=0.2, *, kind: Tail = 'two-sided'):
    """doi: 10.1002/pst.2090"""
    z = Z(alpha, beta, kind=kind)
    tau = 0.25**0.5
    
    num = tau*Z
    den = math.asin(s_alt**0.5)-math.asin(s_null**0.5)
    n = (num/den)**2
    return ceil(n)
    
