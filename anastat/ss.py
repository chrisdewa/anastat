# formulas for sample size

import math
import scipy.stats as stats
import statsmodels.api as sm

from .tools import calc_za, calc_zb


def one_proportion(
    p: float, 
    alpha: float = 0.05, 
    error: float = 0.1
):
    """
    calculates the sample size for one proportion
    Args:
        p (float):
            the proportion
        alpha (float):
            the significance level
        error (float):
            the permissible error range (beta)
    Returns:
        int: the minimal sample size
    """
    q = 1-p
    za = calc_za(alpha)
    n = ((za**2)*p*q) / (error**2)
    return math.ceil(n)

def two_proportions(p1: float, p2: float, alpha: float = 0.05, beta: float=0.2, error: float = 0.1):
    """
    Sample size for comparing two proportions
    Asumes normality
    """
    Za = calc_za(alpha)
    Zb = calc_zb(beta)
    n = ((p1*(1-p1)+p2*(1-p2))/(p1-p2)**2) * (za + zb)**2
    return math.ceil(n)

def ttest_2sam(
    effect_size: float,
    alpha: float = 0.05,
    beta: float = 0.2,
    ratio: float = 1,
):
    """
    returns the sample size for a two sample ttest
    """
    return math.ceil(
        sm.stats.tt_ind_solve_power(
            effect_size=effect_size,
            alpha=alpha,
            power=1-beta,
            ratio=ratio,
            alternative='two-sided'
        )
    )


def corr(
    expected_r: float,
    alpha: float = 0.05,
    beta: float = 0.2,
):
    C = 0.5 * math.log((1+expected_r)/(1-expected_r))
    za = calc_za(alpha)
    zb = calc_zb(beta)
    n = ((za+zb)/C)**2 + 3
    return math.ceil(n)








