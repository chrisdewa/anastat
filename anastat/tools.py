import math
import scipy.stats as stats

def calc_za(alfa: float, decimals=2) -> float:
    """
    returns z score for alfa
    """
    return round(stats.norm.ppf(1-alfa/2), decimals)


def calc_zb(beta: float, decimals=2) -> float:
    """
    returns z score for beta
    """
    return round(stats.norm.ppf(1-beta), decimals)
