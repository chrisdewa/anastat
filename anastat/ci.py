# provides functions to work with descriptives

from math import sqrt

import pandas as pd
import scipy.stats as stats

from .tools import calc_za

def mean_ci(series: pd.Series, confidence=0.95):
    """
    calculates the mean with the confidence interval for a series
    asumes normality
    Args:
        series (Series):
            the pandas series to analyse
        confidence (float):
            the confidence level, default 0.95 (95%)
    """
    mean = series.mean()
    SEM = series.sem()
    za = round(calc_za(1-confidence), 2)
    cil = mean - za*SEM
    cih = mean + za*SEM
    return mean, cil, cih


def proportion_ci(p: float, n: int, confidence=0.95) -> tuple[float, float]:
    """
    calculates the confidence interval for a proportion
    asumes normality
    Args:
        p (float): the proportion
        n (int): the sample size
        confidence (float):
            the confidence level, default 0.95 (95%)
    """
    za = calc_za(1-confidence)
    q = 1-p
    SE = sqrt(p*q/n)
    r = za * SE
    return p-r, p+r


