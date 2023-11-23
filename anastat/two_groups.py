import math
from typing import NamedTuple
import pandas as pd
import numpy as np
import scipy.stats as stats


def cohen_d(x: pd.Series, y: pd.Series) -> float:
    """
    calculates cohen's d for two groups
    Args:
        x (series): group 1
        y (series): group 2
    Returns
        float: cohen's d
    """
    # Calculate the pooled standard deviation
    pooled_std = math.sqrt((x.std() ** 2 + y.std()**2) / 2)

    # Calculate Cohen's d
    d = (x.mean() - y.mean()) / pooled_std

    return d

class TTest2Sam(NamedTuple):
    t: float
    p: float
    d: float # cohen's d

    @classmethod
    def for_ind(cls, x, y) -> 'TTest2Sam':
        t,p = stats.ttest_ind(x, y)
        d = cohen_d(x, y)
        return cls(t, p, d)

    @classmethod
    def for_rel(cls, x, y) -> 'TTest2Sam':
        t,p = stats.ttest_rel(x, y)
        d = cohen_d(x, y)
        return cls(t, p, d)
    

class MannWhitneyU(NamedTuple):
    U: float
    p: float
    r: float

    @classmethod
    def calc(cls, x, y) -> 'MannWhitneyU':
        U, p = stats.mannwhitneyu(x, y)
        r = 1 - (2 * U) / (len(x) * len(y))
        return cls(U, p, r)


class Wilcoxon(NamedTuple):
    s: float
    p: float
    r: float

    @classmethod
    def calc(cls, x, y) -> 'Wilcoxon':
        wres = stats.wilcoxon(x, y, method='approx')
        s = wres.statistic
        p = wres.pvalue
        r = abs(wres.zstatistic/math.sqrt(len(x)))
        return cls(s, p, r)

