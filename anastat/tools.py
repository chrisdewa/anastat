import math
import pandas as pd
import numpy as np
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

def clean_corr(df: pd.DataFrame, **kws) -> pd.DataFrame:
    """
    returns a cleaned correlation matrix
        meaning without the first column and last row and without
        duplicates
    """
    corr = df.corr(**kws)
    corr = corr.mask(np.tril(np.ones(corr.shape)).astype(bool)).iloc[:-1, 1:]
    return corr

def logit_to_ors(result, alpha=0.05):
    """
    result of a statsmodels.api.Logit model fit
    as an odds ratio dataframe
    """
    ORS = result.conf_int(alpha=alpha)
    ORS['OR'] = result.params
    ORS = np.exp(ORS)
    return ORS
