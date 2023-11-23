# módulo para análisis de precisión de pruebas diagnósticas
"""
Glossary:
    crosstab: contingecy table
"""
import math
from typing import NamedTuple

import pandas as pd
import numpy as np
import scipy.stats as stats

from .tools import calc_za

def unpack_crosstab(crosstab: pd.DataFrame) -> float:
    """returns a,b,c,d from a contingency table"""
    a = crosstab.loc[1,1]
    b = crosstab.loc[1,0]
    c = crosstab.loc[0,1]
    d = crosstab.loc[0,0]

    return a, b, c, d

def sensitivity(crosstab: pd.DataFrame) -> float:
    """
    Calculates the sensibility of a test
    Args:
        crosstab (DataFrame):
            2x2 table. Columns is the goldstandard, rows the test.
            1 should be positives, 0 the negatives
    Returns:
        float: the sensitivity
    """
    true_positives = crosstab.loc[1,1]
    diseased = crosstab[1].sum()
    return true_positives / diseased
    
def specificity(crosstab: pd.DataFrame) -> float:
    """
    Calculates the specificity of a test
    Args:
        crosstab (DataFrame):
            2x2 table. Columns is the goldstandard, rows the test.
            1 should be positives, 0 the negatives
    Returns:
        float: the specificity
    """
    true_negatives = crosstab.loc[0,0]
    healthy = crosstab[0].sum()
    return true_negatives / healthy

def ppv(crosstab: pd.DataFrame) -> float:
    """
    Calculates the PPV of a test
    Args:
        crosstab (DataFrame):
            2x2 table. Columns is the goldstandard, rows the test.
            1 should be positives, 0 the negatives
    Returns:
        float: the PPV
    """
    true_positives = crosstab.loc[1,1]
    all_positives = crosstab.loc[1].sum()
    return true_positives / all_positives


def npv(crosstab: pd.DataFrame) -> float:
    """
    Calculates the NPV of a test
    Args:
        crosstab (DataFrame):
            2x2 table. Columns is the goldstandard, rows the test.
            1 should be positives, 0 the negatives
    Returns:
        float: the NPV
    """
    true_negatives = crosstab.loc[0,0]
    all_negatives = crosstab.loc[0].sum()
    return true_negatives / all_negatives


def accuracy(crosstab: pd.DataFrame) -> float:
    """
    Calculates the overall accuracy of a test
    Args:
        crosstab (DataFrame):
            2x2 table. Columns is the goldstandard, rows the test.
            1 should be positives, 0 the negatives
    Returns:
        float: the overall accuracy
    """
    true_positives = crosstab.loc[1,1]
    true_negatives = crosstab.loc[0,0]
    total = crosstab.values.sum()
    return (true_positives + true_negatives) / total

class DOR(NamedTuple):
    OR: float
    cil: float
    cih: float

def dor(crosstab: pd.DataFrame, confidence=0.95) -> DOR:
    """
    calculates the diagnostic ODDs Ratio
    Args:
        crosstab (DataFrame):
            2x2 table. Columns is the goldstandard, rows the test.
            1 should be positives, 0 the negatives
        confidence (float):
            for the confidence interval, defaults to 0.95 (95%)
    Returns:
        DOR: the diagnostic OR (with ci)
    """
    a, b, c, d = unpack_crosstab(crosstab)
    n = crosstab.values.sum()
    
    or_ = (a/c)/(b/d)
    LOR = math.log(or_)
    SE = math.sqrt(1/a+1/b+1/c+1/d)
    za = round(calc_za(1-confidence), 2)
    
    cil = math.exp(LOR - za*SE)
    cih = math.exp(LOR + za*SE)
    
    
    return DOR(or_, cil, cih)

class DiagnosticPerformance(NamedTuple):
    sensitivity: float
    specificity: float
    ppv: float
    npv: float
    accuracy: float

    @classmethod
    def from_crosstab(cls, crosstab: pd.DataFrame) -> float:
        return cls(
            sensitivity=sensitivity(crosstab),
            specificity=specificity(crosstab),
            ppv=ppv(crosstab),
            npv=npv(crosstab),
            accuracy=accuracy(crosstab),
        )














