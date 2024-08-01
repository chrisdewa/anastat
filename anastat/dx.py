# módulo para análisis de precisión de pruebas diagnósticas
"""
Glossary:
    crosstab: contingecy table
"""
import math
from typing import NamedTuple, Self

import pandas as pd
import numpy as np
import scipy.stats as stats

from numpy.typing import NDArray

from .ss import Z

def make_table(a: int, b: int, c: int, d: int) -> NDArray:
    """
    returns a confusion matrix for dianosis analysis
    
    Args:
        a: True positives
        b: False positives
        c: False negatives
        d: True negatives
    """
    return np.array([[a, b], [c, d]])


def unpack_table(table: NDArray) -> tuple[float]:
    """returns a,b,c,d from a contingency table"""
    a = table[0, 0]
    b = table[0, 1]
    c = table[1, 0]
    d = table[1, 1]

    return a, b, c, d


def sensitivity(true_positives, false_negatives) -> float:
    """
    Calculates the sensibility of a test

    Returns:
        float: the sensitivity
    """
    return true_positives / (true_positives + false_negatives)
    
def specificity(true_negatives, false_positives) -> float:
    """
    Calculates the specificity of a test
    
    Returns:
        float: the specificity
    """
    
    return true_negatives / (true_negatives + false_positives)


def ppv(true_positives, false_positives) -> float:
    """
    Calculates the observed positive predictive value of a test

    Returns:
        float: the PPV
    """
    return true_positives / (true_positives + false_positives)


def npv(true_negatives, false_negatives) -> float:
    """
    Calculates the NPV of a test

    Returns:
        float: the NPV
    """
    return true_negatives / (true_negatives + false_negatives)



def accuracy(table: NDArray) -> float:
    """
    Calculates the overall accuracy of a test

    Args:
        crosstab (DataFrame):
            2x2 table. Columns is the goldstandard, rows the test.
            1 should be positives, 0 the negatives
    Returns:
        float: the overall accuracy
    """

    true_positives = table[1,1]
    true_negatives = table[0,0]
    total = table.sum()

    return (true_positives + true_negatives) / total

class DOR(NamedTuple):
    "Diagnostic odds ratio"
    OR: float
    cil: float
    cih: float

def dor(table: NDArray, confidence=0.95) -> DOR:
    """
    calculates the diagnostic ODDs Ratio
    Args:
        table (NDArray):
            2x2 table. columns are status, rows test result.

        confidence (float):
            for the confidence interval, defaults to 0.95 (95%)
    Returns:
        DOR: the diagnostic OR (with ci)
    """
    a, b, c, d = unpack_table(table)
    n = table.sum
    
    or_ = (a/c)/(b/d)
    LOR = math.log(or_)
    SE = math.sqrt(1/a+1/b+1/c+1/d)
    za = Z(1-confidence)
    
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
    def from_table(cls, table: NDArray) -> Self:
        a, b, c, d = unpack_table(table)
        return cls(
            sensitivity=sensitivity(a, c),
            specificity=specificity(d, b),
            ppv=ppv(a, b),
            npv=npv(d, c),
            accuracy=accuracy(table),
        )














