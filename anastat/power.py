import math
from .tools import calc_za


def one_proportion(p, n, alpha=0.05):
  """
  returns the error margin of error for a sample of n size for p proportion
  """
  za = calc_za(alpha)
  q = 1-p
  return math.sqrt(
    (za**2 * p * q)/n
  )
  
  
