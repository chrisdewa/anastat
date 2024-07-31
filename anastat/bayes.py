from dataclasses import dataclass
from typing import Literal

Result = Literal['positive', 'negative']

@dataclass
class Test:
    name: str
    sensitivity: float
    specificity: float

class BayesianDiagnosis:
    def __init__(self, prevalence):
        self.lks = {'basal': (prevalence, 1-prevalence)}
        self.table = pd.DataFrame(index=['positive', 'negative'])
        self.table['basal'] = prevalence, 1-prevalence

    def add_test(self, test: Test, *, result: Result):
        prior = self.table.iloc[:, -1]['positive']
        args = test.sensitivity, test.specificity, prior
        if result == 'positive':
            post = self.ppv(*args) 
            evidence = post, 1-post
        else:
            post = self.npv(*args)
            evidence = 1-post, post
        self.table[test.name] = evidence
        return self.table.copy()

    @staticmethod
    def ppv(sens, esp, prior):
        num = sens*prior
        den = num+(1-esp)*(1-prior)
        return num/den
        
    @staticmethod
    def npv(sens, esp, prior):
        num = esp*(1-prior)
        den = num+(1-sens)*prior
        return num/den