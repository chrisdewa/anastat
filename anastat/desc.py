import pandas as pd
import scipy.stats as stats

def describe_cuants(df: pd.DataFrame):
    """
    df (DataFrame): must only contain cuantitative variables.
    Returns:
        
    """
    return df.describe().T


def describe_cats(df: pd.DataFrame):
    result = pd.DataFrame()
    categoricals = df.select_dtypes(include='category')
    for cat in categoricals.columns:
        freq = df[cat].value_counts()
        perc = df[cat].value_counts(normalize=True)
        var = pd.DataFrame({
            'Categorías': freq.index,
            'n': freq.values,
            '%': (perc * 100).round(2).values
        })
        var.set_index(['Categorías'], inplace=True)
        var.index = pd.MultiIndex.from_product([[cat], var.index], names=['Variable', 'Categoría'])
    
        result = pd.concat([result, var])
    return result