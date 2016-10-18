"""
Uses python introspection to call all function in `data.load_datasets`

Written by Gilles Vandewiele in commission of IDLab - INTEC from University Ghent.
"""

import data.load_datasets
from inspect import getmembers, isfunction


def load_all_datasets():
    """
    Uses python introspection to call all function in `data.load_datasets`

    **Returns**
    -----------
    a list of loaded datasets
    """
    datasets = []
    for o in getmembers(data.load_datasets):
        if isfunction(o[1]):
            df, feature_cols, label_col, name = o[1]()
            datasets.append({'dataframe': df, 'feature_cols': feature_cols, 'label_col': label_col, 'name': name})

    return datasets