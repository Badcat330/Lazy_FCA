from typing import List

from sklearn.preprocessing import MultiLabelBinarizer, LabelBinarizer
import pandas as pd

def binarize_by_percentile(
    X:pd.DataFrame,
    names:List[str], 
    in_line:bool = False,
    quantils:List[float] = [.25, .5, .75]
    ) -> pd.DataFrame:
    
    X_copy = X.copy(deep=True)
    X_copy[names] = X_copy[names].astype(object)
    for name in names:
        percentile = X_copy[name].quantile(quantils).to_list()
        for i, value in enumerate(X_copy[name].iloc):
            if value <= percentile[0]:
                X_copy.at[i, name] = [(0, )]
            elif value <= percentile[1]:
                X_copy.at[i, name] = [(0, 1)]
            elif value <= percentile[2]:
                X_copy.at[i, name] = [(0, 1, 2)]
            else:
                X_copy.at[i, name] = [(0, 1, 2, 3)]
        mlb = MultiLabelBinarizer()
        new_lb = mlb.fit_transform(X_copy[name])
        new_names = [str(name) + '<=' + str(i) for i in quantils] + [str(name) + '<= 1']
        X_copy = pd.concat([X_copy, pd.DataFrame(new_lb, columns=new_names)], axis=1)
        X_copy = X_copy.drop(name, axis=1)
        
    if in_line:
        X = X_copy
        return X
    else:
        return X_copy


def binarize_categorical(X:pd.DataFrame, names:List[str], in_line:bool=False):
    X_copy = X.copy(deep=True)
    for name in names:
        cat = X_copy[name]
        lb = LabelBinarizer()
        new_cat = lb.fit_transform(cat)
        new_names = [str(name) + str(i) for i in range(len(lb.classes_))]
        if len(lb.classes_) != 2:
            X_copy = pd.concat([X_copy, pd.DataFrame(new_cat, columns=new_names)], axis=1)
            X_copy = X_copy.drop(name, axis=1)
        else:
            X_copy = pd.concat([X_copy, pd.DataFrame(new_cat, columns=[name+'new'])], axis=1)
            X_copy = X_copy.drop(name, axis=1)

    if in_line:
        X = X_copy
        return X
    else:
        return X_copy


def binarize_by_range(X:pd.DataFrame, binarize_range:List[float], name:str, in_line:bool=False):
    X_copy = X.copy(deep=True)

    for i, value in enumerate(X_copy[name].iloc):
        for j, treshold in enumerate(binarize_range):
            if value <= treshold:
                X_copy.at[i, name] = j
                break

    X_copy = binarize_categorical(X_copy, [name])

    if in_line:
        X = X_copy
        return X
    else:
        return X_copy
