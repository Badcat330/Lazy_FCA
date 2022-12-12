import numpy as np
from sklearn.utils.multiclass import unique_labels
from sklearn.utils.validation import check_consistent_length, check_array


def accuracy_undefine_score(y_test: np.array, predict: np.array) -> float:
    """
    Count accuracy for prediction with undefine lebels represented as None.

    Parameters
    ----------
    y_test: np.array
        1-D np.array with correct target.
    predict: np.array
        1-D np.array with predicted target.

    Returns
    -------
    accuracy score: float
        Accuracy score for y_test and predict.
    """
    y_test = check_array(y_test, dtype=object, ensure_2d=False)
    predict = check_array(predict, dtype=object, ensure_2d=False)
    check_consistent_length(y_test, predict)
    return np.sum(y_test == predict) / predict.shape[0]


def recall_undefine_score(
        y_test: np.array,
        predict: np.array,
        pos_label: str or int = 1) -> float:
    """
    Count recall for prediction with undefine lebels represented as None.
    Show what propostion of true positive correctly classified.

    Parameters
    ----------
    y_test: np.array
        1-D np.array with correct target.
    predict: np.array
        1-D np.array with predicted target.
    pos_label: str or int
        Positive label. Default will take 1 label from unique_labels util.

    Returns
    -------
    recall score: float
        Recall score for y_test and predict.
    """
    check_consistent_length(y_test, predict)
    y_test = check_array(y_test, dtype=object, ensure_2d=False)
    predict = check_array(predict, dtype=object, ensure_2d=False)
    labels = np.unique(y_test)
    if type(pos_label) is int:
        label = labels[pos_label]
    else:
        label = pos_label

    return np.sum((y_test == predict) & (predict == label)) / np.sum(y_test == label)


def precision_undefine_score(
        y_test: np.array,
        predict: np.array,
        pos_label: str or int = 1) -> float:
    """
    Count precision for prediction with undefine lebels represented as None.
    Show what proportion of classified as positive is truly positive.

    Parameters
    ----------
    y_test: np.array
        1-D np.array with correct target.
    predict: np.array
        1-D np.array with predicted target.
    pos_label: str or int
        Positive label. Default will take 1 label from unique_labels util.

    Returns
    -------
    precision score: float
        Precision score for y_test and predict.
    """
    check_consistent_length(y_test, predict)
    y_test = check_array(y_test, dtype=object, ensure_2d=False)
    predict = check_array(predict, dtype=object, ensure_2d=False)
    labels = np.unique(y_test)
    if type(pos_label) is int:
        label = labels[pos_label]
    else:
        label = pos_label
    return np.sum((y_test == predict) & (predict == label)) / np.sum(predict == label)


def f1_undefine_score(
        y_test: np.array,
        predict: np.array,
        pos_label: str or int = 1) -> float:
    """
    Count f1 score for prediction with undefine lebels represented as None.
    Maximizing this score find balance between find as many positive labels in
    data as posible and all finded labels shoul be truly positive.

    Parameters
    ----------
    y_test: np.array
        1-D np.array with correct target.
    predict: np.array
        1-D np.array with predicted target.
    pos_label: str or int
        Positive label. Default will take 1 label from unique_labels util.

    Returns
    -------
    f1 score: float
        F1 score for y_test and predict.
    """
    check_consistent_length(y_test, predict)
    recall = recall_undefine_score(y_test, predict, pos_label)
    precision = precision_undefine_score(y_test, predict, pos_label)
    f1 = 2 * (precision * recall) / (precision + recall)
    if np.isnan(f1):
        return 0
    else:
        return f1
