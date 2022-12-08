from typing import Iterator, Callable

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_consistent_length, check_array
import pandas as pd
import numpy as np
from tqdm import tqdm

class LazyFCA(BaseEstimator, ClassifierMixin):
    def __init__(
            self, 
            consistency_threshold:float=0.9,
            undefined_treshhold:float=0.8,
            min_extent_size: int = 2, 
            check_number:int=1, 
            numerical_preprocessing:Callable=None) -> None:
        super().__init__()
        self.consistency_threshold = consistency_threshold
        self.undefined_treshhold = undefined_treshhold
        self.min_extent_size = min_extent_size
        self.check_number = check_number
        self.numerical_preprocessing = numerical_preprocessing

    def get_params(self, deep:bool=True):
        return {
            "consistency_threshold": self.consistency_threshold,
            "min_extent_size": self.min_extent_size, 
            "check_all": self.check_all,
            "numerical_preprocessing": self.numerical_preprocessing
        }

    def score(
            self, 
            X_test:np.array,
            y_test:np.array, 
            X_train:np.array, 
            Y_train:np.array) -> float:
        return super().score(X_test, y_test)

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self

    def _compute_instersection(self, x:np.array, x_train:np.array) -> np.array:
        """
        Compute intersection between row from dataset for classification and row from data.
        x: np.array 
            Row from dataset for classification. Should have shape (1, ).
        x_train: 
            Row from test dataset. Should have shape (1, ).
        
        Returns
        -------
        intersaction: np.array
            1-D array containg intersection. Should be use as pattern for finding extent.
        """
        intersection = np.array(x, dtype=object)

        for i in range(x.shape[0]):
            if type(x[i]) is str:
                if x[i] != x_train[i]:
                    intersection[i] = '*'
            else:
                if self.numerical_preprocessing == None:
                    intersection[i] = (min(x[i], x_train[i]), max(x[i], x_train[i]))
                else:
                    intersection[i] = self.numerical_preprocessing(x[i], x_train[i])

        return intersection


    def _compute_extent_target(self, X_train:np.array, Y_train: np.array, intersection: np.array) -> bool:
        """
        Compute extent label. 
        X_train: np.array
            Array of training examples.
        Y_train: np.array
            Array of labels of training examples. Labels should be 0 or 1.
        intersection: np.array
            Intersection that is used as pattern for computing extent. Should have shape (1, )

        Returns
        -------
        target: object
            Return target if extent have persent of this target more then threshold
            otherwise return None, object can't be classified from this extent.
        """
        labels_count = (0, 0)

        for i in range(X_train.shape[0]):
            is_valid = True
            for j in range(X_train.shape[1]):
                if type(X_train[i][j]) is str:
                    if intersection[j] != '*' and X_train[i][j] != intersection[j]:
                        is_valid = False
                        break
                    else:
                        if X_train[i][j] < intersection[j][0] or X_train[i][j] > intersection[j][1]:
                            is_fit = False
                            break

            if is_valid:
                if Y_train[i]:
                    labels_count[1] += 1
                else:
                    labels_count[0] += 0
        
        extent_size = labels_count[0] + labels_count[1]
        if extent_size < self.min_extent_size:
            return None

        if labels_count[0] > labels_count[1]:
            if labels_count[0] / extent_size >= self.consistency_threshold:
                return False
        else:
            if labels_count[1] / extent_size >= self.consistency_threshold:
                return True

        return None


    def predict(self, X:np.array, X_train:np.array, Y_train:np.array, confidence:bool=False, verbose=False) -> Iterator[bool]:
        """
        Predict labels for X dataset base on X_train and Y_train.
        X : np.array
            Data to make prediction for
        X_train: np.array
            Array of training examples
        Y_train: np.array
            Array of labels of training examples
        confidence: bool
            Return confidence of prediction or not.
        verbose: bool
            Show step by step log or not.

        Return
        ------
        prediction: bool
            Python generator with predictions for each x in X[n_train:]
        """
        for i, x in tqdm(
            enumerate(X),
            initial=0, total=X_train.shape[0],
            desc="Predicting data....",
            disable=not verbose
        ):
            pass
