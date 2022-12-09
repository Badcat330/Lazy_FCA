from typing import Iterator, Callable, Tuple

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_array, check_X_y, check_is_fitted
from sklearn.utils.multiclass import unique_labels
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
        if callable(numerical_preprocessing):
            self.numerical_preprocessing = numerical_preprocessing
        elif numerical_preprocessing == 'min_inf_interval':
            self.numerical_preprocessing = LazyFCA._min_inf_interval
        elif numerical_preprocessing == 'max_inf_interval':
            self.numerical_preprocessing = LazyFCA._min_max_interval
        else:
            self.numerical_preprocessing = LazyFCA._basic_interval
        self.classes_ = None
        self.confidence_ = None

    def get_params(self, deep:bool=True):
        return {
            "consistency_threshold": self.consistency_threshold,
            "min_extent_size": self.min_extent_size, 
            "check_all": self.check_all,
            "numerical_preprocessing": self.numerical_preprocessing
        }

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self

    def fit(self, X, y):
        """
        Because we use Lazy model we don't really need fit method. But for compatibility
        with sklearn interfaces we add it for saving data for future predictions.
        Note: If you use prediction method and give train data to prediction method, 
        data saved after fitting will be ignored and overwrited.

        X_train: array-like
            Array of training examples.
        Y_train: array-like
            Array of labels of training examples. Labels should be 0 or 1.
        
        Return
        ------
        self:
            Return self for onelines.
        """
        X, y = check_X_y(X, y)
        self.classes_ = unique_labels(y)
        self.X_ = X
        self.y_ = y
        return self

    @staticmethod
    def _basic_interval(a: float, b: float) -> Tuple[float, float]:
        return (min(a, b), max(a, b))

    @staticmethod
    def _min_inf_interval(a: float, b: float) -> Tuple[float, float]:
        return (min(a, b), float('inf'))

    @staticmethod
    def _min_max_interval(a: float, b: float) -> Tuple[float, float]:
        return (max(a, b), float('inf'))

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
            if type(x[i]) is str or type(x[i]) is bool:
                if x[i] != x_train[i]:
                    intersection[i] = '*'
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
        positive_count = 0
        negative_count = 0

        for i in range(X_train.shape[0]):
            is_valid = True
            for j in range(X_train.shape[1]):
                if type(X_train[i][j]) is str or type(X_train[i][j]) is bool:
                    if intersection[j] != '*' and X_train[i][j] != intersection[j]:
                        is_valid = False
                        break
                else:
                    if X_train[i][j] < intersection[j][0] or X_train[i][j] > intersection[j][1]:
                        is_valid = False
                        break

            if is_valid:
                if Y_train[i]:
                    positive_count += 1
                else:
                    negative_count += 1
        
        extent_size = positive_count + negative_count
        if extent_size < self.min_extent_size:
            return None

        if negative_count > positive_count:
            if negative_count / extent_size >= self.consistency_threshold:
                return False
        else:
            if positive_count / extent_size >= self.consistency_threshold:
                return True

        return None


    def predict(
        self, 
        X:np.array, 
        X_train:np.array=None, 
        Y_train:np.array=None, 
        confidence:bool=False, 
        verbose=False
        ) -> np.array:
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
        prediction: np.array
            Python generator with predictions for each x in X[n_train:]. 
            If label can't be predict return None.
        """
        if X_train is None or Y_train is None:
            check_is_fitted(self)
            X_train = self.X_
            Y_train = self.y_
        else:
            X_train, Y_train = check_X_y(X_train, Y_train, dtype=object)
            X = check_array(X, dtype=object)
            self.classes_ = unique_labels(Y_train)

        if len(self.classes_) < 2 or len(self.classes_) > 2:
            raise ValueError
         
        # Binarised Y_train
        if self.classes_[0] not in {0, 1} or self.classes_[1] not in {0, 1}:
            Y_train = np.where(Y_train==self.classes_[0], 0, 1)
        
        prediction = np.empty(X.shape[0], dtype=np.bool_)
        self.confidence_ = np.array(prediction)
        for i, x in tqdm(
            enumerate(X),
            initial=0, total=len(X),
            desc="Predicting data....",
            disable=not verbose
        ):
            number_checked = 0
            negative_count = 0
            positive_count = 0
            for x_train in X_train:
                # Try to predict base on intersection of i train data row
                intersection = self._compute_instersection(x, x_train)
                extent_target = self._compute_extent_target(X_train, Y_train, intersection)
                if extent_target is None:
                    continue
                elif extent_target:
                    positive_count += 1
                    number_checked += 1
                else:
                    negative_count += 1
                    number_checked += 1

                # If enough predictions stop
                if number_checked >= self.check_number:
                    break
            
            # If enough targets predicted count avg prediction and save confidence if needed
            if positive_count + negative_count >= self.check_number:
                if negative_count > positive_count:
                    confidence_value = negative_count / number_checked
                    if confidence_value > self.consistency_threshold:
                        if confidence:
                            self.confidence_[i] = confidence_value / number_checked
                        prediction[i] = self.classes_[0]
                        continue
                elif positive_count > negative_count:
                    confidence_value = positive_count / number_checked
                    if confidence_value > self.consistency_threshold:
                        if confidence:
                            self.confidence_[i] = confidence_value / number_checked
                        prediction[i] = self.classes_[1]
                        continue
            prediction[i] = None
            if confidence:
                self.confidence_[i] = None

        return prediction
            