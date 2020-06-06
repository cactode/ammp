import abc
import numpy as np

from sklearn import linear_model
from matplotlib import pyplot as plt

from models import T_lin, F_lin, T_x_vector, T_x_vector_padded, Fy_x_vector
from objects import Data, Conditions, EndMill, Prediction


class Model(abc.ABC):
    """
    Represents a model that can be trained with one training datum at a time.
    """
    @abc.abstractmethod
    def ingest_datum(self, datum):
        """
        Ingests one datum.
        Args:
            datum : A Data object.
        """
        pass

    def ingest_data(self, data):
        for datum in data:
            self.ingest_datum(datum)

    @abc.abstractmethod
    def predict_one(self, conditions):
        """
        Predicts milling forces using the model as it currently is.
        """
        pass

    def predict(self, conditions):
        return list(map(self.predict_one, conditions))


class LinearModel(Model):
    def __init__(self):
        self.training_T_x = list()
        self.training_T_y = list()
        self.training_Fy_x = list()
        self.training_Fy_y = list()

        self.regressor_T = linear_model.LinearRegression(fit_intercept=False)
        self.regressor_Fy = linear_model.LinearRegression(fit_intercept=False)
        self.params = np.array([0, 0, 0, 0], dtype='float64')

    def ingest_datum(self, datum):
        # decompose
        _, _, _, _, _, Ts, Fys = datum.unpack()
        T, Fy = np.median(Ts[:, 1]), np.median(Fys[:, 1])
        # get linear coefficients
        T_x = T_x_vector(datum.conditions())
        Fy_x = Fy_x_vector(datum.conditions())

        # add to training set
        self.training_T_x.append(T_x)
        self.training_T_y.append(T)
        self.training_Fy_x.append(Fy_x)
        self.training_Fy_y.append(Fy)
        self.update()

    def update(self):
        # convert force into numpy arrays for convenience
        training_Fy_x = np.array(self.training_Fy_x)
        training_Fy_y = np.array(self.training_Fy_y)

        # calculate best fit from data
        self.regressor_T.fit(self.training_T_x, self.training_T_y)
        K_tc, K_te = self.regressor_T.coef_
        self.params[0], self.params[1] = K_tc, K_te

        # transform Fy into a smaller linear problem and fit
        intercepts = training_Fy_x @ np.array([K_tc, K_te, 0, 0])[np.newaxis].T
        training_Fy_y_no_intercepts = np.reshape(
            training_Fy_y - intercepts.T, (-1))
        self.regressor_Fy.fit(
            training_Fy_x[:, 2:], training_Fy_y_no_intercepts)
        K_rc, K_re = self.regressor_Fy.coef_
        self.params[2], self.params[3] = K_rc, K_re

    def predict_one(self, conditions):
        # evaluate
        T = T_lin(conditions, *self.params[:2])
        F = F_lin(conditions, *self.params)

        # repack and return
        return Prediction(*conditions.unpack(), T, F)

class UnifiedLinearModel(Model):
    def __init__(self):
        self.training_Tx = list()
        self.training_Ty = list()
        self.training_Fyx = list()
        self.training_Fyy = list()

        self.regressor = linear_model.LinearRegression(fit_intercept=False)
        self.params = np.array([0, 0, 0, 0], dtype='float64')

    def ingest_datum(self, datum):
        # decompose
        _, _, _, _, _, Ts, Fys = datum.unpack()
        T, Fy = np.median(Ts[:, 1]), np.median(Fys[:, 1])
        # get linear coefficients
        T_x = np.array(T_x_vector_padded(datum.conditions()))
        Fy_x = np.array(Fy_x_vector(datum.conditions()))

        # we want to artificially inflate T to be as big as F
        # this is a little arbitrary, might not be the best idea lol
        ratio = Fy_x[:2].mean() / T_x[:2].mean()
        T_x *= ratio
        T *= ratio

        # add T to training set
        self.training_Tx.append(T_x)
        self.training_Ty.append(T)

        # add Fy to training set
        self.training_Fyx.append(Fy_x)
        self.training_Fyy.append(Fy)

        self.update()

    def update(self):
        # calculate best fit from data
        self.regressor.fit(self.training_Tx + self.training_Fyx, self.training_Ty + self.training_Fyy)
        self.params = np.array(self.regressor.coef_)

    def predict_one(self, conditions):
        # evaluate
        T = T_lin(conditions, *self.params[:2])
        F = F_lin(conditions, *self.params)

        # repack and return
        return Prediction(*conditions.unpack(), T, F)

class RANSACLinearModel(Model):
    def __init__(self):
        self.training_Tx = list()
        self.training_Ty = list()
        self.training_Fyx = list()
        self.training_Fyy = list()

        base_regressor = linear_model.LinearRegression(fit_intercept=False)
        self.regressor = linear_model.RANSACRegressor(base_regressor, min_samples=5)
        self.params = np.array([0, 0, 0, 0], dtype='float64')

    def ingest_datum(self, datum):
        # decompose
        _, _, _, _, _, Ts, Fys = datum.unpack()
        T, Fy = np.median(Ts[:, 1]), np.median(Fys[:, 1])
        # get linear coefficients
        T_x = np.array(T_x_vector_padded(datum.conditions()))
        Fy_x = np.array(Fy_x_vector(datum.conditions()))

        # we want to artificially inflate T to be as big as F
        # this is a little arbitrary, might not be the best idea lol
        ratio = Fy_x[:2].mean() / T_x[:2].mean()
        T_x *= ratio
        T *= ratio

        # add T to training set
        self.training_Tx.append(T_x)
        self.training_Ty.append(T)

        # add Fy to training set
        self.training_Fyx.append(Fy_x)
        self.training_Fyy.append(Fy)

        if (len(self.training_Tx) + len(self.training_Fyx)) > 4:
            self.update()

    def update(self):
        # calculate best fit from data
        self.regressor.fit(self.training_Tx + self.training_Fyx, self.training_Ty + self.training_Fyy)
        self.params = np.array(self.regressor.estimator_.coef_)

    def predict_one(self, conditions):
        # evaluate
        T = T_lin(conditions, *self.params[:2])
        F = F_lin(conditions, *self.params)

        # repack and return
        return Prediction(*conditions.unpack(), T, F)

