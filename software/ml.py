import abc
import numpy as np

from sklearn import linear_model
from matplotlib import pyplot as plt

from models import T_lin, F_lin, T_x_vector, Fy_x_vector
from objects import Data, Reading, EndMill, Prediction

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
        return list(map(data, self.ingest_datum))

    @abc.abstractmethod
    def predict_one(self, conditions):
        """
        Predicts milling forces using the model as it currently is.
        """
        pass

    def predict(self, conditions):
        return list(map(conditions, self.predict_one))


class LinearModel(Model):
    def __init__(self):
        self.training_T_x = list()
        self.training_T_y = list()
        self.training_Fy_x = list()
        self.training_Fy_y = list()
        self.regressor_T = linear_model.Lasso(fit_intercept = False, warm_start=True)
        self.regressor_Fy = linear_model.Lasso(fit_intercept = False, warm_start=True)
        self.params = np.array([0,0,0,0])

    def ingest_datum(self, datum):
        # decompose
        D, W, f_r, w, endmill, Ts, Fys = datum.D, datum.W, datum.f_r, datum.w, datum.endmill, datum.Ts, datum.Fys
        R, N = endmill.R, endmill, N
        T, Fy = np.median(Ts[:, 1]), np.median(Fys[:, 1])

        # get linear coefficients
        T_x = T_x_vector(D, N, R, W, f_r, w)
        Fy_x = Fy_x_vector(D, N, R, W, f_r, w)

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
        K_tc, K_te = self.regressor.coef_
        self.params[0], self.params[1] = K_tc, K_te

        # transform Fy into a smaller linear problem and fit
        intercepts = training_Fy_x @ np.array([K_tc, K_te, 0, 0])[np.newaxis].T
        training_Fy_y_no_intercepts = training_Fy_y - intercepts
        self.regressor_Fy.fit(training_Fy_x[:, 2:], training_Fy_y_no_intercepts)

    def predict_one(self, conditions):
        # decompose
        D, W, f_r, w, endmill = conditions.D, conditions.W, conditions.f_r, conditions.w, conditions.endmill
        R, N = endmill.R, endmill, N

        # evaluate
        T = T_lin(D, N, R, W, f_r, w, self.params[0], self.params[1])
        F = F_lin(D, N, R, W, f_r, w, *self.params)

        # repack and return
        return Prediction(D, W, f_r, w, endmill, T, F)


