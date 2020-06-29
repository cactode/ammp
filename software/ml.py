import abc
import numpy as np

from sklearn import linear_model
from scipy import stats
from matplotlib import pyplot as plt

from models import T_lin, F_lin, T_lin_full, F_lin_full, T_x_vector, T_x_vector_padded, Fy_x_vector, T_x_vector_full, Fy_x_vector_full
from objects import Data, Conditions, EndMill, Prediction

# https://stackoverflow.com/questions/11686720
def mean_no_outliers(data, m=2):
    d = np.abs(data - np.median(data))
    mdev = np.median(d)
    s = d / (mdev if mdev else 1.)
    return np.mean(data[s < m])

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
    def __init__(self, initial_params = [0, 0, 0, 0]):
        self.training_T_x = list()
        self.training_T_y = list()
        self.training_Fy_x = list()
        self.training_Fy_y = list()

        self.regressor_T = linear_model.LinearRegression(fit_intercept=False)
        self.regressor_Fy = linear_model.LinearRegression(fit_intercept=False)
        self.params = initial_params

    def ingest_datum(self, datum):
        # decompose
        _, _, _, _, _, Ts, Fys = datum.unpack()
        T, Fy = mean_no_outliers(Ts[:, 1]), mean_no_outliers(Fys[:, 1])
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
        T = T_lin(conditions, *self.params)
        F = F_lin(conditions, *self.params)

        # repack and return
        return Prediction(*conditions.unpack(), T, F)

class UnifiedLinearModel(Model):
    def __init__(self, initial_params = [0,0,0,0]):
        self.training_Tx = list()
        self.training_Ty = list()
        self.training_Fyx = list()
        self.training_Fyy = list()

        self.regressor = linear_model.LinearRegression(fit_intercept=False)
        self.params = initial_params

    def ingest_datum(self, datum):
        # decompose
        _, _, _, _, _, Ts, Fys = datum.unpack()
        # mean filter + rejection of outliers
        # T, Fy = np.median(Ts[:, 1]), np.median(Fys[:, 1])
        T, Fy = mean_no_outliers(Ts[:, 1]), mean_no_outliers(Fys[:, 1])
        # get linear coefficients
        T_x = np.array(T_x_vector_padded(datum.conditions()))
        Fy_x = np.array(Fy_x_vector(datum.conditions()))

        # normalizing independently while preserving ratios
        norm_T = np.linalg.norm(T_x)
        T_x /= norm_T
        T /= norm_T
        norm_Fy = np.linalg.norm(Fy_x)
        Fy_x /= norm_Fy
        Fy /= norm_Fy

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
        T = T_lin(conditions, *self.params)
        F = F_lin(conditions, *self.params)

        # repack and return
        return Prediction(*conditions.unpack(), T, F)

    def score(self):
        return self.regressor.score(self.training_Tx + self.training_Fyx, self.training_Ty + self.training_Fyy)


class UnifiedLinearModelFull(Model):
    def __init__(self, initial_params = [0,0,0,0,0,0]):
        self.training_Tx = list()
        self.training_Ty = list()
        self.training_Fyx = list()
        self.training_Fyy = list()

        self.regressor = linear_model.LinearRegression(fit_intercept=False)
        self.params = initial_params

    def ingest_datum(self, datum):
        # decompose
        _, _, _, _, _, Ts, Fys = datum.unpack()
        # mean filter + rejection of outliers
        # T, Fy = np.median(Ts[:, 1]), np.median(Fys[:, 1])
        T_f, Fy_f = reject_outliers(Ts[:, 1]), reject_outliers(Fys[:, 1])
        T, Fy = np.mean(T_f), np.mean(Fy_f)
        # get linear coefficients
        T_x = np.array(T_x_vector_full(datum.conditions()))
        Fy_x = np.array(Fy_x_vector_full(datum.conditions()))

        # normalizing independently while preserving ratios
        norm_T = np.linalg.norm(T_x)
        T_x /= norm_T
        T /= norm_T
        norm_Fy = np.linalg.norm(Fy_x)
        Fy_x /= norm_Fy
        Fy /= norm_Fy

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
        T = T_lin_full(conditions, *self.params)
        F = F_lin_full(conditions, *self.params)

        # repack and return
        return Prediction(*conditions.unpack(), T, F)

class RANSACLinearModel(Model):
    def __init__(self, initial_params = [0,0,0,0]):
        self.training_Tx = list()
        self.training_Ty = list()
        self.training_Fyx = list()
        self.training_Fyy = list()

        base_regressor = linear_model.LinearRegression(fit_intercept=False)
        self.regressor = linear_model.RANSACRegressor(base_regressor, min_samples=5)
        self.params = initial_params

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
        T = T_lin(conditions, *self.params)
        F = F_lin(conditions, *self.params)

        # repack and return
        return Prediction(*conditions.unpack(), T, F)

class BayesianLinearModel(Model):
    """
    Utilizes a Bayesian approach to pick "safe" values for the coefficients.
    That is, we have a confidence interval for our parameters. We pick the upper end for
        each coefficient so we don't make too aggressive of a cut
    """
    def __init__(self, initial_params = [0,0,0,0], percentile = .90):
        self.training_Tx = list()
        self.training_Ty = list()
        self.training_Fyx = list()
        self.training_Fyy = list()

        self.regressor = linear_model.ARDRegression(fit_intercept=False, tol = 1e-6, alpha_1 = 1e-9, alpha_2 = 1e-9, lambda_1 = 1e-9, lambda_2 = 1e-9)
        self.params = initial_params
        self.zscore = stats.norm.ppf(percentile)

    def ingest_datum(self, datum):
        # decompose
        _, _, _, _, _, Ts, Fys = datum.unpack()
        T, Fy = np.median(Ts[:, 1]), np.median(Fys[:, 1])
        # get linear coefficients
        T_x = np.array(T_x_vector_padded(datum.conditions()))
        Fy_x = np.array(Fy_x_vector(datum.conditions()))

        # normalize while preserving ratio between x and y
        norm_T = np.linalg.norm(T_x)
        T_x /= norm_T
        T /= norm_T
        norm_Fy = np.linalg.norm(Fy_x)
        Fy_x /= norm_Fy
        Fy /= norm_Fy

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

        # get params and variance matrix, convert to std deviation
        param_mean = np.array(self.regressor.coef_)
        param_stdv = np.sqrt(np.diag(self.regressor.sigma_))
        print("Param mean: ", param_mean)
        print("param stdv: ", param_stdv)

        # set params to the lower end of our confidence interval, but make sure they're above 0
        self.params = param_mean

    def predict_one(self, conditions):
        # evaluate
        T = T_lin(conditions, *self.params)
        F = F_lin(conditions, *self.params)

        # repack and return
        return Prediction(*conditions.unpack(), T, F)

