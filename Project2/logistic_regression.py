import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sn
from gradient_descent import sgd

class LogisticRegression:
    """
    Class for logistic regression with two output classes.
    """
    def __init__(self, penalty):
        """
        Arguments:
            penalty (float): penalty parameter
        """
        self.penalty = penalty

        # set default learning parameters
        self.a = 1e-3
        self.b = 1e0

    def C(self, X, y, beta):
        """
        Cost function of logistic regression.
        Arguments:
            X (array): design matrix
            y (array): response
            beta (array): coefficient estimate
        Returns:
            C (float): cost function value
        """
        a = X@beta

        C = -np.sum(y*a - np.log(1+exp(a))) + self.penalty*(beta.T)@beta

        return C

    def del_C(self, x, beta):
        """
        Gradient of the cost function for logistic regression.
        Arguments:
            x (array): array containing design matrix and response
            beta (array): coefficient estimates
        Returns:
            del_C (float): gradient of cost function
        """
        y = x[:,-1]
        X = x[:,:-1]

        a = X@beta
        b = np.exp(a-np.max(a))
        p = (b)/(1+b)

        del_C = -(X.T)@(y-p) + 2*self.penalty*beta

        return del_C

    def set_learning_params(self, a, b):
        """
        Change the parameters for the learning schedule.
        Arguments:
            a (float): parameter 1
            b (float): parameter 2
        """
        self.a = a
        self.b = b

    def gamma(self, j):
        """
        Learning schedule for the stochastic gradient descent.
        Arguments:
            j (int): index of iteration
        Returns:
            gamma (float): learning parameter for specified index
        """
        gamma = self.a / (1 + self.b*j)

        return gamma

    def fit(self, X, y, n_minibatches, n_epochs, std_beta=0.01):
        """
        Fit a Ridge model to a data set using stochastic gradient descent.
        Arguments:
            X (array): design matrix
            y (array): response
            n_minibatches (int): number of minibatches
            n_epochs (int): number of epochs
            std_beta (float, optional): standard deviation of initial beta
                                        selection, defaults to 0.01
        """
        self.n_epochs = n_epochs
        n_features = X.shape[1]
        beta0 = np.random.normal(size=n_features)*std_beta

        x = np.c_[X, y]

        self.beta_path = sgd(beta0, self.del_C, self.gamma, x, n_minibatches,
                             n_epochs)
        self.beta = self.beta_path[-1]

    def predict(self, X):
        """
        Function for predicting the probabilites of each class.
        Arguments:
            X (array): design matrix
        Returns:
            p (array): probabilities of each class
        """
        a = X@self.beta
        b = np.exp(a-np.max(a))
        p = b/(1+b)

        return p

    def classify(self, X, threshold=0.5):
        """
        Function for classifying based on predicted probabilities.
        Arguments:
            X (array): design matrix
            threshold (float, optional): probability threshold at which to
                                         separate classes, defaults to 0.5
        Returns:
            y_pred (array): predicted classes
        """
        p = self.predict(X)

        y_pred = np.zeros(p.shape)
        y_pred[p >= threshold] = 1

        self.y_pred = y_pred
        return y_pred


class MultipleLogisticRegression:
    """
    Class for logistic regression with multiple classes.
    """
    def __init__(self, penalty):
        """
        Arguments:
            penalty (float): penalty parameter of logistic regression
        """
        self.penalty = penalty

        # set default learning parameters
        self.a = 0.01
        self.b = 5

    def set_learning_params(self, a, b):
        """
        Change the parameters for the learning schedule.
        Arguments:
            a (float): parameter 1
            b (float): parameter 2
        """
        self.a = a
        self.b = b

    def gamma(self, j):
        """
        Learning schedule for the stochastic gradient descent.
        Arguments:
            j (int): index of iteration
        Returns:
            gamma (float): learning parameter for specified index
        """
        gamma = self.a*np.exp(-self.b*j/self.n_epochs)

        return gamma

    def C(self, y_true, y_pred, penalty=None, beta=None):
        """
        Cost function for regression (squared loss + penalty).
        Arguments:
            y_true (array): observations
            y_pred (array): predictions (probabilities)
            penalty (float, optional): L2 penalty
            beta (array, optional): regression coefficients
        Returns:
            C (float): cost function value
        """
        if penalty is None:
            penalty = self.penalty
        if beta is None:
            beta = self.beta

        # one-hot encoding
        y_true_ = np.zeros(y_pred.shape)
        for i, y in enumerate(y_true.flatten()):
            y_true_[i,y] = 1

        # cost function
        C = - np.sum(y_true_ * np.log(y_pred)) + 0.5*penalty*np.sum(beta**2)

        return C

    def del_C(self, x, beta):
        """
        Gradient of cost functin for logistic regression.
        Arguments:
            x (array): design matrix and response
            beta (array): regression coefficients
        Returns:
            del_C (float): gradient value
        """
        y = x[:,-self.k:]
        X = x[:,:-self.k]

        a = X@beta
        b = np.exp(a-np.max(a))
        p = b/np.sum(b, axis=1, keepdims=True)

        del_C = - (X.T)@(y-p) + self.penalty*beta

        return del_C

    def fit(self, X, y, n_minibatches, n_epochs, std_beta=0.01, one_hot=False):
        """
        Fit a logistic regression model to a data set using stochastic gradient
        descent.
        Arguments:
            X (array): design matrix
            y (array): response
            n_minibatches (int): number of minibatches
            n_epochs (int): number of epochs
            std_beta (float, optional): standard deviation of initial beta
                                        selection, defaults to 0.01
            one_hot (bool, optional): whether to perform one-hot encoding,
                                      defaults to False
        """
        self.n_epochs = n_epochs
        n_features = X.shape[1]

        # perform one hot encoding if necessay
        if one_hot == False:
            self.unique_y = np.unique(y)
            self.dtype_y = y.dtype
            y_ = np.zeros([len(self.unique_y), y.size])
            for i, elem in enumerate(self.unique_y):
                y_[i] = (y == elem)*1
            y = y_
            self.encoded = True
            self.k = len(self.unique_y)
        else:
            self.encoded = False
            self.k = len(y)

        x = np.c_[X, y.T]

        beta0 = np.random.normal(size=n_features*self.k)*std_beta
        beta0 = beta0.reshape(n_features, self.k)

        self.beta_path = sgd(beta0, self.del_C, self.gamma, x, n_minibatches,
                             n_epochs)
        self.beta = self.beta_path[-1]

    def predict(self, X):
        """
        Predict probabilities for each class.
        Arguments:
            X (array): design matrix
        Returns:
            p (array): probabilitites
        """
        a = X@self.beta
        b = np.exp(a-np.max(a))
        p = b/np.sum(b, axis=1, keepdims=True)

        return p

    def classify(self, X):
        """
        Classify input data.
        Arguments:
            X (array): design matrix
        Returns:
            y (array): predicted classes 
        """
        # compute probabilities
        p = self.predict(X)

        # find classification index
        y = np.argmax(p, axis=1)

        if self.encoded == True:
            y_ = np.zeros(y.shape).astype(self.dtype_y)
            for i, elem in enumerate(self.unique_y):
                y_[y==i] = elem
            y = y_

        return y

if __name__ == "__main__":
    # create a dataset
    x = np.linspace(-1,1,100)
    y = np.zeros(x.shape)
    y[x>=0] = 1
    x += 0.5*np.random.randn(*x.shape)
    X = np.c_[np.ones(*x.shape), x]

    # test logistic regression class
    Log1 = LogisticRegression(0)
    Log1.fit(X, y, 5, 1000)
    y_pred = Log1.classify(X)
