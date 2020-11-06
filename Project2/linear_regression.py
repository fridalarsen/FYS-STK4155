import numpy as np
import matplotlib.pyplot as plt
from gradient_descent import sgd

class Ridge:
    """
    Class for linear regression using the Ridge method.
    """
    def __init__(self, penalty):
        """
        Arguments:
            penalty (float): penalty parameter of the ridge model
        """
        self.penalty = penalty

        # set default learning parameters
        self.a = 0.01
        self.b = 5

    def C(self, X, y, beta):
        """
        Cost function of Ridge regression.
        Arguments:
            X (array): design matrix
            y (array): response
            beta (array): regression coefficients
        Returns:
            C (array): cost function
        """
        C = ((y-X@beta).T)@(y-X@beta) + self.penalty*(beta.T)@beta

        return C

    def del_C(self, x, beta):
        """
        Gradient of Ridge cost function.
        Arguments:
            x (array): design matrix and response
            beta (array): regression coefficients
        Returns:
            del_C (array): gradient of cost function
        """
        y = x[:,-1]
        X = x[:,:-1]

        del_C = -2*(X.T)@(y-X@beta) + 2*self.penalty*beta

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
        gamma = self.a*np.exp(-self.b*j/self.n_epochs)

        return gamma

    def fit_sgd(self, X, y, n_minibatches, n_epochs, std_beta=0.01):
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

    def fit_minv(self, X, y):
        """
        Fit a Ridge model to a data set using matrix inversion.
        Arguments:
            X (array): design matrix
            y (array): response
        """
        XTX = (X.T)@X
        a = np.linalg.inv(XTX + self.penalty*np.eye(X.shape[1], X.shape[1]))

        self.beta = (a@(X.T))@y


    def predict(self, X):
        """
        Predict response.
        Arguments:
            X (array): design matrix
        """
        y = X@self.beta

        return y


if __name__ == "__main__":
    # create a dataset
    x = np.linspace(0,1,100)
    y = np.sin(x*np.pi) + 0.15*np.random.randn(*x.shape)

    # create design matrix
    X = np.c_[np.ones(x.shape), x, x**2, x**3]

    # test Ridge class
    Ridge1 = Ridge(1e-2)
    Ridge1.fit_minv(X, y)
    y_pred1 = Ridge1.predict(X)

    Ridge1.fit_sgd(X, y, 5, 1000)
    y_pred2 = Ridge1.predict(X)

    plt.scatter(x, y, label="data", s=5, color="brown")
    plt.plot(x, y_pred1, label="minv", color="red")
    plt.plot(x, y_pred2, label="sgd", color="orange")
    plt.legend()
    plt.show()
