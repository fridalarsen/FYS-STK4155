import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sn
from gradient_descent import sgd

class LogisticRegression:
    """
    """
    def __init__(self, penalty):
        self.penalty = penalty

        # set default learning parameters
        self.a = 0.01
        self.b = 5

    def C(self, X, y, beta):
        a = X@beta

        C = -np.sum(y*a - np.log(1+exp(a))) + self.penalty*(beta.T)@beta

        return C

    def del_C(self, x, beta):
        y = x[:,-1]
        X = x[:,:-1]

        a = X@beta

        p = (np.exp(a))/(1+np.exp(a))

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
        gamma = self.a*np.exp(-self.b*j/self.n_epochs)

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
        a = X@self.beta
        p = np.exp(a)/(1+np.exp(a))

        return p

    def classify(self, X, threshold=0.5):
        p = self.predict(X)

        y_pred = np.zeros(p.shape)
        y_pred[p >= threshold] = 1

        self.y_pred = y_pred
        return y_pred

    def plot_confusion_matrix(self, y_true, normalize=True, figtitle=None):
        """
        Function for visualising the confusion matrix.
        """
        if not hasattr(self, "y_pred"):
            raise AttributeError("Nothing has been classified yet.")

        c = confusion_matrix(y_true, self.y_pred)
        if normalize is True:
            c = c/np.sum(c)

        fig, ax = plt.subplots(figsize=(5,4.5))
        im = ax.matshow(c, vmin=0, vmax=1, cmap="autumn_r")
        plt.colorbar(im)
        for (i, j), z in np.ndenumerate(c):
            ax.text(j, i, "{:0.3f}".format(z), ha="center", va="center",
                    fontsize=16)
        ax.set_xlabel("Predicted", fontsize=12)
        ax.xaxis.set_label_position("top")
        ax.set_ylabel("True", fontsize=12)
        fig.suptitle("Confusion matrix", fontsize=16)
        fig.subplots_adjust(top=0.84)
        if figtitle is not None:
            plt.savefig(f"Figures/{figtitle}.png", dpi=300)
        plt.show()


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

    Log1.plot_confusion_matrix(y)
