import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

class C_regression:
    """
    Class containing the cost and output activation function for regression.
    """
    @staticmethod
    def C(y_true, y_pred, penalty=0, W=0):
        """
        Cost function for regression (squared loss + penalty).
        Arguments:
            y_true (array): observations
            y_pred (array): predictions
            penalty (float, optional): penalty, defaults to 0
            W (array, optional): weigths, defaults to 0
        Returns:
            C (float): cost function value
        """
        C = 0.5*((y_true-y_pred).T)@(y_true-y_pred)

        if penalty > 0:
            for w in W:
                C += 0.5*penalty*np.sum(w**2)

        return C

    @staticmethod
    def output_act(x):
        """
        Output activation for regression (identity).
        """
        return x

class C_classification:
    """
    Class containing the cost and output activation function for classification.
    """
    @staticmethod
    def C(y_true, y_pred, penalty=0, W=0):
        """
        Cost function for classification (cross entropy + penalty).
        Arguments:
            y_true (array): observed response
            y_pred (array): predicted response
            penalty (float, optional): penalty parameter, defaults to 0
            W (float, optional): weight, defaults to 0
        """
        # fix dimensions
        y_pred = y_pred.T

        # one-hot encoding
        y_true_ = np.zeros(y_pred.shape)
        for i, y in enumerate(y_true.flatten()):
            y_true_[i,y] = 1

        # cost function
        C = - np.sum(y_true_ * np.log(y_pred))

        # weights penalty
        if penalty > 0:
            for w in W:
                C += 0.5*penalty*np.sum(w**2)

        return C

    @staticmethod
    def output_act(x):
        """
        Output activation function for classification (softmax).
        """
        a = np.exp(x-np.max(x))
        sm = a / np.sum(a, axis=0, keepdims=True)
        return sm

class logit:
    """
    Class for the logit function.
    """
    @staticmethod
    def f(x):
        """
        The logit function.
        """
        a = np.exp(x-np.max(x))
        f = a/(1+a)
        return f

    @staticmethod
    def df(x):
        """
        The derivative of the logit function.
        """
        f = logit.f(x)
        df = f*(1-f)
        return df

class tanh:
    """
    Class for the tanh function.
    """
    @staticmethod
    def f(x):
        """
        The tanh function.
        """
        f = np.tanh(x)
        return f

    @staticmethod
    def df(x):
        """
        The derivative of the tanh function.
        """
        df = 1 - np.tanh(x)**2
        return df

class id:
    """
    Class for the identity function.
    """
    @staticmethod
    def f(x):
        """
        The identity function.
        """
        return x

    @staticmethod
    def df(x):
        """
        The derivative of the identity function.
        """
        id_d = np.ones(x.shape)
        return id_d


def plot_confusion_matrix(y_true, y_pred, normalize=True, figscale=1, figtitle=None):
    """
    Function for visualising the confusion matrix.
    Arguments:
        y_true (array): observations
        y_pred (array): predictions
        normalize (bool, optional): whether the confusion matrix is to be
                                    normalized, defaults to True
        figscale (float, optional): parameter for scaling the figure size,
                                    defaults to 1
        figtitle (str, optional): figure is saved under this name if provided,
                                  defaults to None.
    """
    c = confusion_matrix(y_true, y_pred)
    if normalize is True:
        # normalize confusion matrix
        c = c/np.sum(c)

    fig, ax = plt.subplots(figsize=(figscale*5, figscale*4.5))
    vmax = 1 if normalize else c.max()
    im = ax.matshow(c, vmin=0, vmax=vmax, cmap="autumn_r")
    plt.colorbar(im)
    s = "{:0.1f}" if normalize else "{:d}"
    for (i, j), z in np.ndenumerate(c):
        ax.text(j, i, s.format(z), ha="center", va="center",
                fontsize=16)
    ax.set_xlabel("Predicted", fontsize=12)
    ax.xaxis.set_label_position("top")
    ax.set_ylabel("True", fontsize=12)
    fig.suptitle("Confusion matrix", fontsize=16)
    fig.subplots_adjust(top=0.84)
    if figtitle is not None:
        plt.savefig(f"Figures/{figtitle}.png", dpi=300)
    plt.show()

class NeuralNetwork:
    """
    Class for neural network.
    """
    def __init__(self, n_hidden_layers, n_hidden_nodes, penalty=0,
                 activation="logit", regression=True):
        """
        Arguments:
            n_hidden_layers (int): number of hidden layers
            n_hidden_nodes (int): number of hidden nodes
            penalty (float, optional): penalty parameter, defaults to 0
            activation (function, optional): activation function to use,
                                             defaults to logit
            regression (bool, optional): whether to perform regression (True) or
                                         classification (False), defaults to
                                         True
        Raises:
            ValueError: if the activation function given is not valid
        """
        self.n_hidden_layers = n_hidden_layers
        self.n_hidden_nodes = n_hidden_nodes
        self.regression = regression

        # set cost function
        if regression is True:
            self.C = C_regression
        else:
            self.C = C_classification

        # set activation function
        if activation == "logit":
            self.act = logit
        elif activation == "tanh":
            self.act = tanh
        else:
            raise ValueError("Activation must be logit or tanh")

        # set default learning parameters
        self.a1 = 1e-3
        self.a2 = 1e0

        # set penalty
        self.penalty = penalty

    def set_learning_params(self, a1, a2):
        """
        Change the parameters for the learning schedule.
        Arguments:
            a1 (float): parameter 1
            a2 (float): parameter 2
        """
        self.a1 = a1
        self.a2 = a2

    def gamma(self, j):
        """
        Learning schedule for the stochastic gradient descent.
        Arguments:
            j (int): index of iteration
        Returns:
            gamma (float): learning parameter for specified index
        """
        gamma = self.a1 * np.exp(-self.a2*j/self.n_epochs)

        return gamma

    def fit(self, X, y, n_minibatches, n_epochs, std_W=0.01, const_b=0.01,
            one_hot=False, track_cost=None):
        """
        Fit a model to the input data using stochastic gradient descent.
        Arguments:
            X (array): design matrix
            y (array): response
            n_minibatches (int): number of minibatches in sgd
            n_epochs (int): number of epochs
            std_W (float, optional): standard deviation of initial W guess,
                                     defaults to 0.01
            const_b (float, optional): initial bias, defaults to 0.01
            one_hot (bool, optional): whether the input is one hot encoded,
                                      default to False
            track_cost (tuple or None, optional): if None, the cost function
                                                  is not tracked. If not None,
                                                  must be on the form
                                                  [X_test, Z_test]. Defaults to
                                                  None.
        """
        # format inputs
        X = X if X.ndim>1 else X[:,None]
        X = X.T
        y = y if y.ndim>1 else y[:,None]
        y = y.T

        if self.regression == False and one_hot == False:
            self.unique_y = np.unique(y)
            self.dtype_y = y.dtype
            y_ = np.zeros([len(self.unique_y), y.size])
            for i, elem in enumerate(self.unique_y):
                y_[i] = (y == elem)*1
            y = y_
            self.encoded = True
        else:
            self.encoded = False

        self.n_epochs = n_epochs
        n_features = X.shape[0]
        n_targets = y.shape[0]

        # initial guess of weights and biases
        self.W = [std_W * np.random.randn(self.n_hidden_nodes, n_features)]
        self.b = [const_b + np.zeros((self.n_hidden_nodes,1))]
        for i in range(self.n_hidden_layers-1):
            self.W.append(std_W * np.random.randn(self.n_hidden_nodes,
                                                  self.n_hidden_nodes))
            self.b.append(const_b + np.zeros((self.n_hidden_nodes,1)))
        self.W.append(std_W * np.random.randn(n_targets, self.n_hidden_nodes))
        self.b.append(const_b + np.zeros((n_targets,1)))

        if track_cost is not None:
            self.cost = np.zeros(n_epochs)

        # training loop
        for epoch in range(0, n_epochs):
            # shuffle dataset
            idx = np.random.choice(X.shape[1], size=X.shape[1], replace=False)
            minibatches = np.array_split(idx, n_minibatches)
            for i in range(n_minibatches):
                # select minibatch
                k = np.random.randint(n_minibatches)
                X_k = X[:,minibatches[k]]
                y_k = y[:,minibatches[k]]

                self.feed_forward(X_k)
                self.backpropagation(y_k)

                # compute new weights and biases
                for j in range(self.n_hidden_layers+1):
                    self.W[j] = self.W[j] - self.gamma(epoch)*self.dC_dW[j]
                    self.b[j] = self.b[j] - self.gamma(epoch)*self.dC_db[j]

            if track_cost is not None:
                y_pred = self.predict(track_cost[0])
                self.cost[epoch] = self.C.C(track_cost[1], y_pred,
                                            penalty=self.penalty, W=self.W)

    def feed_forward(self, X):
        """
        Feed forward algorithm.
        Arguments:
            X (array): design matrix
        """
        self.z = []
        self.y = [X]  # pre-transposed
        for layer in range(self.n_hidden_layers):
            self.z.append(self.W[layer]@self.y[layer] + self.b[layer])
            self.y.append(self.act.f(self.z[layer]))
        self.z.append(self.W[-1]@self.y[-1] + self.b[-1])
        self.y.append(self.C.output_act(self.z[-1]))

    def backpropagation(self, y_true):
        """
        Perform back propagation.
        Arguments:
            y_true (array): observed response
        """
        delta = [self.y[-1] - y_true]
        for i in range(self.n_hidden_layers):
            l = self.n_hidden_layers - i
            delta.append((self.W[l].T)@(delta[-1]*self.act.df(self.z[l])))
        delta.reverse()

        self.dC_dW = []
        self.dC_db = []
        for l in range(self.n_hidden_layers+1):
            grad_W_3d = np.einsum('ij,jk->jik',delta[l],self.y[l].T)
            grad_W = np.sum(grad_W_3d,axis=0)
            self.dC_dW.append(grad_W)
            self.dC_db.append(np.sum(delta[l], axis=1, keepdims=True))

            if self.penalty > 0:
                self.dC_dW[l] += self.penalty * self.W[l]

    def predict(self, X):
        """
        Perform prediction.
        Arguments:
            X (array): input design matrix
        Returns:
            y (array): predicted response (probabilities in the case of
                       classification)
        """
        X = X if X.ndim>1 else X[:,None]
        y = X.T
        for layer in range(self.n_hidden_layers):
            z = self.W[layer]@y + self.b[layer]
            y = self.act.f(z)
        z = self.W[-1]@y + self.b[-1]
        y = self.C.output_act(z)
        if y.shape[0] == 1: y = y.flatten()
        return y

    def classify(self, X):
        """
        Perform classification.
        Arguments:
            X (array): input design matrix
        Returns:
            y (array): predicted output classes
        """
        # compute probabilities
        p = self.predict(X)

        # find classification index
        y = np.argmax(p, axis=0)

        if self.encoded == True:
            y_ = np.zeros(y.shape).astype(self.dtype_y)
            for i, elem in enumerate(self.unique_y):
                y_[y==i] = elem
            y = y_

        return y
