import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

class C_regression:
    @staticmethod
    def C(y_true, y_pred):
        C = ((y_true-y_pred).T)@(y_true-y_pred)

        return C

    @staticmethod
    def output_act(x):
        return x

class C_classification:
    @staticmethod
    def C(y_true, y_pred):
        C = -((y_true.T)@np.log(y_pred)+((1-y_true).T)@np.log(1-y_pred))

        return C

    @staticmethod
    def output_act(x):
        a = np.exp(x-np.max(x))
        sm = a / np.sum(a, axis=0, keepdims=True)
        return sm

class logit:
    @staticmethod
    def f(x):
        f = np.exp(x)/(1+np.exp(x))
        return f

    @staticmethod
    def df(x):
        f = logit.f(x)
        df = f*(1-f)
        return df

class tanh:
    @staticmethod
    def f(x):
        f = np.tanh(x)
        return f

    @staticmethod
    def df(x):
        df = 1 - np.tanh(x)**2
        return df

class id:
    @staticmethod
    def f(x):
        return x

    @staticmethod
    def df(x):
        id_d = np.ones(x.shape)
        return id_d

def plot_confusion_matrix(y_true, y_pred, normalize=True, figtitle=None):
    """
    Function for visualising the confusion matrix.
    """
    c = confusion_matrix(y_true, y_pred)
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

class NeuralNetwork:
    def __init__(self, n_hidden_layers, n_hidden_nodes, penalty=0, activation="logit", regression=True):
        self.n_hidden_layers = n_hidden_layers
        self.n_hidden_nodes = n_hidden_nodes
        self.regression = regression

        if regression is True:
            self.C = C_regression
        else:
            self.C = C_classification

        if activation == "logit":
            self.act = logit
        elif activation == "tanh":
            self.act = tanh
        else:
            raise ValueError("Activation must be logit or tanh")

        # set default learning parameters
        self.a1 = 0.01
        self.a2 = 5

        # set penalty
        self.penalty = penalty

    def set_learning_params(self, a1, a2):
        """
        Change the parameters for the learning schedule.
        Arguments:
            a (float): parameter 1
            b (float): parameter 2
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
        gamma = self.a1*np.exp(-self.a2*j/self.n_epochs)

        return gamma

    def fit(self, X, y, n_minibatches, n_epochs, std_W=0.01, const_b=0.01, one_hot=False):
        """
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

    def feed_forward(self, X):
        self.z = []
        self.y = [X]  # pre-transposed
        for layer in range(self.n_hidden_layers):
            self.z.append(self.W[layer]@self.y[layer] + self.b[layer])
            self.y.append(self.act.f(self.z[layer]))
        self.z.append(self.W[-1]@self.y[-1] + self.b[-1])
        self.y.append(self.C.output_act(self.z[-1]))

    def backpropagation(self, y_true):
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

if __name__ == "__main__":
    # regression test
    x = np.linspace(0,1,100)
    y = np.sin(x*np.pi) + 0.15*np.random.randn(*x.shape)
    X = np.c_[x, x**2, x**3]

    NN = NeuralNetwork(2, 10, activation="tanh")
    NN.set_learning_params(a1=0.05, a2 = 2)
    NN.fit(X, y, 5, 1000)
    y_pred = NN.predict(X)

    plt.scatter(x, y, label="data", s=5, color="red")
    plt.plot(x, y_pred, label="nn", color="orange")
    plt.legend()
    plt.show()

    # classification test
    x = np.linspace(0,1,100)
    y = np.zeros(x.shape).astype(str)
    y[x >= 0.0] = "a"
    y[x > 0.33] = "b"
    y[x > 0.66] = "c"
    x += 0.5*np.random.randn(*x.shape)

    NN = NeuralNetwork(2, 10, activation="tanh", regression=False)
    NN.set_learning_params(a1=0.05, a2 = 2)
    NN.fit(x, y, 5, 1000)
    y_pred = NN.classify(x)

    plot_confusion_matrix(y, y_pred)
