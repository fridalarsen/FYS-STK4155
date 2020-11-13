import numpy as np


def gradient_descent(a0, del_f, k, N):
    """
    Function for finding the minimum of a function using gradient descent.
    Arguments:
        a0 (float): initial guess
        del_f (function): gradient of function to be minimized
        k (float): learning rate
        N (int): maximum number of steps
    Returns:
        a (float): computed minimum
    """
    a = a0
    for i in range(N):
        a  = a - k*del_f(a)

    return a

def sgd(beta0, del_f, gamma, x, n_minibatches, n_epochs):
    """
    Function for finding the minimum of a function using stochastic gradient
    descent.
    Arguments:
        beta0 (float): initial guess
        del_f (function): gradient of function to be minimized
        gamma (function): learning schedule
        x (array): data points
        n_minibatches (int): number of minibatches
        n_epochs (int): number of epochs
    Returns:
        beta (array): path of minimisation parameter
    """
    beta = np.zeros([n_epochs+1, *np.shape(beta0)])
    beta[0] = beta0
    beta_ = beta0

    for epoch in range(0, n_epochs):
        # shuffle dataset
        idx = np.random.choice(x.shape[0], size=x.shape[0], replace=False)
        minibatches = np.array_split(idx, n_minibatches)

        for i in range(n_minibatches):
            # select minibatch
            k = np.random.randint(n_minibatches)
            Bk = x[minibatches[k]]

            # compute new suggestion
            beta_ = beta_ - gamma(epoch)*(del_f(Bk, beta_))

        beta[epoch+1] = beta_

    return beta
