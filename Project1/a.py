import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

def design_matrix(x, y, n=1, incl_ones=False):
    """
    Function for creating a design matrix.
    Arguments:
        x (array): explanatory variable 1
        y (array): explanatory variable 2
        n (int, optional): order of polynomial, defautls to 1
        incl_ones (bool, optional): incluces the 0th order column if True,
                                    defaults to False.
    Returns:
        X (array): design matrix
    """
    rows = len(x)
    cols = (n+1)*(n+2)/2.

    X = np.ones((int(rows), int(cols)))

    l = 0
    for ny in range(0, n+1):
        for nx in range(0, n+1):
            if l < cols and nx+ny <= n:
                X[:, l] = (x**nx)*(y**ny)
                l += 1

    if incl_ones is False:
        X = X[:, 1:]

    return X

def design_matrix_column_order(n=1, incl_ones=False):
    """
    Function for finding the order of the columns in the design matrix.
    Arguments:
        n (int, optional): order of polynomial, defaults to 1
        incl_ones (bool, optional): incluces the 0th order column if True,
                                    defaults to False.
    Returns:
        column_names (list): list of column names
    """
    cols = (n+1)*(n+2)/2.
    column_names = []

    l = 0
    for ny in range(0, n+1):
        for nx in range(0, n+1):
            if l < cols and nx+ny <= n:
                column_names.append(f"x^{nx} y^{ny}")
                l += 1

    if incl_ones is False:
        column_names = column_names[1:]

    return column_names

def LinearRegression(X, r, r_var=None):
    """
    Function for finding regression coefficients
    Arguments:
        X (array): design matrix
        r (array): response vector
        r_var (float, optional): variance of response is calculated if none is
                                 provided, defaults to None
    Returns:
        beta (array): vector of coefficients
        beta_var (array): variance of beta values
    """
    if r_var is None:
        r_var = np.var(r)

    XTX_inv = np.linalg.inv((X.T)@X)
    beta = np.linalg.pinv(X)@r

    #beta = XTX_inv@((X.T)@r)
    r_pred = X@beta

    beta_var = r_var*np.diag(XTX_inv)

    return beta, beta_var

def MSE_R2(pred, true):
    """
    Function for calculating the mean squared error and R^2 score of a
    prediction.
    Arguments:
        pred (array): predicted response
        true (array): true response
    Returns:
        MSE (float): mean squared error of prediction
        R2 (float): R^2 score of prediction
    """
    MSE = np.mean((true-pred)**2)
    R2 = 1 - np.sum((true-pred)**2)/np.sum((true-np.mean(true))**2)

    return MSE, R2

if __name__ == "__main__":

    # import data
    data = np.load("data.npy")
    x, y, z = data.T

    # standardizing response
    z_var = np.var(z)
    z_mean = np.mean(z)

    z = (z-z_mean)/np.sqrt(z_var)

    # perform OLS
    N = 20

    beta_variance = []
    MSE_test = np.zeros(N)
    MSE_train = np.zeros(N)
    R2_test = np.zeros(N)
    R2_train = np.zeros(N)

    for i in range(1, N+1):
        # preparing data
        X = design_matrix(x, y, n=i)
        X_train, X_test, z_train, z_test = train_test_split(X, z, test_size=0.35)

        # preprocessing data
        scaler = StandardScaler()
        scaler.fit(X_train)
        X_train = scaler.transform(X_train)
        X_test = scaler.transform(X_test)

        # fit
        beta, beta_var = LinearRegression(X_train, z_train, r_var=1)
        beta_variance.append(beta_var)

        # predict
        z_hat_train = X_train@beta
        z_hat_test = X_test@beta

        # quality check
        MSE_train[i-1], R2_train[i-1] = MSE_R2(z_hat_train, z_train)
        MSE_test[i-1], R2_test[i-1] = MSE_R2(z_hat_test, z_test)


    complexity = np.linspace(1, N, N)
    plt.plot(complexity, MSE_test, label="Test Sample")
    plt.plot(complexity, MSE_train, label="Training Sample")
    plt.legend()
    plt.yscale("log")
    plt.xlabel("Model Complexity", fontsize=12)
    plt.ylabel("Mean Squared Error", fontsize=12)
    plt.title("Model Complexity Optimization", fontsize=15)
    plt.savefig("Figures/b_1.png", dpi=300)
    plt.show()
