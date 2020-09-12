import numpy as np

def design_matrix(x, y, n=1):
    """
    Function for creating a design matrix.
    Arguments:
        x (array): explanatory variable 1
        y (array): explanatory variable 2
        n (int, optional): order of polynomial, defautls to 1
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

    return X

def design_matrix_column_order(n=1):
    """
    Function for finding the order of the columns in the design matrix.
    Arguments:
        n (int, optional): order of polynomial, defaults to 1
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
    return column_names

# import data
data = np.load("data.npy")
x, y, z = data.T

n = len(z)
sigma2 = np.var(z)
z_mean = (1/n)*np.sum(z)

beta_variance = []
MSE = []
R2 = []

# perform OLS
N = 5
for i in range(0, N):
    X = design_matrix(x, y, n=i)
    beta = np.linalg.inv(((X.T)@X))@(X.T)@z

    z_hat = X@beta

    beta_variance.append(sigma2*np.linalg.inv((X.T)@X))
    MSE.append((1/n)*np.sum((z - z_hat)**2))
    R2.append(1 - (np.sum((z - z_hat)**2)) / (np.sum((z - z_mean)**2)))


















# y0
