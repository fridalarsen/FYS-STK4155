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










# y0
