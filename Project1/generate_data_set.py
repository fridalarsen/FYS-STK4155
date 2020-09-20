import numpy as np

def FrankeFunction(x, y):
    """
    Function for computing the Franke function.
    Arguments:
        x (float): x-value, must be in [0,1]
        y (float): y-value, must be in [0,1]
    Returns:
        z (float): the resulting Franke function value
    """
    term1 = 0.75*np.exp(-(0.25*(9*x-2)**2) - 0.25*((9*y-2)**2))
    term2 = 0.75*np.exp(-((9*x+1)**2)/49.0 - 0.1*(9*y+1))
    term3 = 0.5*np.exp(-(9*x-7)**2/4.0 - 0.25*((9*y-3)**2))
    term4 = -0.2*np.exp(-(9*x-4)**2 - (9*y-7)**2)
    return term1 + term2 + term3 + term4

def generate_data_set(N, sigma):
    """
    Function for generating a data set based on the Franke function with noise.
    Arguments:
        N (int): number of data points to generate
        sigma (float): standard deviation of noise
    Returns:

    """
    # make coordinate grid
    x = np.linspace(0, 1, N)
    y = np.linspace(0, 1, N)
    X, Y = np.meshgrid(x,y)

    # make surface
    Z = FrankeFunction(X, Y)

    # make 1d-arrays
    x = X.flatten()
    y = Y.flatten()
    z = Z.flatten()

    # add noise
    z = z + sigma*np.random.randn(len(z))

    return x, y, z
