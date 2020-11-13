import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

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

def generate_franke_data(Nx, Ny, sigma):
    """
    Function for generating a data set based on the Franke function with noise.
    Arguments:
        Nx (int): number of x-coordinates to generate
        Ny (int): number of y-coordinates to generate
        sigma (float): standard deviation of noise
    Returns:
        x (array): x-coordinates
        y (array): y-coordinates
        z (array): Franke function values with noise
    """
    # make coordinate grid
    x = np.linspace(0, 1, Nx)
    y = np.linspace(0, 1, Ny)
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

if __name__ == "__main__":
    # plot the Franke function
    x = np.linspace(0, 1, 100)
    y = np.linspace(0, 1, 100)
    X, Y = np.meshgrid(x,y)

    Z = FrankeFunction(X,Y)

    fig = plt.figure()
    ax = plt.axes(projection="3d")
    ax.plot_surface(X, Y, Z, cmap="autumn")
    ax.set_title("The Franke function", fontsize=15)
    ax.set_xlabel("x", fontsize=12)
    ax.set_ylabel("y", fontsize=12)
    ax.set_zlabel("z", fontsize=12)
    plt.savefig("Figures/franke_function.png", dpi=300)
    plt.show()
