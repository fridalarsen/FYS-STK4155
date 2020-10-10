import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from generate_data_set import generate_data_set, FrankeFunction

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
        incl_ones (bool, optional): includes the 0th order column if True,
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

def prep_surface(x, y, beta, deg, scaler=None):
    """
    Function for preparing the a surface plot based on a regression.
    Arguments:
        x (array): x-coordinates of surface
        y (array): y-coordinates of surface
        beta (array): regression coefficients
        deg (int): polynomial degree
        scaler (StandardScaler, optional): fitted sklearn StandardScaler,
                                           ignored if None
    Returns:
        X_surf (array): 2D x-coordinates of surface
        Y_surf (array): 2D y-coordinates of surface
        Z_surf (array): predicted 2D z-coordinates of surface
    """
    X_surf, Y_surf = np.meshgrid(x,y)

    X_design = design_matrix(X_surf.flatten(), Y_surf.flatten(), n=deg)

    if scaler is not None:
        X_design = scaler.transform(X_design)

    z = X_design@beta
    Z_surf = z.reshape(X_surf.shape)

    return X_surf, Y_surf, Z_surf

def LinearRegression(X, r, r_var=None, **kwargs):
    """
    Function for finding regression coefficients
    Arguments:
        X (array): design matrix
        r (array): response vector
        r_var (float, optional): variance of response is calculated if none is
                                 provided, defaults to None
        **kwargs: compatibility with RidgeRegression
    Returns:
        beta (array): regression coefficients
        beta_var (array): variance of beta values
    """
    if r_var is None:
        r_var = np.var(r)

    XTX_inv = np.linalg.inv((X.T)@X)
    beta = np.linalg.pinv(X)@r

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
    # generate data
    N_points = 100
    x, y, z = generate_data_set(N_points, 1)

    # standardize response
    z_var = np.var(z)
    z_mean = np.mean(z)

    z = (z-z_mean)/np.sqrt(z_var)

    # perform OLS
    N = 15                                      # highest polynomial degree

    beta_variance = []
    MSE_test = np.zeros(N)
    MSE_train = np.zeros(N)
    R2_test = np.zeros(N)
    R2_train = np.zeros(N)

    for i in range(1, N+1):
        # prepare data
        X = design_matrix(x, y, n=i)
        X_train, X_test, z_train, z_test = train_test_split(X, z, test_size=0.35)

        # preprocess data
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

        # plot an example of the approximations
        if i == 4 or i == 10:
            x_surf = np.linspace(0, 1, 100)
            y_surf = np.linspace(0, 1, 100)

            X_surf, Y_surf, Z_surf = prep_surface(x_surf, y_surf,
                                                     beta, deg=i)

            fig = plt.figure()
            ax = plt.axes(projection="3d")
            ax.plot_surface(X_surf, Y_surf, Z_surf, cmap="autumn")
            ax.set_title(f"Franke Function OLS approximation,\n deg={i} and " \
                         + f"N={N_points}", fontsize=15)
            ax.set_xlabel("x", fontsize=12)
            ax.set_ylabel("y", fontsize=12)
            ax.set_zlabel("z", fontsize=12)
            plt.savefig(f"Figures/franke_OLS_deg{i}_N{N_points}.png", dpi=300)
            plt.show()

            plt.imshow((Z_surf - FrankeFunction(X_surf, Y_surf))**2,
                       cmap="RdYlGn", origin="lower",
                       extent=[X_surf.min(), X_surf.max(), Y_surf.min(),
                       Y_surf.max()])
            plt.title(f"Difference between prediction and true value,\n"\
                      + f" deg={i} and N={N_points}", fontsize=15)
            plt.xlabel("x", fontsize=12)
            plt.ylabel("y", fontsize=12)
            cb = plt.colorbar()
            cb.set_label(label="Squared error", fontsize=12)
            plt.savefig(f"Figures/franke_OLS_error_deg{i}_N{N_points}.png", dpi=300)
            plt.show()

    complexity = np.linspace(1, N, N)
    plt.plot(complexity, MSE_test, label="Testing set", color="orange")
    plt.plot(complexity, MSE_train, label="Training set", color="darkred")
    plt.legend()
    plt.subplots_adjust(left=0.16)
    plt.xlabel("Model Complexity", fontsize=12)
    plt.ylabel("Mean Squared Error", fontsize=12)
    plt.title(f"Model Complexity Optimization, N={N_points}", fontsize=15)
    plt.savefig(f"Figures/model_complexity_mse_franke_N{N_points}.png", dpi=300)
    plt.show()
