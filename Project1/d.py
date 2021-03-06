import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from generate_data_set import generate_data_set, FrankeFunction
from a import design_matrix, prep_surface
from b import bias_variance_tradeoff
from c import kfold

def Ridge(X, r, lambda_, r_var=None):
    """
    Function for finding regression coefficients using ridge regression.
    Arguments:
        X (array): design matrix
        r (array): response vector
        lambda_ (float): tuning parameter
        r_var (float, optional): variance of response is calculated if none is
                                 provided, defaults to None
    Returns:
        beta (array): vector of coefficients
        beta_var (array): variance of beta values
    """
    if r_var is None:
        r_var = np.var(r)

    XTX = (X.T)@X
    a = np.linalg.inv(XTX + lambda_*np.eye(X.shape[1], X.shape[1]))

    beta = (a@(X.T))@r
    beta_var = r_var*(a@XTX@(a.T))

    return beta, beta_var

if __name__ == "__main__":
    # create data
    N = 100
    x, y, z = generate_data_set(N, 1)

    # standardizing response
    z_var = np.var(z)
    z_mean = np.mean(z)

    z = (z-z_mean)/np.sqrt(z_var)

    k = 5                                                   # number of folds

    N_pol = [3, 5, 7]                                       # polynomial degrees
    lambdas = np.logspace(-4, 3, 8)                         # penalties

    MSE_b = np.zeros([len(N_pol), len(lambdas)])
    bias_b = np.zeros([len(N_pol), len(lambdas), 2])
    var_b = np.zeros([len(N_pol), len(lambdas), 2])

    MSE_k = np.zeros([len(N_pol), len(lambdas), 2])

    for i, deg in enumerate(N_pol):
        # prepare data
        X = design_matrix(x, y, n=deg)

        # preprocess data
        scaler = StandardScaler()
        scaler.fit(X)
        X = scaler.transform(X)

        for j, l in enumerate(lambdas):
            # bootstrap
            MSE_b[i,j], bias_b[i,j], var_b[i,j] = bias_variance_tradeoff(X, z,
                                                  500, model=Ridge, lambda_=l)

            # kfold
            MSE_k[i,j] = kfold(k, X, z, model=Ridge, lambda_=l)

    for i, deg in enumerate(N_pol):
        plt.errorbar(lambdas, bias_b[i,:,0], yerr=bias_b[i,:,1], fmt="o",
                     capsize=7, label="bias", color="darkorange")
        plt.errorbar(lambdas, var_b[i,:,0], yerr=var_b[i,:,1], fmt="o",
                     capsize=7, label="var", color="brown")
        plt.scatter(lambdas, MSE_b[i], marker="o", label="MSE", color="red")
        plt.xscale("log")
        plt.legend(loc="lower left")
        plt.title(f"Ridge bias-variance tradeoff, polynomial degree {deg}",
                  fontsize=15)
        plt.xlabel(r"$\lambda$", fontsize=12)
        plt.ylabel(r"Error$^2$", fontsize=12)
        plt.savefig(f"Figures/ridge_bias_variance_deg{deg}.png", dpi=300)
        plt.show()

        plt.errorbar(lambdas, MSE_k[i,:,0], yerr=MSE_k[i,:,1], fmt="o",
                     capsize=7, color="red")
        plt.xscale("log")
        plt.yscale("log")
        plt.subplots_adjust(left=0.20)
        plt.xlabel(r"$\lambda$", fontsize=12)
        plt.ylabel(r"Error$^2$", fontsize=12)
        plt.title(f"{k}fold mean squared error, polynomial degree {deg}",
                  fontsize=15)
        plt.savefig(f"Figures/ridge_kfold_deg{deg}.png", dpi=300)
        plt.show()

    # make an example approximation
    # prepare data
    X = design_matrix(x, y, n=5)
    X_train, X_test, z_train, z_test = train_test_split(X, z, test_size=0.35)

    # preprocess data
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    # fit
    beta, beta_var = Ridge(X_train, z_train, lambda_=1e-2, r_var=1)

    x_surf = np.linspace(0, 1, 100)
    y_surf = np.linspace(0, 1, 100)

    X_surf, Y_surf, Z_surf = prep_surface(x_surf, y_surf, beta, deg=5)

    fig = plt.figure()
    ax = plt.axes(projection="3d")
    ax.plot_surface(X_surf, Y_surf, Z_surf, cmap="autumn")
    ax.set_title(f"Franke Function Ridge approximation,\n deg=5, " \
                 + f"N=100 and " + r"$\lambda=10^{-2}$", fontsize=15)
    ax.set_xlabel("x", fontsize=12)
    ax.set_ylabel("y", fontsize=12)
    ax.set_zlabel("z", fontsize=12)
    plt.savefig(f"Figures/franke_ridge_deg5_N100.png", dpi=300)
    plt.show()

    plt.imshow((Z_surf - FrankeFunction(X_surf, Y_surf))**2,
               cmap="RdYlGn", origin="lower",
               extent=[X_surf.min(), X_surf.max(), Y_surf.min(),
               Y_surf.max()])
    plt.title(f"Difference between prediction and true value,\n"\
              + f" deg=5, N=100 and " + r"$\lambda=10^{-2}$", fontsize=15)
    plt.xlabel("x", fontsize=12)
    plt.ylabel("y", fontsize=12)
    cb = plt.colorbar()
    cb.set_label(label="Squared error", fontsize=12)
    plt.savefig(f"Figures/franke_ridge_error_deg5_N100.png", dpi=300)
    plt.show()
