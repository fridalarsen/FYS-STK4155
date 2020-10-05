import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from generate_data_set import generate_data_set, FrankeFunction
from a import design_matrix, design_matrix_column_order, LinearRegression, MSE_R2

def bias_variance_tradeoff(X, z, B, model=LinearRegression, **kwargs):
    """
    Function for finding the bias and variance of a model using a bootstrap
    resampling technique.
    Arguments:
        X (array): design matrix
        z (array): response
        B (int): number of bootstrap samples
        model (function, optional): function of model to be used in the
                                    regression, defaults to LinearRegression
        **kwargs: additional arguments for model
    Returns:
        MSE (float): mean squared error of main model
        Bias2 (float): bootstrap estimate of bias^2
        ModVar (float): bootstrap estimate of variance
    """
    # split data
    test_size = int(0.35*X.shape[0])
    X_train, X_test, Z_train, Z_test = train_test_split(X, z, test_size=test_size)

    # make main model
    beta, beta_var = model(X_train, Z_train, **kwargs)
    Z_hat = X_test@beta
    MSE, R2 = MSE_R2(Z_hat, Z_test)

    # bootstrap
    Z_pred = np.zeros((B, Z_test.shape[0]))

    for boot in range(B):
        # draw indexes for training
        idx = np.random.randint(0, X_train.shape[0], X_train.shape[0])

        # make bootstrap model
        beta, beta_var = model(X_train[idx], Z_train[idx], **kwargs)
        Z_pred[boot] = X_test@beta

    # compute statistics (pointwise)
    Z_diff = Z_test - Z_pred
    Z_pred_mean = np.mean(Z_pred, axis=0)

    Bias2_pw = np.mean(np.abs(Z_test - Z_pred_mean), axis=0)**2
    ModVar_pw = np.mean((Z_pred - Z_pred_mean)**2, axis=0)

    # bootstrap average
    Bias2 = np.array((np.mean(Bias2_pw), np.std(Bias2_pw)))
    ModVar = np.array((np.mean(ModVar_pw), np.std(ModVar_pw)))

    return MSE, Bias2, ModVar

if __name__ == "__main__":
    max_deg = 10
    N_points = [25, 40, 100]                             # number of data points
    N_pol = np.linspace(1, max_deg, max_deg).astype(int) # polynomial degrees

    MSE = np.zeros([len(N_points), len(N_pol)])
    bias = np.zeros([len(N_points), len(N_pol),2])
    var = np.zeros([len(N_points), len(N_pol),2])

    for i, N in enumerate(N_points):
        # create data
        x, y, z = generate_data_set(N, 1)

        # standardizing response
        z_var = np.var(z)
        z_mean = np.mean(z)

        z = (z-z_mean)/np.sqrt(z_var)

        for j, deg in enumerate(N_pol):
            # preparing data
            X = design_matrix(x, y, n=deg)

            # preprocess data
            scaler = StandardScaler()
            scaler.fit(X)
            X = scaler.transform(X)

            # do bootstrap
            MSE[i,j], bias[i,j], var[i,j] = bias_variance_tradeoff(X, z, 500)

        plt.errorbar(N_pol, bias[i,:,0], yerr=bias[i,:,1], fmt="o", capsize=7,
                     label="bias", color="darkorange")
        plt.errorbar(N_pol, var[i,:,0], yerr=var[i,:,1], fmt="o", capsize=7,
                     label="variance", color="brown")
        plt.scatter(N_pol, MSE[i], marker="o", label="MSE", color="red")
        plt.legend()
        plt.title(f"Bias variance tradeoff, N={N_points[i]}", fontsize=15)
        plt.xlabel("Polynomial degree", fontsize=12)
        plt.ylabel(r"Error$^2$", fontsize=12)
        plt.savefig(f"Figures/b-v_tradeoff_franke_N{N_points[i]}.png", dpi=300)
        plt.show()

    plt.scatter(N_pol, MSE[-1], marker="o", color="red")
    plt.title(f"Model MSE, N={N_points[-1]}", fontsize=15)
    plt.xlabel("Polynomial degree", fontsize=12)
    plt.ylabel("MSE", fontsize=12)
    plt.savefig(f"Figures/model_mse_OLS_N{N_points[-1]}.png", dpi=300)
    plt.show()
