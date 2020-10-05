import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from generate_data_set import generate_data_set
from a import design_matrix, LinearRegression, MSE_R2

def kfold(k, X, z, model=LinearRegression, **kwargs):
    """
    Function for implementing kfold cross validation.
    Arguments:
        k (int): number of folds
        X (array): design matrix (input data)
        z (array): response
        model (function, optional): model to cross validate, defaults to
                                    LinearRegression
        **kwargs: additional arguments for model
    Returns:
        MSE_mean (float): kfold estimate of MSE
        MSE_std (float): standard deviation of MSE estimate
    """
    idx = np.random.choice(X.shape[0], size=X.shape[0], replace=False)
    split = np.array_split(idx, k)

    MSE = np.zeros(k)

    for i in range(k):
        # split data
        test_idx = split[i]
        train_idx = np.concatenate([split[j] for j in range(k) if j != i])

        X_train = X[test_idx]
        z_train = z[test_idx]
        X_test = X[train_idx]
        z_test = z[train_idx]

        # fit
        beta, beta_var = model(X_train, z_train, r_var=1, **kwargs)

        # predict
        z_hat = X_test@beta

        # quality check
        MSE[i], R2 = MSE_R2(z_hat, z_test)

    MSE_mean = np.mean(MSE)
    MSE_std = np.std(MSE)

    return MSE_mean, MSE_std


if __name__ == "__main__":
    # create data
    N = 100
    x, y, z = generate_data_set(N, 1)

    # standardize response
    z_var = np.var(z)
    z_mean = np.mean(z)

    z = (z-z_mean)/np.sqrt(z_var)

    max_deg = 8

    N_pol = np.linspace(1, max_deg, max_deg).astype(int)

    MSE = np.zeros(len(N_pol))
    std = np.zeros(len(N_pol))
    k = 5
    for j, deg in enumerate(N_pol):
        # prepare data
        X = design_matrix(x, y, n=deg)

        # preprocess data
        scaler = StandardScaler()
        scaler.fit(X)
        X = scaler.transform(X)

        # do kfold
        MSE[j], std[j] = kfold(k, X, z)


    plt.errorbar(N_pol, MSE, yerr=std, fmt="o", capsize=7, color="red")
    plt.title(f"{k}fold cross validated MSE, N={N}", fontsize=15)
    plt.xlabel("Polynomial degree", fontsize=12)
    plt.ylabel("MSE", fontsize=12)
    plt.savefig(f"Figures/kfold_mse_N{N}.png", dpi=300)
    plt.show()
