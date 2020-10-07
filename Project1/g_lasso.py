import numpy as np
import matplotlib.pyplot as plt
from imageio import imread
from sklearn.preprocessing import StandardScaler
from a import design_matrix, MSE_R2
from b import bias_variance_tradeoff
from c import kfold
from e import Lasso

# import data
terrain = imread("SRTM_data_Norway_2.tif")

x = np.linspace(0, terrain.shape[1], terrain.shape[1])
y = np.linspace(0, terrain.shape[0], terrain.shape[0])
X, Y = np.meshgrid(x,y)

x = X.flatten()
y = Y.flatten()
z = terrain.flatten()

# standardizing response
z_var = np.var(z)
z_mean = np.mean(z)

z = (z-z_mean)/np.sqrt(z_var)

N_pol = [2, 3, 4]                                  # polynomial orders to study
B = 5                                              # number of bootstrap samples
k = 5                                              # number of folds
lambdas = np.logspace(-7, 1, 9)                    # penalties

MSE_b = np.zeros([len(N_pol), len(lambdas)])
bias_b = np.zeros([len(N_pol), len(lambdas), 2])
var_b = np.zeros([len(N_pol), len(lambdas), 2])

MSE_kfold = np.zeros([len(N_pol), len(lambdas), 2])

for i, deg in enumerate(N_pol):
    # prepare data
    X = design_matrix(x, y, n=deg)

    # preprocess data
    scaler = StandardScaler()
    scaler.fit(X)
    X = scaler.transform(X)

    for j, lambda_ in enumerate(lambdas):
        # bootstrap
        MSE_b[i,j], bias_b[i,j], var_b[i,j] = bias_variance_tradeoff(X, z, B,
                                              model=Lasso, lambda_=lambda_)
        # kfold
        MSE_kfold[i,j] = kfold(k, X, z, model=Lasso, lambda_=lambda_)

    # bias-variance tradeoff
    plt.errorbar(lambdas, bias_b[i,:,0], yerr=bias_b[i,:,1], fmt="o", capsize=7,
                 label="bias", color="darkorange")
    plt.errorbar(lambdas, var_b[i,:,0], yerr=var_b[i,:,1], fmt="o",
                 capsize=7, label="variance", color="brown")
    plt.scatter(lambdas, MSE_b[i], marker="o", label="MSE", color="red")
    plt.legend()
    plt.xscale("log")
    plt.title(f"Bias variance tradeoff, deg={deg}", fontsize=15)
    plt.xlabel(r"$\lambda$", fontsize=12)
    plt.ylabel(r"Error$^2$", fontsize=12)
    xmin, xmax = plt.xlim()
    plt.savefig(f"Figures/b-v_tradeoff_terrain_lasso_deg{deg}.png", dpi=300)
    plt.show()

    plt.scatter(lambdas, MSE_b[i], marker="o", color="red")
    plt.xlim(xmin, xmax)
    plt.title(f"Model MSE, deg={deg}", fontsize=15)
    plt.xscale("log")
    plt.xlabel(r"$\lambda$", fontsize=12)
    plt.ylabel("MSE", fontsize=12)
    plt.savefig(f"Figures/model_mse_terrain_lasso_deg{deg}.png", dpi=300)
    plt.show()

    # kfold error analysis
    plt.errorbar(lambdas, MSE_kfold[i,:,0], yerr=MSE_kfold[i,:,1], fmt="o",
                 capsize=7, color="red")
    plt.xscale("log")
    plt.title(f"{k}fold cross validated MSE, deg={deg}", fontsize=15)
    plt.xlabel(r"$\lambda$", fontsize=12)
    plt.ylabel("MSE", fontsize=12)
    plt.savefig(f"Figures/kfold_mse_terrain_lasso_deg{deg}.png", dpi=300)
    plt.show()
