import numpy as np
import matplotlib.pyplot as plt
from imageio import imread
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from a import design_matrix, MSE_R2, prep_surface
from b import bias_variance_tradeoff
from c import kfold
from d import Ridge

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
lambdas = np.logspace(-3, 5, 9)                    # penalties

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
                                              model=Ridge, lambda_=lambda_)
        # kfold
        MSE_kfold[i,j] = kfold(k, X, z, model=Ridge, lambda_=lambda_)

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
    plt.savefig(f"Figures/b-v_tradeoff_terrain_ridge_deg{deg}.png", dpi=300)
    plt.show()

    plt.scatter(lambdas, MSE_b[i], marker="o", color="red")
    plt.xlim(xmin, xmax)
    plt.title(f"Model MSE, deg={deg}", fontsize=15)
    plt.xscale("log")
    plt.xlabel(r"$\lambda$", fontsize=12)
    plt.ylabel("MSE", fontsize=12)
    plt.savefig(f"Figures/model_mse_terrain_ridge_deg{deg}.png", dpi=300)
    plt.show()

    # kfold error analysis
    plt.errorbar(lambdas, MSE_kfold[i,:,0], yerr=MSE_kfold[i,:,1], fmt="o",
                 capsize=7, color="red")
    plt.xscale("log")
    plt.title(f"{k}fold cross validated MSE, deg={deg}", fontsize=15)
    plt.xlabel(r"$\lambda$", fontsize=12)
    plt.ylabel("MSE", fontsize=12)
    plt.savefig(f"Figures/kfold_mse_terrain_ridge_deg{deg}.png", dpi=300)
    plt.show()

# make an example approximation
# prepare data
X = design_matrix(x, y, n=4)
X_train, X_test, z_train, z_test = train_test_split(X, z, test_size=0.35)

# preprocess data
scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

# fit
beta, beta_var = Ridge(X_train, z_train, lambda_=1e1, r_var=1)

x_surf = np.linspace(0, terrain.shape[1], terrain.shape[1])
y_surf = np.linspace(0, terrain.shape[0], terrain.shape[0])

X_surf, Y_surf, Z_surf = prep_surface(x_surf, y_surf, beta, deg=4)

fig = plt.figure()
ax = plt.axes(projection="3d")
ax.plot_surface(X_surf, Y_surf, Z_surf, cmap="autumn")
ax.set_title(f"Terrain data Ridge approximation,\n deg=4 and " \
             + r"$\lambda=10$", fontsize=15)
ax.set_xlabel("x", fontsize=12)
ax.set_ylabel("y", fontsize=12)
ax.set_zlabel("z", fontsize=12)
plt.savefig(f"Figures/terrain_ridge_deg4.png", dpi=300)
plt.show()

plt.imshow((Z_surf - terrain)**2,
           cmap="RdYlGn", origin="lower",
           extent=[X_surf.min(), X_surf.max(), Y_surf.min(),
           Y_surf.max()])
plt.title(f"Difference between prediction and true value,\n"\
          + "deg=4 and " + r"$\lambda=10$", fontsize=15)
plt.xlabel("x", fontsize=12)
plt.ylabel("y", fontsize=12)
plt.subplots_adjust(left=0, right=0.79)
cb = plt.colorbar()
cb.set_label(label="Squared error", fontsize=12)
plt.savefig(f"Figures/terrain_ridge_error_deg4.png", dpi=300)
plt.show()
