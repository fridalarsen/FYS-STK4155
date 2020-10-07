import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from imageio import imread
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from a import design_matrix, design_matrix_column_order, LinearRegression, \
              MSE_R2, prep_surface
from b import bias_variance_tradeoff
from c import kfold

# import data
terrain = imread("SRTM_data_Norway_2.tif")

x = np.linspace(0, terrain.shape[1], terrain.shape[1])
y = np.linspace(0, terrain.shape[0], terrain.shape[0])
X, Y = np.meshgrid(x,y)

x = X.flatten()
y = Y.flatten()
z = terrain.flatten()

# standardize response
z_var = np.var(z)
z_mean = np.mean(z)

z = (z-z_mean)/np.sqrt(z_var)

N = 5                                       # highest polynomial order
B = 5                                       # number of bootstrap samples
k = 5                                       # number of folds

MSE_test = np.zeros(N)
MSE_train = np.zeros(N)
R2_test = np.zeros(N)
R2_train = np.zeros(N)

MSE_boots = np.zeros(N)
bias_boots = np.zeros([N, 2])
var_boots = np.zeros([N, 2])

MSE_kfold = np.zeros(N)
std_kfold = np.zeros(N)

for i in range(1, N+1):
    # prepare data
    X = design_matrix(x, y, n=i)
    X_train, X_test, z_train, z_test = train_test_split(X, z, test_size=0.35)

    # preprocess data
    scaler_train = StandardScaler()
    scaler_train.fit(X_train)
    X_train = scaler_train.transform(X_train)
    X_test = scaler_train.transform(X_test)

    scaler_X = StandardScaler()
    scaler_X.fit(X)
    X = scaler_X.transform(X)

    # fit
    beta, beta_var = LinearRegression(X_train, z_train, r_var=1)

    # predict
    z_hat_train = X_train@beta
    z_hat_test = X_test@beta

    # quality check
    MSE_train[i-1], R2_train[i-1] = MSE_R2(z_hat_train, z_train)
    MSE_test[i-1], R2_test[i-1] = MSE_R2(z_hat_test, z_test)

    # bootstrap
    MSE_boots[i-1], bias_boots[i-1], var_boots[i-1] = bias_variance_tradeoff(X, z, B)

    # kfold
    MSE_kfold[i-1], std_kfold[i-1] = kfold(k, X, z)

    print(f"Completed loops: {i}")

    # plot example approximations
    if i == 5 or i == 10:
        x_surf = np.linspace(0, terrain.shape[1], terrain.shape[1])
        y_surf = np.linspace(0, terrain.shape[0], terrain.shape[0])

        X_surf, Y_surf, Z_surf = prep_surface(x_surf, y_surf, beta, deg=i,
                                              scaler=scaler_train)

        fig = plt.figure()
        ax = plt.axes(projection="3d")
        ax.plot_surface(X_surf, Y_surf, Z_surf, cmap="autumn")
        ax.set_title(f"Terrain OLS approximation, deg={i}", fontsize=15)
        ax.set_xlabel("x", fontsize=12)
        ax.set_ylabel("y", fontsize=12)
        ax.set_zlabel("z", fontsize=12)
        plt.savefig(f"Figures/terrain_OLS_deg{i}.png", dpi=300)
        plt.show()

        plt.imshow((Z_surf - terrain)**2, cmap="RdYlGn",
                   extent=[X_surf.min(), X_surf.max(), Y_surf.max(),
                   Y_surf.min()])
        plt.title(f"OLS prediction error, deg={i}", fontsize=15)
        plt.xlabel("x", fontsize=12)
        plt.ylabel("y", fontsize=12)
        cb = plt.colorbar()
        cb.set_label(label="Squared error", fontsize=12)
        plt.savefig(f"Figures/terrain_OLS_error_deg{i}.png", dpi=300)
        plt.show()

# plain OLS error analysis
complexity = np.linspace(1, N, N)
plt.plot(complexity, MSE_test, label="Test Sample", color="orange")
plt.plot(complexity, MSE_train, label="Training Sample", color="darkred")
plt.legend()
plt.subplots_adjust(left=0.16)
plt.xlabel("Polynomial degree", fontsize=12)
plt.ylabel("Mean Squared Error", fontsize=12)
plt.title("Model Complexity Optimization", fontsize=15)
plt.savefig("Figures/model_complexity_mse_terrain.png", dpi=300)
plt.show()

# bias-variance tradeoff
plt.errorbar(complexity, bias_boots[:,0], yerr=bias_boots[:,1], fmt="o", capsize=7,
             label="bias", color="darkorange")
plt.errorbar(complexity, var_boots[:,0], yerr=var_boots[:,1], fmt="o", capsize=7,
             label="variance", color="brown")
plt.scatter(complexity, MSE_boots, marker="o", label="MSE", color="red")
plt.legend()
plt.title(f"Bias variance tradeoff", fontsize=15)
plt.xlabel("Polynomial degree", fontsize=12)
plt.ylabel(r"Error$^2$", fontsize=12)
plt.savefig(f"Figures/b-v_tradeoff_terrain_OLS.png", dpi=300)
plt.show()

plt.scatter(complexity, MSE_boots, marker="o", color="red")
plt.title(f"Model MSE", fontsize=15)
plt.xlabel("Polynomial degree", fontsize=12)
plt.ylabel("MSE", fontsize=12)
plt.savefig(f"Figures/model_mse_terrain_OLS.png", dpi=300)
plt.show()

# kfold error analysis
plt.errorbar(complexity, MSE_kfold, yerr=std_kfold, fmt="o", capsize=7, color="red")
plt.title(f"{k}fold cross validated MSE", fontsize=15)
plt.xlabel("Polynomial degree", fontsize=12)
plt.ylabel("MSE", fontsize=12)
plt.savefig(f"Figures/kfold_mse_terrain_OLS.png", dpi=300)
plt.show()
