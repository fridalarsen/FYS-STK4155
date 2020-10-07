import numpy as np
import matplotlib.pyplot as plt
from imageio import imread
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from a import design_matrix, MSE_R2
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

N = 5                                               # highest polynomial order
N_pol = np.linspace(1, N, N).astype(int)            # polynomial degrees
lambdas = np.logspace(-7, 1, 9)                     # penalties

# grid search for best lambdas and polynomials
MSE_test = np.zeros([N, len(lambdas)])
MSE_train = np.zeros([N, len(lambdas)])
R2_test = np.zeros([N, len(lambdas)])
R2_train = np.zeros([N, len(lambdas)])

for i, deg in enumerate(N_pol):
    for j, lambda_ in enumerate(lambdas):
        # prepare data
        X = design_matrix(x, y, n=deg)
        X_train, X_test, z_train, z_test = train_test_split(X, z, test_size=0.35)

        # preprocess data
        scaler = StandardScaler()
        scaler.fit(X_train)
        X_train = scaler.transform(X_train)
        X_test = scaler.transform(X_test)

        # fit
        beta, beta_var = Lasso(X_train, z_train, lambda_, r_var=1)

        # predict
        z_hat_train = X_train@beta
        z_hat_test = X_test@beta

        # quality check
        MSE_train[i-1,j], R2_train[i-1,j] = MSE_R2(z_hat_train, z_train)
        MSE_test[i-1,j], R2_test[i-1,j] = MSE_R2(z_hat_test, z_test)

centers = [np.log10(lambdas.min()), np.log10(lambdas.max()),
           N_pol.min(), N_pol.max()]
dx, = np.diff(centers[:2])/(MSE_test.shape[1]-1)
dy, = -np.diff(centers[2:])/(MSE_test.shape[0]-1)
extent = [centers[0]-dx/2, centers[1]+dx/2, centers[2]+dy/2, centers[3]-dy/2]
plt.imshow(MSE_test, cmap="RdYlGn", extent=extent, aspect="auto")
plt.xticks(np.arange(centers[0], centers[1]+dx, dx))
plt.yticks(np.arange(centers[3], centers[2]+dy, dy))
plt.xlabel(r"$\log(\lambda)$", fontsize=12)
plt.ylabel("Polynomial degree", fontsize=12)
plt.title("Lasso prediction errors", fontsize=15)
cb = plt.colorbar()
cb.set_label(label="Squared error", fontsize=12)
plt.savefig("Figures/lasso_grid_search.png", dpi=300)
plt.show()
