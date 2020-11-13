import numpy as np
import matplotlib.pyplot as plt
from linear_regression import Ridge
from franke import generate_franke_data
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error as MSE
from sklearn.metrics import r2_score as R2
from sklearn.model_selection import train_test_split
np.seterr(all='raise')

# generate data
Nx = 20
Ny = 20
sigma = 0.1
x, y, z = generate_franke_data(Nx, Ny, sigma)

# create design matrix
X = np.c_[x, y, x**2, x*y, y**2, x**3, (x**2)*y, x*(y**2), y**3]

# split data set
test_size = 0.35
train_size = 1-test_size
X_train, X_test, Z_train, Z_test = train_test_split(X, z, train_size=train_size,
                                                    test_size=test_size)

# scale data
scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

scaler2 = StandardScaler()
scaler2.fit(Z_train.reshape(-1,1))
Z_train = scaler2.transform(Z_train.reshape(-1,1)).flatten()
Z_test = scaler2.transform(Z_test.reshape(-1,1)).flatten()

# investigate convergence
penalties = [0, 1, 5, 10]
a = 1e-3
b = 1e0

n_minibatches = 5
n_epochs = 1000

for lm, penalty in enumerate(penalties):
    Ridge1 = Ridge(penalty=penalty)
    Ridge1.set_learning_params(a, b)

    Ridge1.fit_sgd(X_train, Z_train, n_minibatches, n_epochs)
    betas_sgd = Ridge1.beta_path

    C = np.zeros(len(betas_sgd))
    for i in range(len(C)):
        C[i] = Ridge1.C(X_test, Z_test, betas_sgd[i])

    plt.plot(np.arange(1,len(C)+1), C, label=f"$\lambda$={penalty}")
plt.xlabel("Number of epochs", fontsize=12)
plt.ylabel("Cost function", fontsize=12)
plt.title("Evolution of cost function in SGD", fontsize=15)
plt.legend()
plt.yscale("log")
plt.xscale("log")
plt.savefig("Figures/ridge_sgd_cost_function.png", dpi=300)
plt.show()

# grid search - learning parameters

# learning parameters
a = np.logspace(-4, -2, 12)
b = np.logspace(-1, 1, 12)
beta_diff = np.zeros([len(a), len(b)])
C = np.zeros([len(a), len(b)])
N = 5

# perfect beta
model = Ridge(penalty=0)
model.fit_minv(X_train, Z_train)
b_minv = model.beta

for i, a_ in enumerate(a):
    for j, b_ in enumerate(b):
        try:
            beta_diffs = 0
            C_ = 0
            for n in range(N):
                model.set_learning_params(a=a_, b=b_)
                model.fit_sgd(X_train, Z_train, n_minibatches, n_epochs)
                b_sgd = model.beta

                C_ += model.C(X_train, Z_train, b_sgd)

                beta_diffs += np.sum(np.abs((b_minv-b_sgd)/b_minv))/len(b_minv)
            beta_diff[i,j] = beta_diffs/N
            C[i,j] = C_/N
        except:
            beta_diff[i,j] = -1
            print(f"Crash for a = {a_} and b = {b_} at n = {n}.")
            C[i,j] = -1

# maximum value in colorbar
betadiff_max = 3
C_max = 1e3

mask = (beta_diff>=0)&(beta_diff<=betadiff_max)
beta_diff_ = np.zeros(beta_diff.shape)
beta_diff_[mask] = beta_diff[mask]
beta_diff_[~mask] = np.max(beta_diff_)

mask = (C>=0)&(C<=C_max)
C_ = np.zeros(C.shape)
C_[mask] = C[mask]
C_[~mask] = np.max(C_)

b_lims = np.log10(np.array([np.min(b), np.max(b)]))
a_lims = np.log10(np.array([np.min(a), np.max(a)]))
dx, = np.diff(b_lims)/(len(b)-1)
dy, = -np.diff(a_lims)/(len(a)-1)
extent = [b_lims[0]-dx/2, b_lims[1]+dx/2, a_lims[0]+dy/2, a_lims[1]-dy/2]
xlabels = [f"{b_:.2f}" for b_ in np.log10(b)]
ylabels = [f"{a_:.2f}" for a_ in np.log10(a)]

plt.imshow(beta_diff_, cmap="autumn_r", extent=extent, aspect="auto")
plt.xticks(np.arange(b_lims[0], b_lims[1]+dx, dx), labels=xlabels)
plt.yticks(np.arange(a_lims[1], a_lims[0]+dy, dy), labels=ylabels)
plt.colorbar()
plt.xlabel("log(b)", fontsize=12)
plt.ylabel("log(a)", fontsize=12)
plt.title("Beta diff")
plt.savefig("Figures/ridge_sgd_beta_diff.png", dpi=300)
plt.show()

plt.imshow(C_, cmap="autumn_r", extent=extent, aspect="auto")
plt.xticks(np.arange(b_lims[0], b_lims[1]+dx, dx), labels=xlabels)
plt.yticks(np.arange(a_lims[1], a_lims[0]+dy, dy), labels=ylabels)
plt.colorbar()
plt.xlabel("log(b)", fontsize=12)
plt.ylabel("log(a)", fontsize=12)
plt.title("Cost function")
plt.savefig("Figures/ridge_sgd_C.png", dpi=300)
plt.show()

# investigate minibatches

penalty = 0
a = 1e-3
b = 1e0
n_minibatches = [2, 10, 30, 50]
n_epochs = 1000

Ridge1 = Ridge(penalty=penalty)
Ridge1.set_learning_params(a, b)
for n, nm in enumerate(n_minibatches):
    Ridge1.fit_sgd(X_train, Z_train, nm, n_epochs)
    betas_sgd = Ridge1.beta_path

    C = np.zeros(len(betas_sgd))
    for i in range(len(C)):
        C[i] = Ridge1.C(X_test, Z_test, betas_sgd[i])

    plt.plot(np.arange(2,len(C)+1), C[1:], label=f"n={nm}")
plt.xlabel("Number of epochs", fontsize=12)
plt.ylabel("Cost function", fontsize=12)
plt.title("Evolution of cost function in SGD", fontsize=15)
plt.legend()
plt.yscale("log")
plt.xscale("log")
plt.savefig("Figures/ridge_sgd_cost_function2.png", dpi=300)
plt.show()
