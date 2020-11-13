import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error as MSE
from sklearn.metrics import r2_score as R2
from sklearn.preprocessing import StandardScaler
from franke import generate_franke_data
from neural_network import NeuralNetwork
from sklearn.model_selection import train_test_split
from linear_regression import Ridge
from matplotlib.colors import Normalize, LogNorm

# generate data
Nx = 30
Ny = 30
sigma = 0.1
x, y, z = generate_franke_data(Nx, Ny, sigma)

# create design matrix
X = np.c_[x, y]

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

n_hidden_layers = 5
n_hidden_nodes = 15
penalties = [0, 0.01, 0.1, 1]
n_minibatches = 5
n_epochs = int(2e2)
a1 = 2.0e-3
a2 = 1.2e0
std_W = 0.1
const_b = 0

for penalty in penalties:
    NN = NeuralNetwork(n_hidden_layers, n_hidden_nodes, penalty, activation="tanh")
    NN.set_learning_params(a1, a2)
    NN.fit(X_train, Z_train, n_minibatches, n_epochs, std_W=std_W, const_b=const_b,
           track_cost=[X_test, Z_test])

    Z_pred = NN.predict(X_test)

    print(f"Neural Network with penalty lambda = {penalty}")
    print("  MSE score =", MSE(Z_test, Z_pred))
    print("  R2 score  =", R2(Z_test, Z_pred))

    plt.plot(np.arange(1,n_epochs+1), NN.cost, label=f"$\lambda={penalty:.2f}$")

plt.xlabel("Number of epochs", fontsize=12)
plt.ylabel("Cost function", fontsize=12)
#plt.xscale("log")
#plt.yscale("log")
plt.title("Evolution of cost function", fontsize=15)
plt.legend()
#plt.savefig("Figures/NN_sgd_cost_function.png", dpi=300)
plt.show()

# test with Ridge
R = Ridge(0)
R.fit_minv(X_train, Z_train)
Z_pred = R.predict(X_test)

print("OLS:")
print("  MSE score =", MSE(Z_test, Z_pred))
print("  R2 score  =", R2(Z_test, Z_pred))


# grid search learning parameters

n_hidden_layers = 5
n_hidden_nodes = 15
penalty = 0
n_minibatches = 5
n_epochs = int(2e2)
#a1 = np.linspace(1e-4, 7.0e-3, 5)
a1 = np.logspace(-3.6, -1.8, 15)
a2 = np.linspace(0, 8, 15)
std_W = 0.1
const_b = 0

cost_learning = np.zeros([len(a1), len(a2)])

NN = NeuralNetwork(n_hidden_layers, n_hidden_nodes, penalty, activation="tanh")

np.seterr(all='raise')
for i, a in enumerate(a1):
    for j, b in enumerate(a2):
        try:
            NN.set_learning_params(a1=a, a2=b)
            NN.fit(X_train, Z_train, n_minibatches, n_epochs, std_W=std_W, const_b=const_b)
            Z_pred = NN.predict(X_test)

            cost_learning[i,j] = NN.C.C(Z_test, Z_pred, penalty=penalty, W=NN.W)
        except:
            print(f"Crash with a1 = {a}, a2 = {b}.")
            cost_learning[i,j] = np.nan
np.seterr(all='ignore')

a2_lims = np.array([np.min(a2), np.max(a2)])
a1_lims = np.log10(np.array([np.min(a1), np.max(a1)]))
dx, = np.diff(a2_lims)/(len(a2)-1)
dy, = -np.diff(a1_lims)/(len(a1)-1)
extent = [a2_lims[0]-dx/2, a2_lims[1]+dx/2, a1_lims[0]+dy/2, a1_lims[1]-dy/2]
xlabels = [f"{a2_:.1f}" for a2_ in a2]
ylabels = [f"{a1_:.1f}" for a1_ in np.log10(a1)]

mask = ~np.isnan(cost_learning)
cost_learning_real = cost_learning[mask]
vmin = np.nanmin(cost_learning)
vmax = np.max( cost_learning_real[cost_learning_real <= 1e3] )

plt.imshow(cost_learning, cmap="autumn_r", extent=extent, aspect="auto",
           norm=Normalize(vmin=vmin, vmax=vmax))
plt.xticks(np.arange(a2_lims[0], a2_lims[1]+dx, dx), labels=xlabels)
plt.yticks(np.arange(a1_lims[1], a1_lims[0]+dy, dy), labels=ylabels)
plt.colorbar()
plt.xlabel("a2", fontsize=12)
plt.ylabel("log(a1)", fontsize=12)
plt.title("Cost function", fontsize=15)
plt.savefig("Figures/NNreg_learning_params_lin.png", dpi=300)
plt.show()

plt.imshow(cost_learning, cmap="autumn_r", extent=extent, aspect="auto",
           norm=LogNorm(vmin=vmin, vmax=vmax))
plt.xticks(np.arange(a2_lims[0], a2_lims[1]+dx, dx), labels=xlabels)
plt.yticks(np.arange(a1_lims[1], a1_lims[0]+dy, dy), labels=ylabels)
plt.colorbar()
plt.xlabel("a2", fontsize=12)
plt.ylabel("log(a1)", fontsize=12)
plt.title("Cost function", fontsize=15)
plt.savefig("Figures/NNreg_learning_params_log.png", dpi=300)
plt.show()

# grid search network architecture

n_hidden_layers = np.arange(1, 11, 1)
n_hidden_nodes = np.arange(5, 105, 5)
penalty = 0
n_minibatches = 5
n_epochs = int(2e2)
a1 = 2e-3
a2 = 1.2e0
std_W = 0.1
const_b = 0

MSE_ = np.zeros([len(n_hidden_layers), len(n_hidden_nodes)])
R2_ = np.zeros([len(n_hidden_layers), len(n_hidden_nodes)])

np.seterr(all='raise')
for i, n in enumerate(n_hidden_layers):
    for j, m in enumerate(n_hidden_nodes):
        NN = NeuralNetwork(n, m, penalty, activation="tanh")
        NN.set_learning_params(a1=a1, a2=a2)
        try:
            NN.fit(X_train, Z_train, n_minibatches, n_epochs, std_W=std_W, const_b=const_b)
            Z_pred = NN.predict(X_test)

            MSE_[i,j] = MSE(Z_test, Z_pred)
            R2_[i,j] = R2(Z_test, Z_pred)
        except:
            print(f"Crash with #layers = {n}, #nodes = {m}.")
            MSE_[i,j] = np.nan
            R2_[i,j] = np.nan
np.seterr(all='ignore')

# prevent negative R2
R2_[R2_ <= 0] = np.nan

N, M = n_hidden_layers, n_hidden_nodes

N_lims = np.array([np.min(N), np.max(N)])
M_lims = np.log10(np.array([np.min(M), np.max(M)]))
dx, = np.diff(M_lims)/(len(M)-1)
dy, = -np.diff(N_lims)/(len(N)-1)
extent = [M_lims[0]-dx/2, M_lims[1]+dx/2, N_lims[0]+dy/2, N_lims[1]-dy/2]
xlabels = [f"{m:d}" for m in M]
ylabels = [f"{n:d}" for n in N]

MSE_vmin = np.min(MSE_)
MSE_vmax = np.max(MSE_)
R2_vmin = np.nanmin(R2_)
R2_vmax = np.nanmax(R2_)

plt.imshow(MSE_, cmap="winter", extent=extent, aspect="auto",
           norm=LogNorm(vmin=MSE_vmin, vmax=MSE_vmax))
plt.xticks(np.arange(M_lims[0], M_lims[1]+dx, dx), labels=xlabels)
plt.yticks(np.arange(N_lims[1], N_lims[0]+dy, dy), labels=ylabels)
plt.colorbar()
plt.xlabel("Number of hidden nodes", fontsize=12)
plt.ylabel("Number of hidden layers", fontsize=12)
plt.title("Mean Squared Error", fontsize=15)
plt.savefig("Figures/NNreg_MSE_architecture.png", dpi=300)
plt.show()

plt.imshow(R2_, cmap="winter_r", extent=extent, aspect="auto",
           norm=LogNorm(vmin=R2_vmin, vmax=R2_vmax))
plt.xticks(np.arange(M_lims[0], M_lims[1]+dx, dx), labels=xlabels)
plt.yticks(np.arange(N_lims[1], N_lims[0]+dy, dy), labels=ylabels)
plt.colorbar()
plt.xlabel("Number of hidden nodes", fontsize=12)
plt.ylabel("Number of hidden layers", fontsize=12)
plt.title("R2-score", fontsize=15)
plt.savefig("Figures/NNreg_R2_architecture.png", dpi=300)
plt.show()

# grid search penalty parameter

architecture = [[2, 10], [5, 20], [10, 50]]  # [layers, nodes]
penalties = np.logspace(-5, 0, 6)
N_repetitions = 3
n_minibatches = 5
n_epochs = int(2e2)
a1 = 2e-3
a2 = 1.2e0
std_W = 0.1
const_b = 0

MSE_ = np.zeros((len(architecture), len(penalties), N_repetitions))
R2_ = np.zeros((len(architecture), len(penalties), N_repetitions))

np.seterr(all='raise')
for i, (n,m) in enumerate(architecture):
    for j, penalty in enumerate(penalties):
        for k in range(N_repetitions):
            NN = NeuralNetwork(n, m, penalty, activation="tanh")
            NN.set_learning_params(a1=a1, a2=a2)
            try:
                NN.fit(X_train, Z_train, n_minibatches, n_epochs, std_W=std_W, const_b=const_b)
                Z_pred = NN.predict(X_test)

                MSE_[i,j,k] = MSE(Z_test, Z_pred)
                R2_[i,j,k] = R2(Z_test, Z_pred)
            except:
                print(f"Crash with penalty = {penalty}.")
                MSE_[i,j,k] = np.nan
                R2_[i,j,k] = np.nan
np.seterr(all='ignore')

for mse, (n,m) in zip(MSE_, architecture):
    plt.errorbar(x=penalties, y=np.nanmean(mse, axis=1), yerr=np.nanstd(mse, axis=1),
                 fmt="o", label=f"{n} layers, {m} nodes", capsize=5, ms=5)
plt.xscale("log")
plt.xlabel(f"penalty ($\lambda$)", fontsize=12)
plt.ylabel("MSE", fontsize=12)
plt.title("Mean Squared Error", fontsize=15)
plt.legend()
plt.savefig("Figures/NNreg_MSE_penalties.png", dpi=300)
plt.show()

for r2, (n,m) in zip(R2_, architecture):
    plt.errorbar(x=penalties, y=np.nanmean(r2, axis=1), yerr=np.nanstd(r2, axis=1),
                 fmt="o", label=f"{n} layers, {m} nodes", capsize=5, ms=5)
plt.xscale("log")
plt.xlabel(f"penalty ($\lambda$)", fontsize=12)
plt.ylabel(f"R$^2$", fontsize=12)
plt.title("R2-score", fontsize=15)
plt.legend()
plt.savefig("Figures/NNreg_R2_penalties.png", dpi=300)
plt.show()
