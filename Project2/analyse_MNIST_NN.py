import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from neural_network import NeuralNetwork
from neural_network import plot_confusion_matrix
from logistic_regression import MultipleLogisticRegression
from sklearn.metrics import accuracy_score as ACC
from matplotlib.colors import Normalize, LogNorm

# get dataset
digits = datasets.load_digits()
images = digits.images
values = digits.target

# plot example image
plt.imshow(images[8], cmap="Greys_r")
plt.xlabel("x", fontsize=12)
plt.ylabel("y", fontsize=12)
plt.title("MNIST image example", fontsize=15)
plt.savefig("Figures/MNIST_example.png", dpi=300)
plt.show()

# reshape image matrix
images = images.reshape(len(images), -1).astype(int)

# split data set
test_size = 0.35
train_size = 1-test_size
X_train, X_test, Z_train, Z_test = train_test_split(images, values, train_size=train_size,
                                                    test_size=test_size)

# scale data
scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

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
    NN = NeuralNetwork(n_hidden_layers, n_hidden_nodes, penalty,
                       activation="tanh", regression=False)
    NN.set_learning_params(a1, a2)
    NN.fit(X_train, Z_train, n_minibatches, n_epochs, std_W=std_W, const_b=const_b,
           track_cost=[X_test, Z_test])

    Z_pred = NN.classify(X_test)

    print(f"Neural Network with penalty lambda = {penalty}")
    print("  Accuracy score =", ACC(Z_test, Z_pred))

    plt.plot(np.arange(1,n_epochs+1), NN.cost, label=f"$\lambda={penalty:.2f}$")

plt.xlabel("Number of epochs", fontsize=12)
plt.ylabel("Cost function", fontsize=12)
plt.title("Evolution of cost function", fontsize=15)
plt.legend()
plt.savefig("Figures/NNcla_sgd_cost_function.png", dpi=300)
plt.show()

# grid search learning parameters

n_hidden_layers = 5
n_hidden_nodes = 15
penalty = 0
n_minibatches = 5
n_epochs = int(2e2)
a1 = np.logspace(-3.0, -2.0, 15)
a2 = np.linspace(0, 4, 15)
std_W = 0.1
const_b = 0

cost_learning = np.zeros([len(a1), len(a2)])

NN = NeuralNetwork(n_hidden_layers, n_hidden_nodes, penalty, activation="tanh",
                   regression=False)

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
plt.savefig("Figures/NNcla_learning_params_lin.png", dpi=300)
plt.show()

# grid search network architecture

n_hidden_layers = np.arange(1, 11, 5)
n_hidden_nodes = np.arange(5, 105, 30)
penalty = 0
n_minibatches = 5
n_epochs = int(2e2)
a1 = 10**(-2.3)
a2 = 0e0
std_W = 0.1
const_b = 0

accuracy = np.zeros([len(n_hidden_layers), len(n_hidden_nodes)])

np.seterr(all='raise')
for i, n in enumerate(n_hidden_layers):
    for j, m in enumerate(n_hidden_nodes):
        NN = NeuralNetwork(n, m, penalty, activation="tanh", regression=False)
        NN.set_learning_params(a1=a1, a2=a2)
        try:
            NN.fit(X_train, Z_train, n_minibatches, n_epochs, std_W=std_W, const_b=const_b)
            Z_pred = NN.classify(X_test)

            accuracy[i,j] = ACC(Z_test, Z_pred)
        except:
            print(f"Crash with #layers = {n}, #nodes = {m}.")
            accuracy[i,j] = np.nan
np.seterr(all='ignore')

N, M = n_hidden_layers, n_hidden_nodes

N_lims = np.array([np.min(N), np.max(N)])
M_lims = np.log10(np.array([np.min(M), np.max(M)]))
dx, = np.diff(M_lims)/(len(M)-1)
dy, = -np.diff(N_lims)/(len(N)-1)
extent = [M_lims[0]-dx/2, M_lims[1]+dx/2, N_lims[0]+dy/2, N_lims[1]-dy/2]
xlabels = [f"{m:d}" for m in M]
ylabels = [f"{n:d}" for n in N]

acc_vmin = np.nanmin(accuracy)
acc_vmax = np.nanmax(accuracy)

plt.imshow(accuracy, cmap="winter", extent=extent, aspect="auto",
           norm=Normalize(vmin=acc_vmin, vmax=acc_vmax))
plt.xticks(np.arange(M_lims[0], M_lims[1]+dx, dx), labels=xlabels)
plt.yticks(np.arange(N_lims[1], N_lims[0]+dy, dy), labels=ylabels)
plt.colorbar()
plt.xlabel("Number of hidden nodes", fontsize=12)
plt.ylabel("Number of hidden layers", fontsize=12)
plt.title("Accuracy score", fontsize=15)
plt.savefig("Figures/NNcla_acc_architecture.png", dpi=300)
plt.show()

# grid search penalty perameter

architecture = [[2, 10], [5, 20], [10, 50]]  # [layers, nodes]
penalties = np.logspace(-5, 0, 6)
N_repetitions = 3
n_minibatches = 5
n_epochs = int(2e2)
a1 = 2e-3
a2 = 1.2e0
std_W = 0.1
const_b = 0

accuracy = np.zeros((len(architecture), len(penalties), N_repetitions))

np.seterr(all='raise')
for i, (n,m) in enumerate(architecture):
    for j, penalty in enumerate(penalties):
        for k in range(N_repetitions):
            NN = NeuralNetwork(n, m, penalty, activation="tanh", regression=False)
            NN.set_learning_params(a1=a1, a2=a2)
            try:
                NN.fit(X_train, Z_train, n_minibatches, n_epochs, std_W=std_W, const_b=const_b)
                Z_pred = NN.classify(X_test)

                accuracy[i,j,k] = ACC(Z_test, Z_pred)
            except:
                print(f"Crash with penalty = {penalty}.")
                accuracy[i,j,k] = np.nan
np.seterr(all='ignore')

for acc, (n,m) in zip(accuracy, architecture):
    plt.errorbar(x=penalties, y=np.nanmean(acc, axis=1), yerr=np.nanstd(acc, axis=1),
                 fmt="o", label=f"{n} layers, {m} nodes", capsize=5, ms=5)
plt.xscale("log")
plt.xlabel(f"penalty ($\lambda$)", fontsize=12)
plt.ylabel("accuracy", fontsize=12)
plt.title("Accuracy score", fontsize=15)
plt.legend()
plt.savefig("Figures/NNreg_acc_penalties.png", dpi=300)
plt.show()
