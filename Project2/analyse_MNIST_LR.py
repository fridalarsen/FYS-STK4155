import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from logistic_regression import MultipleLogisticRegression as LogReg
from sklearn.metrics import accuracy_score as ACC
from matplotlib.colors import Normalize, LogNorm
from neural_network import plot_confusion_matrix

# get dataset
digits = datasets.load_digits()
images = digits.images
values = digits.target

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

penalties = [0, 0.01, 0.1, 1]
n_minibatches = 5
n_epochs = int(2e2)
a1 = 1.0e-2
a2 = 1.0e-2
std_beta = 1

for penalty in penalties:
    LR = LogReg(penalty)
    LR.set_learning_params(a1, a2)
    LR.fit(X_train, Z_train, n_minibatches, n_epochs, std_beta=std_beta)
    Z_pred = LR.predict(X_test)

    C = np.zeros(len(LR.beta_path))
    for i, beta in enumerate(LR.beta_path):
        C[i] = LR.C(Z_test, Z_pred, beta=beta)

    plt.plot(np.arange(1,len(C)+1), C, label=f"$\lambda$={penalty}")

    Z_pred = LR.classify(X_test)

    print(f"Log Reg with penalty lambda = {penalty}")
    print("  Accuracy score =", ACC(Z_test, Z_pred))

plt.xlabel("Number of epochs", fontsize=12)
plt.ylabel("Cost function", fontsize=12)
plt.title("Evolution of cost function", fontsize=15)
plt.legend()
plt.savefig("Figures/LR_sgd_cost_function.png", dpi=300)
plt.show()

plot_confusion_matrix(Z_test, Z_pred, normalize=False)

# grid search learning parameters

penalty = 0
n_minibatches = 5
n_epochs = int(2e2)
a1 = np.logspace(-4.0, -1.0, 12)
a2 = np.logspace(-5, 1, 13)
std_beta = 1

cost_learning = np.zeros([len(a1), len(a2)])

LR = LogReg(penalty)

np.seterr(all='raise')
for i, a in enumerate(a1):
    for j, b in enumerate(a2):
        try:
            LR.set_learning_params(a=a, b=b)
            LR.fit(X_train, Z_train, n_minibatches, n_epochs, std_beta=std_beta)
            Z_pred = LR.predict(X_test)

            cost_learning[i,j] = LR.C(Z_test, Z_pred)
        except:
            print(f"Crash with a = {a}, b = {b}.")
            cost_learning[i,j] = np.nan
np.seterr(all='ignore')

a2_lims = np.log10(np.array([np.min(a2), np.max(a2)]))
a1_lims = np.log10(np.array([np.min(a1), np.max(a1)]))
dx, = np.diff(a2_lims)/(len(a2)-1)
dy, = -np.diff(a1_lims)/(len(a1)-1)
extent = [a2_lims[0]-dx/2, a2_lims[1]+dx/2, a1_lims[0]+dy/2, a1_lims[1]-dy/2]
xlabels = [f"{a2_:.1f}" for a2_ in np.log10(a2)]
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
plt.xlabel("log(b)", fontsize=12)
plt.ylabel("log(a)", fontsize=12)
plt.title("Cost function", fontsize=15)
plt.savefig("Figures/LR_learning_params_lin.png", dpi=300)
plt.show()

# grid search penalty perameter

penalties = np.logspace(-3, 2, 21)
N_repetitions = 3
n_minibatches = 5
n_epochs = int(2e2)
a = 2e-3
b = 1.2e0
std_beta = 1.0

accuracy = np.zeros((len(penalties), N_repetitions))

np.seterr(all='raise')
for j, penalty in enumerate(penalties):
    LR = LogReg(penalty)
    LR.set_learning_params(a=a, b=b)
    for k in range(N_repetitions):
        try:
            LR.fit(X_train, Z_train, n_minibatches, n_epochs, std_beta=std_beta)
            Z_pred = LR.classify(X_test)

            accuracy[j,k] = ACC(Z_test, Z_pred)
        except:
            print(f"Crash with penalty = {penalty}.")
            accuracy[j,k] = np.nan
np.seterr(all='ignore')

plt.errorbar(x=penalties, y=np.nanmean(accuracy, axis=1),
             yerr=np.nanstd(accuracy, axis=1), fmt="o", capsize=5, ms=5)
plt.xscale("log")
plt.xlabel(f"penalty ($\lambda$)", fontsize=12)
plt.ylabel("accuracy", fontsize=12)
plt.title("Accuracy score", fontsize=15)
plt.savefig("Figures/LR_acc_penalties.png", dpi=300)
plt.show()
