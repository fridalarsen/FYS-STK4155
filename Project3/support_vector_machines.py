import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import accuracy_score as ACC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold as KFold
from sklearn.svm import SVC
from confusion_matrix import plot_confusion_matrix

# read data
data = pd.read_csv("heart_failure_data.csv")

# prepare dataframe
death = data[["DEATH_EVENT"]]
data = data.drop("DEATH_EVENT", axis=1)
data = data.drop("time", axis=1)
continuous_features = ["age", "creatinine_phosphokinase", "ejection_fraction",
                       "platelets", "serum_creatinine", "serum_sodium"]

# kfold for kernel and penalty parameter
kernels = ["linear", "poly", "rbf"]
penalties = np.logspace(-5, 4, 17)
#penalties = np.linspace(1, 10, 10)

N = 5
kfold = KFold(n_splits=N, shuffle=True)

acc_mean = np.zeros([len(kernels), len(penalties)])
acc_std = np.zeros([len(kernels), len(penalties)])
best_penalties = np.zeros(len(kernels))

accuracy_kfold = np.zeros(N)
for i, kernel in enumerate(kernels):
    for j, penalty in enumerate(penalties):
        model = SVC(C=1/float(penalty), kernel=kernel, gamma="scale")
        for k, (train_index, test_index) in enumerate(kfold.split(data, death)):
            x_train = data.iloc[train_index]
            y_train = np.ravel(death.iloc[train_index])
            x_test = data.iloc[test_index]
            y_test = np.ravel(death.iloc[test_index])

            # extract continuous features
            features_to_extract = [f for f in continuous_features if f in data]
            x_train_cont = x_train[features_to_extract]
            x_test_cont = x_test[features_to_extract]

            # scale continuous data
            scaler = StandardScaler()
            scaler.fit(x_train_cont)
            x_train_cont = scaler.transform(x_train_cont)
            x_test_cont = scaler.transform(x_test_cont)

            # fill scaled data
            with pd.option_context('mode.chained_assignment', None):
                for l,f in enumerate(features_to_extract):
                    x_train.loc[:,f] = x_train_cont[:,l]
                    x_test.loc[:,f] = x_test_cont[:,l]

            model.fit(x_train, y_train)
            y_pred = model.predict(x_test)

            accuracy_kfold[k] = ACC(y_test, y_pred)
        acc_mean[i,j] = accuracy_kfold.mean()
        acc_std[i,j] = accuracy_kfold.std()
    plt.errorbar(penalties, acc_mean[i,:], yerr=acc_std[i,:], label=kernel,
                 fmt="o", capsize=5, markersize=7)
    best_penalties[i] = penalties[np.argmax(acc_mean[i,:])]

print("Best penalties:", best_penalties)

plt.legend()
plt.xscale("log")
plt.xlabel("Penalty", fontsize=12)
plt.ylabel("Accuracy", fontsize=12)
plt.title("Support Vector Machine Penalty Tuning", fontsize=15)
plt.savefig("Figures/SVM_penalty_tuning.png", dpi=300)
plt.show()

# feature selection using recursive feature elimination
penalty = best_penalties[0]
#penalty = 5
rep = 500

working_features = list(data.columns)
removed_features = []

acc_mean = np.zeros(len(data.columns))
acc_std = np.zeros(len(data.columns))

acc = np.zeros(rep)
for i in range(len(data.columns)):
    working_data = data[working_features]
    var_coef = np.zeros([rep, len(working_features)])
    for j in range(rep):
        # split data set
        test_size = 0.35
        train_size = 1-test_size
        X_train, X_test, Z_train, Z_test = train_test_split(working_data,
                                           np.ravel(death),
                                           train_size=train_size,
                                           test_size=test_size)
        # extract continuous features
        features_to_extract = [f for f in continuous_features if f in working_data]
        X_train_cont = X_train[features_to_extract]
        X_test_cont = X_test[features_to_extract]

        # scale continuous data
        scaler = StandardScaler()
        scaler.fit(X_train_cont)
        X_train_cont = scaler.transform(X_train_cont)
        X_test_cont = scaler.transform(X_test_cont)

        # fill scaled data
        with pd.option_context('mode.chained_assignment', None):
            for l,f in enumerate(features_to_extract):
                X_train.loc[:,f] = X_train_cont[:,l]
                X_test.loc[:,f] = X_test_cont[:,l]

        model = SVC(C=1/float(penalty), kernel="linear", gamma="scale")
        model.fit(X_train, Z_train)

        Z_pred = model.predict(X_test)

        acc[j] = ACC(Z_test, Z_pred)
        var_coef[j,:] = (np.ravel(model.coef_))**2
    acc_mean[i] = acc.mean()
    acc_std[i] = acc.std()

    avg_var_coef = var_coef.mean(axis=0)
    idx = np.argmin(avg_var_coef)

    worst_feature = working_features[idx]

    removed_features.append(worst_feature)
    working_features.remove(worst_feature)

print("--Order of feature removal--")
print(removed_features)

x = np.linspace(0, len(data.columns)-1, len(data.columns))
plt.errorbar(x, acc_mean, yerr=acc_std, fmt="o", capsize=5,
             markersize=7, color="darkgreen")
xmin, xmax = plt.xlim()
plt.hlines(acc_mean[0]+acc_std[0], xmin, xmax, linestyle="dashed",
           color="slategray")
plt.hlines(acc_mean[0]-acc_std[0], xmin, xmax, linestyle="dashed",
           color="slategray")
plt.ylim(0.5, 1)
plt.xlabel("Number of features removed", fontsize=12)
plt.ylabel("Accuracy", fontsize=12)
plt.title(f"Support Vector Machine Recursive Feature Elimination", fontsize=15)
plt.savefig("Figures/recursive_feature_elimination_SVM.png", dpi=300)
plt.show()

# visualize confusion matrix
# split data set
test_size = 0.35
train_size = 1-test_size
X_train, X_test, Z_train, Z_test = train_test_split(data, np.ravel(death),
                                   train_size=train_size, test_size=test_size)

# extract continuous features
features_to_extract = [f for f in continuous_features if f in data]
X_train_cont = X_train[features_to_extract]
X_test_cont = X_test[features_to_extract]

# scale continuous data
scaler = StandardScaler()
scaler.fit(X_train_cont)
X_train_cont = scaler.transform(X_train_cont)
X_test_cont = scaler.transform(X_test_cont)

# fill scaled data
with pd.option_context('mode.chained_assignment', None):
    for l,f in enumerate(features_to_extract):
        X_train.loc[:,f] = X_train_cont[:,l]
        X_test.loc[:,f] = X_test_cont[:,l]

model = SVC(C=1/float(best_penalties[0]), kernel="linear", gamma="scale")
model.fit(X_train, Z_train)

Z_pred = model.predict(X_test)

plot_confusion_matrix(Z_test, Z_pred, normalize=True, ndecimals=3,
                      title="Support Vector Machine Confusion Matrix",
                      savename="CM_SVM")

# compute final estimate of accuracy
N = 5
kfold = KFold(n_splits=N, shuffle=True)

accuracy_kfold = np.zeros(N)
model = SVC(C=1/float(best_penalties[0]), kernel="linear", gamma="scale")
for k, (train_index, test_index) in enumerate(kfold.split(data, death)):
    x_train = data.iloc[train_index]
    y_train = np.ravel(death.iloc[train_index])
    x_test = data.iloc[test_index]
    y_test = np.ravel(death.iloc[test_index])

    # extract continuous features
    features_to_extract = [f for f in continuous_features if f in data]
    x_train_cont = x_train[features_to_extract]
    x_test_cont = x_test[features_to_extract]

    # scale continuous data
    scaler = StandardScaler()
    scaler.fit(x_train_cont)
    x_train_cont = scaler.transform(x_train_cont)
    x_test_cont = scaler.transform(x_test_cont)

    # fill scaled data
    with pd.option_context('mode.chained_assignment', None):
        for l,f in enumerate(features_to_extract):
            x_train.loc[:,f] = x_train_cont[:,l]
            x_test.loc[:,f] = x_test_cont[:,l]

    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)

    accuracy_kfold[k] = ACC(y_test, y_pred)

print(f"Final accuracy average        = {accuracy_kfold.mean():.3f}")
print(f"Final accuracy std. deviation = {accuracy_kfold.std():.3f}")

"""
sample run:

Best penalties: [7. 1. 2.]
--Order of feature removal--
['platelets', 'diabetes', 'creatinine_phosphokinase', 'anaemia', 'serum_sodium',
'smoking', 'high_blood_pressure', 'sex', 'age', 'ejection_fraction',
'serum_creatinine']
Final accuracy average        = 0.752
Final accuracy std. deviation = 0.050
"""
