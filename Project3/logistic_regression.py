import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import accuracy_score as ACC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold as KFold
from sklearn.linear_model import LogisticRegression as LR
from confusion_matrix import plot_confusion_matrix

# read data
data = pd.read_csv("heart_failure_data.csv")

# prepare dataframe
death = data[["DEATH_EVENT"]]
data = data.drop("DEATH_EVENT", axis=1)
data = data.drop("time", axis=1)

# kfold to determine optimal penalty
penalties = np.logspace(-4, 3, 15)

N = 5
kfold = KFold(n_splits=N, shuffle=True)

acc_mean = np.zeros(len(penalties))
acc_std = np.zeros(len(penalties))

accuracy_kfold = np.zeros(N)
acc = np.zeros([len(penalties),N])
for i, penalty in enumerate(penalties):
    model = LR(penalty="l2", C=1/float(penalty), fit_intercept=True,
               solver="liblinear")
    for k, (train_index, test_index) in enumerate(kfold.split(data, death)):
        x_train = data.iloc[train_index]
        y_train = np.ravel(death.iloc[train_index])
        x_test = data.iloc[test_index]
        y_test = np.ravel(death.iloc[test_index])

        # scale data
        scaler = StandardScaler()
        scaler.fit(x_train)
        x_train = scaler.transform(x_train)
        x_test = scaler.transform(x_test)

        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)

        accuracy_kfold[k] = ACC(y_test, y_pred)
        acc[i, k] = accuracy_kfold[k]


    acc_mean[i] = accuracy_kfold.mean()
    acc_std[i] = accuracy_kfold.std()

best_penalty = penalties[np.argmax(acc_mean)]
print("Best penalty:", best_penalty)

plt.errorbar(penalties, acc_mean, yerr=acc_std, fmt="o", capsize=5,
             markersize=7, color="darkgreen")
plt.ylim(0.5, 1)
plt.xscale("log")
plt.xlabel("Penalty", fontsize=12)
plt.ylabel("Accuracy", fontsize=12)
plt.title(f"{N}fold Cross Validation", fontsize=15)
plt.savefig(f"Figures/{N}fold_CV_LR_penalties.png", dpi=300)
plt.show()

# feature selection using recursive feature elimination
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
        # scale data
        scaler = StandardScaler()
        scaler.fit(X_train)
        X_train = scaler.transform(X_train)
        X_test = scaler.transform(X_test)

        model = LR(penalty="l2", C=1/float(best_penalty), fit_intercept=True,
                   solver="liblinear")
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
plt.title(f"Logistic Regression Recursive Feature Elimination", fontsize=15)
plt.savefig("Figures/recursive_feature_elimination_LR.png", dpi=300)
plt.show()

# visualize confusion matrix
# split data set
test_size = 0.35
train_size = 1-test_size
X_train, X_test, Z_train, Z_test = train_test_split(data, np.ravel(death),
                                   train_size=train_size, test_size=test_size)

model = LR(penalty="l2", C=1/float(best_penalty), fit_intercept=True,
           solver="liblinear")
model.fit(X_train, Z_train)

Z_pred = model.predict(X_test)

plot_confusion_matrix(Z_test, Z_pred, normalize=True,
                      title="Logistic Regression Confusion Matrix",
                      savename="CM_LR")
"""
sample run:

Best penalty: 0.1
--Order of feature removal--
['diabetes', 'platelets', 'smoking', 'sex', 'anaemia', 'creatinine_phosphokinase', 'high_blood_pressure', 'serum_sodium', 'age', 'ejection_fraction', 'serum_creatinine']
"""
