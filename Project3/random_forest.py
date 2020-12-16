import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import accuracy_score as ACC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold as KFold
from sklearn.ensemble import RandomForestClassifier as RFC
from confusion_matrix import plot_confusion_matrix

# read data
data = pd.read_csv("heart_failure_data.csv")

# prepare dataframe
death = data[["DEATH_EVENT"]]
data = data.drop("DEATH_EVENT", axis=1)
data = data.drop("time", axis=1)

# perform kfold on number of trees
N = 5
kfold = KFold(n_splits=N, shuffle=True)

n_trees = np.linspace(1, 200, 10)

accuracy_mean = np.zeros(len(n_trees))
accuracy_std = np.zeros(len(n_trees))

accuracy_kfold = np.zeros(N)
for i, n in enumerate(n_trees):
    model = RFC(n_estimators=int(n))
    for j, (train_index, test_index) in enumerate(kfold.split(data, death)):
        x_train = data.iloc[train_index]
        y_train = np.ravel(death.iloc[train_index])
        x_test = data.iloc[test_index]
        y_test = np.ravel(death.iloc[test_index])

        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)

        accuracy_kfold[j] = ACC(y_test, y_pred)
    accuracy_mean[i] = accuracy_kfold.mean()
    accuracy_std[i] = accuracy_kfold.std()

plt.errorbar(n_trees, accuracy_mean, yerr=accuracy_std, fmt="o", capsize=5,
             markersize=7, color="darkgreen")
plt.ylim(0.5, 1)
plt.xlabel("Number of trees", fontsize=12)
plt.ylabel("Accuracy", fontsize=12)
plt.title(f"{N}-fold Cross Validation", fontsize=15)
plt.savefig(f"Figures/{N}fold_CV_RFC_ntrees.png", dpi=300)
plt.show()

# grid search maximum depth and maximum number of features
max_features = np.linspace(1, len(data.columns), len(data.columns)).astype(int)
max_depth = np.linspace(1, 20, 10).astype(int)

N = 5
kfold = KFold(n_splits=N, shuffle=True)

acc_mean = np.zeros([len(max_features), len(max_depth)])
acc_std = np.zeros([len(max_features), len(max_depth)])

accuracy_kfold = np.zeros(N)
for i, mf in enumerate(max_features):
    for j, md in enumerate(max_depth):
        model = RFC(n_estimators=50, max_features=mf, max_depth=md)
        for k, (train_index, test_index) in enumerate(kfold.split(data, death)):
            x_train = data.iloc[train_index]
            y_train = np.ravel(death.iloc[train_index])
            x_test = data.iloc[test_index]
            y_test = np.ravel(death.iloc[test_index])

            model.fit(x_train, y_train)
            y_pred = model.predict(x_test)

            accuracy_kfold[k] = ACC(y_test, y_pred)
        acc_mean[i,j] = accuracy_kfold.mean()
        acc_std[i,j] = accuracy_kfold.std()

ind = np.unravel_index(np.argmax(acc_mean, axis=None), acc_mean.shape)
mf_best = max_features[ind[0]]
md_best = max_depth[ind[1]]
print("--Best Hyperparameters--")
print("Max features:", mf_best)
print("Max depth:", md_best)

N, M = max_features, max_depth

N_lims = np.array([np.min(N), np.max(N)])
M_lims = np.log10(np.array([np.min(M), np.max(M)]))
dx, = np.diff(M_lims)/(len(M)-1)
dy, = -np.diff(N_lims)/(len(N)-1)
extent = [M_lims[0]-dx/2, M_lims[1]+dx/2, N_lims[0]+dy/2, N_lims[1]-dy/2]
xlabels = [f"{m:d}" for m in M]
ylabels = [f"{n:d}" for n in N]

plt.imshow(acc_mean, cmap="Greens", extent=extent, aspect="auto")
plt.xticks(np.arange(M_lims[0], M_lims[1]+dx, dx), labels=xlabels)
plt.yticks(np.arange(N_lims[1], N_lims[0]+dy, dy), labels=ylabels)
plt.colorbar()
plt.ylabel("Maximum number of features", fontsize=12)
plt.xlabel("Maximum depth of trees", fontsize=12)
plt.title("Random Forest Hyperparameter Tuning", fontsize=15)
plt.savefig("Figures/hyperparameter_tuning_RFC.png", dpi=300)
plt.show()

# feature selection based on feature imporances
model = RFC(n_estimators=50, max_depth=md_best, max_features=mf_best)
model.fit(data, np.ravel(death))

print("--Features and importances--")
for i, imp in enumerate(model.feature_importances_):
    print(f"{data.columns[i]}: {imp}")

names = {"creatinine_phosphokinase":"CPK",
         "ejection_fraction":"ejection\nfraction",
         "high_blood_pressure":"high blood\npressure",
         "serum_creatinine":"serum\ncreatinine",
         "serum_sodium":"serum\nsodium"}
name_map = {col:col for col in data.columns}
name_map.update(names)
data.columns = [name_map[col] for col in data.columns]

order = np.argsort(model.feature_importances_)
x = np.linspace(0, len(data.columns), len(data.columns))

plt.barh(x, model.feature_importances_[order], color="red")
plt.yticks(x, data.columns[order], fontsize=12)
plt.xlabel("Feature importance", fontsize=12)
plt.title("Random Forest Feature Importances", fontsize=15)
plt.gcf().subplots_adjust(left=0.16, right=0.98, top=0.94)
plt.savefig("Figures/feature_importances_RFC.png", dpi=300)
plt.show()

# feature selection using recursive feature elimination
rep = 100

working_features = list(data.columns)
removed_features = []

acc_mean = np.zeros(len(data.columns))
acc_std = np.zeros(len(data.columns))

acc = np.zeros(rep)
for i in range(len(data.columns)):
    working_data = data[working_features]
    var_imp = np.zeros([rep, len(working_features)])
    mf_best = min(mf_best, len(working_features))
    for j in range(rep):
        # split data set
        test_size = 0.35
        train_size = 1-test_size
        X_train, X_test, Z_train, Z_test = train_test_split(working_data,
                                           np.ravel(death),
                                           train_size=train_size,
                                           test_size=test_size)

        model = RFC(n_estimators=50, max_depth=md_best, max_features=mf_best)
        model.fit(X_train, Z_train)

        Z_pred = model.predict(X_test)

        acc[j] = ACC(Z_test, Z_pred)
        var_imp[j,:] = model.feature_importances_
    acc_mean[i] = acc.mean()
    acc_std[i] = acc.std()

    avg_var_imp = var_imp.mean(axis=0)
    idx = np.argmin(avg_var_imp)

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
plt.title(f"Random Forest Recursive Feature Elimination", fontsize=15)
plt.savefig("Figures/recursive_feature_elimination_RFC.png", dpi=300)
plt.show()

# visualize confusion matrix
# split data set
test_size = 0.35
train_size = 1-test_size
X_train, X_test, Z_train, Z_test = train_test_split(data, np.ravel(death),
                                   train_size=train_size, test_size=test_size)

model = RFC(n_estimators=50, max_depth=md_best, max_features=mf_best)
model.fit(X_train, Z_train)
Z_pred = model.predict(X_test)

plot_confusion_matrix(Z_test, Z_pred, normalize=True,
                      title="Random Forest Confusion Matrix", savename="CM_RF")

"""
sample run:

--Best Hyperparameters--
Max features: 7
Max depth: 15
--Features and importances--
age: 0.15469979033027928
anaemia: 0.013514135484760366
creatinine_phosphokinase: 0.12853819088418603
diabetes: 0.013321978925365624
ejection_fraction: 0.19096076986315605
high_blood_pressure: 0.014775865117591658
platelets: 0.11919699402325026
serum_creatinine: 0.23100971173325577
serum_sodium: 0.09767476475414737
sex: 0.023266523998380993
smoking: 0.013041274885626602
--Order of feature removal--
['smoking', 'diabetes', 'anaemia', 'sex', 'high blood\npressure', 'serum\nsodium', 'CPK', 'age', 'ejection\nfraction', 'serum\ncreatinine', 'platelets']
"""
