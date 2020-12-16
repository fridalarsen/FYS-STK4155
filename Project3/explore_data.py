import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

data = pd.read_csv("heart_failure_data.csv")
#print(data.head())
#print(data.info())
death = data[["DEATH_EVENT"]]
data = data.drop("DEATH_EVENT", axis=1)
data = data.drop("time", axis=1)

names = {"creatinine_phosphokinase":"CPK",
         "ejection_fraction":"ejection\nfraction",
         "high_blood_pressure":"high blood\npressure",
         "serum_creatinine":"serum\ncreatinine",
         "serum_sodium":"serum\nsodium"}
name_map = {col:col for col in data.columns}
name_map.update(names)
data.columns = [name_map[col] for col in data.columns]

x = np.linspace(0, len(data.columns)-1, len(data.columns))
plt.figure()
plt.matshow(data.corr(), cmap="YlOrRd", fignum=0, aspect="auto")
plt.colorbar()
plt.yticks(x, data.columns, fontsize=12)
plt.xticks(x, data.columns, fontsize=12, rotation=90)
plt.ylim(len(x)-0.5, -0.5)
plt.xlim(-0.5, len(x)-0.5)
plt.gca().xaxis.set_ticks_position('bottom')
plt.title("Correlation Matrix", fontsize=15)
plt.gcf().subplots_adjust(left=0.16, right=1.05, top=0.90, bottom=0.21)
plt.savefig("Figures/correlation_matrix.png", dpi=300)
plt.show()
