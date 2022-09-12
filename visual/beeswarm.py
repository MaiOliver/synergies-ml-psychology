import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

import pickle as rick

import shap

f = open("data/shap_values_12000", "rb")
shap_values = rick.load(f)
f.close()

feature_names = shap_values.feature_names
feature_names[0] = "Individual Placement"
feature_names[1] = "Relay Placement"
feature_names[2] = "FINA Points"
feature_names[3] = "Age"
feature_names[4] = "Schedule"
feature_names[21] = "Started 1st"
feature_names[22] = "Started 2nd"
feature_names[23] = "Started 3rd"
feature_names[24] = "Started 4th"
feature_names[12] = "Female"
feature_names[13] = "Male"
feature_names[17] = "200 m Freestyle"
feature_names[16] = "100 m Freestyle"
feature_names[20] = "50 m Freestyle"
feature_names[32] = "Relay Finale"
feature_names[33] = "Relay Preliminary"

order = [0, 1, 2, 21, 22, 23, 24, 3, 12, 13, 17, 16, 20, 32, 33]
n_selected = len(order)
for rest in range(len(feature_names)):
    if rest not in order:
        order.append(rest)

shap.plots.beeswarm(shap_values, max_display=n_selected, plot_size=(12, 6), show=False,
                    order=order, color_bar=False)
plt.tight_layout()
plt.rcParams.update({"font.size": 14})
cbar = plt.colorbar(label="Feature Value", ticks=[0, 54])
cbar.ax.set_yticklabels(["Low", "High"])
plt.xlim([-0.3, 0.3])
# plt.show()

plt.savefig("diagrams/shap_beeswarm_1.png")
plt.close()
