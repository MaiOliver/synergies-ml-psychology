import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from matplotlib import rc
rc("text", usetex=True)

# location of data files; should in the future be a sub-folder of the current working directory
folder = "./data"

np.random.seed(1337)
hypothesis_one = pd.read_csv(folder + "/lr_rf_nn_hybrid_big_hypot_3.csv", delimiter=";", decimal=".")
# Sample size in future data
title = f"Size of training sample: 200000\n Number of repetitions: " \
        f"{int(hypothesis_one.shape[0]/5)}"
surrogate_y = np.random.rand(int(hypothesis_one.shape[0]/5.0))-0.5

R2_test_models = {}
RMSE_test_models = {}
procedures = hypothesis_one["Procedure"].unique()
for proc in procedures:
    R2_test_models[proc] = hypothesis_one.loc[hypothesis_one["Procedure"] == proc, "R2test"]
    RMSE_test_models[proc] = hypothesis_one.loc[hypothesis_one["Procedure"] == proc, "RMSEtest"]

alpha = 0.4
fig1, ax1 = plt.subplots(len(procedures), 1, sharex=True)
colors = [[[1., alpha, alpha]], [[alpha, 0.8, alpha]], [[alpha, alpha, 1.]],
          [[alpha, alpha, 0.0]], [[0.0, alpha, alpha]]]
for i, proc in enumerate(procedures):
    ax1[i].scatter(R2_test_models[proc], surrogate_y, c=colors[i])
    if i == 1:
        ax1[i].scatter(np.mean(R2_test_models[proc]), 0.0, marker="x", c="k", label="Mean")
    else:
        ax1[i].scatter(np.mean(R2_test_models[proc]), 0.0, marker="x", c="k")

    ax1[i].tick_params(axis="y", which="both", left=False, labelleft=False)
    ax1[i].set_ylabel(proc, rotation=0, labelpad=50)

    if i == len(procedures)-1:
        ax1[i].set_xlabel(r"$R^2$")
fig1.suptitle(title)
fig1.legend()
plt.subplots_adjust(left=0.24)
alpha = 0.4
fig2, ax2 = plt.subplots(len(procedures), 1, sharex=True)
colors = [[[1., alpha, alpha]], [[alpha, 0.8, alpha]], [[alpha, alpha, 1.]],
          [[alpha, alpha, 0.0]], [[0.0, alpha, alpha]]]
for i, proc in enumerate(procedures):
    ax2[i].scatter(RMSE_test_models[proc], surrogate_y, c=colors[i])
    if i == 1:
        ax2[i].scatter(np.mean(RMSE_test_models[proc]), 0.0, marker="x", c="k", label="Mean")
    else:
        ax2[i].scatter(np.mean(RMSE_test_models[proc]), 0.0, marker="x", c="k")

    ax2[i].tick_params(axis="y", which="both", left=False, labelleft=False)
    ax2[i].set_ylabel(proc, rotation=0, labelpad=50)

    if i == len(procedures)-1:
        ax2[i].set_xlabel("RMSE")

fig2.suptitle(title)
fig2.legend()
plt.subplots_adjust(left=0.24)
fig1.savefig("./diagrams/l_rf_nn_R2_new.png")
fig2.savefig("./diagrams/l_rf_nn_RMSE_new.png")

plt.show()
