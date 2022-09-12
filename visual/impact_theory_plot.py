import numpy as np
import matplotlib.pyplot as plt

acc_wo_theory = np.genfromtxt("./data/categorical/raw_nn_acc_vs_n_wo.csv", delimiter=";")
acc_w_theory = np.genfromtxt("./data/categorical/raw_nn_acc_vs_n_w.csv", delimiter=";")
Ns = np.genfromtxt("./data/categorical/raw_nn_acc_vs_n_ns.csv", delimiter=";")

means = {"w_theory": [], "wo_theory": []}
vars = {"w_theory": [], "wo_theory": []}

for i in range(acc_w_theory.shape[0]):
    means["w_theory"].append(np.mean(acc_w_theory[i]))
    means["wo_theory"].append(np.mean(acc_wo_theory[i]))
    vars["w_theory"].append(np.var(acc_w_theory[i]))
    vars["wo_theory"].append(np.var(acc_wo_theory[i]))

plt.figure(1)
plt.scatter(Ns, means["wo_theory"], label="All Features", marker="o")
plt.scatter(Ns, means["w_theory"], label="Preselected Features", marker="x")
plt.plot(Ns, means["w_theory"], c="grey", alpha=0.5)
plt.plot(Ns, means["wo_theory"], c="grey", alpha=0.5)

plt.xlabel("Sample Size")
plt.ylabel("Mean Accuracy")
plt.legend()
plt.title("Comparison of Machine Learning Model Performances")
plt.savefig("diagrams/categorical/categorical_nn_samples_mean.png")
plt.show()

plt.figure(2)
plt.scatter(Ns, vars["wo_theory"], label="All Features", marker="o")
plt.scatter(Ns, vars["w_theory"], label="Preselected Features", marker="x")
plt.plot(Ns, vars["wo_theory"], c="grey", alpha=0.5)
plt.plot(Ns, vars["w_theory"], c="grey", alpha=0.5)

plt.xlabel("Sample Size")
plt.ylabel("Variance of Accuracy")
plt.title("ANN Performance")
plt.legend()

plt.show()
