import pandas as pd
import scipy.stats
import numpy as np
import matplotlib.pyplot as plt
from tabulate import tabulate

folder = "./data"
# read in csv files prepared in script "Datenaufbereitung.py" as pandas dataframe
# for old data set
selected_data_set = pd.read_csv(folder + "/cleaned/selected_data_set.csv", delimiter=";", decimal=".")
# for new data set
new_selected_data_set = pd.read_csv(folder + "/cleaned/new_selected_data_set.csv", delimiter=";", decimal=".")

comb = [("high", "high"), ("high", "low"), ("low", "high"), ("low", "low")]
titles = ["Valence: " + item[0] + " and Expout: " + item[1] for item in comb]
folders = ["diagrams/histograms/old_data/", "diagrams/histograms/new_data/"]
old = []
new = []
for j, data in enumerate([selected_data_set, new_selected_data_set]):
    cases = [(data["valence_" + item[0]] == 1) & (data["expout_" + item[1]] == 1) for item in comb]
    for i, case in enumerate(cases):
        data[case].hist(column="deltatimecor", bins=100)
        mean = np.mean(data[case].deltatimecor)
        var = np.var(data[case].deltatimecor)
        plt.title(titles[i] + f"\nMean: {mean:.3f} Variance: {var:.3f}")
        plt.ylabel("Count")
        plt.xlabel("Deltatime Corrected")
        plt.savefig(folders[j] + "valence_"+comb[i][0] + "_expout_" + comb[i][1] + ".png")

        if j == 1:
            old.append(data[case])
        else:
            new.append(data[case])


# Reproduce table 3 of Hertel et al. (2017) [could and should be done in the above loop]

def mean_confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
    return m, m-h, m+h


for j, data in enumerate([selected_data_set, new_selected_data_set]):
    list = [[], [], [], []]  # orderes as in comb
    position = [f"storder_{k+1}" for k in range(4)]

    for pos in position:
        cases = [(data["valence_" + item[0]] == 1) & (data["expout_" + item[1]] == 1) & (data[pos] == 1)
                 for item in comb]
        for i, case in enumerate(cases):
            # mean = np.mean(data[case].deltatimecor)
            m, ml, mh = mean_confidence_interval(data[case].deltatimecor)
            list[i].append(f"{m:.3f}[{ml:.3f}, {mh:.3f}]")

    # This feels naive
    filler_exp = ["High"]
    filler_exp.extend(["" for k in range(len(position))])  # + 1 for spacer
    filler_exp.append("Low")
    filler_exp.extend(["" for k in range(len(position) - 1)])
    filler_pos = [f"{(k % len(position)) + 1}" for k in range(2*len(position))]
    filler_pos.insert(len(position), "")  # spacer
    dic = {"Expected team outcome": filler_exp, "Serial position": filler_pos,
            "Valence High": list[0] + [""] + list[1], "Valence Low": list[2] + [""] + list[3]}

    if j == 1:
        pre = "new_"
    else:
        pre = ""

    with open("./data/" + pre + "table_hertel.txt", "w") as file:
        file.write(tabulate(dic, headers="keys"))

    print(tabulate(dic, headers="keys"))
