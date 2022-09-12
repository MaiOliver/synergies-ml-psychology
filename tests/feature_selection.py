import numpy as np
import pandas as pd
from tqdm import tqdm  # for progress bar in loop

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif

import matplotlib.pyplot as plt

seed = 21131

folder = "./data"

data = pd.read_csv(folder + "/cleaned/new_big_data_set.csv", delimiter=";", decimal=".")

data.loc[data.deltatimecor >= 0, "cat"] = 0
data.loc[data.deltatimecor < 0, "cat"] = 1
time = data.pop("deltatimecor")

labels = data.pop("cat")

normalized_df = (data - data.min()) / (data.max() - data.min())  # Pandas should apply these functions column-wise
# normalized_df = (data-data.mean())/(data.std())

train_set = normalized_df.sample(frac=0.7, random_state=seed)
train_labels = labels[train_set.index]

test_set = normalized_df.drop(train_set.index)
test_labels = labels[test_set.index]

fs = SelectKBest(score_func=f_classif, k="all")
fs.fit(train_set, train_labels)
train_set_fs = fs.transform(train_set)
test_set_fs = fs.transform(test_set)

plt.bar([i for i in range(len(fs.scores_))], fs.scores_)
plt.show()


def undummify(df, prefix_sep="_"):
    cols2collapse = {
        item.split(prefix_sep)[0]: (prefix_sep in item) for item in df.columns
    }
    series_list = []
    for col, needs_to_collapse in cols2collapse.items():
        if needs_to_collapse:
            undummified = (
                df.filter(like=col)
                .idxmax(axis=1)
                .apply(lambda x: x.split(prefix_sep, maxsplit=1)[1])
                .rename(col)
            )
            series_list.append(undummified)
        else:
            series_list.append(df[col])
    undummified_df = pd.concat(series_list, axis=1)
    return undummified_df
