import numpy as np
import pandas as pd

import json

import copy as cp

import matplotlib.pyplot as plt

from methods import import_new_data, create_extensive_model

seed = 124653

data, new_big_data_set, new_selected_data_set, minimal_data = import_new_data()

# Import swim set data and fit a NN to classify whether an athlete improves (1) or worsens (-1) between
# individual and relay performance.

# Read the prognosis of the theory model
theory_percent = len(new_big_data_set[(new_big_data_set.prog == new_big_data_set.cat)]) / len(new_big_data_set.prog)
# naive_percent = len(data[(data.doodoo == data.cat)]) / len(data.prog)

Y = new_big_data_set["cat"].astype("int64").values


p_big = new_big_data_set.pop("prog")
p_sel = new_selected_data_set.pop("prog")
data.drop(["valence", "prog", "expout", "YEAR", "MEETDATE", "deltatime", "MEETCITY",  "relay"], axis=1, inplace=True)
# model = create_baseline(36)


def train(d, y):
    # Method that takes a dataset d and a result vector y as input
    # and trains a neural network on it
    model = create_extensive_model(d.shape[1])
    # model = create_baseline(d.shape[1])
    normalized_df = (d-d.mean())/(d.std()) # (dat-dat.min())/(dat.max()-dat.min())
    h = model.fit(normalized_df, y, epochs=100, batch_size=12000, validation_split=0.3)
    return model, h


# Add one feature at a time and see if predicition improves
addone_accs = []
addone_feats = ["NATION", "age", "points", "individual", "RELAYTOTAL", "place_i", "place_r", "DATE_R", "gender", "style",
                "MEETNATION"]

for feature in addone_feats:
    df = cp.deepcopy(minimal_data)
    ydf = df.pop("cat").astype("int64").values

    df[feature] = data[feature]
    df = pd.get_dummies(df)

    _, hist = train(df, ydf)
    addone_accs.append(hist.history["val_accuracy"][-1])
    print(f"Accuracy after adding feature '{feature}': {addone_accs[-1]}")


# make baseline to compare against
df = cp.deepcopy(minimal_data)
ydf = df.pop("cat").astype("int64").values
_, hist = train(df, ydf)
base_acc = hist.history["val_accuracy"][-1]

# Rename features to be more verbose
addone_feats[addone_feats.index("place_i")] = "Placement in Solo competition"
addone_feats[addone_feats.index("age")] = "Age"
addone_feats[addone_feats.index("MEETNATION")] = "Country of Competition"
addone_feats[addone_feats.index("NATION")] = "Swimmer Nationality"
addone_feats[addone_feats.index("points")] = "FINA Points"


# Get improvement in percentage
diffs = [(feature, 100*(1 - base_acc/acc)) for feature, acc in zip(addone_feats, addone_accs)]

addone = {item[0]: item[1] for item in diffs}

with open("./data/categorical/addone_feature.json", "w") as file:
    json.dump(addone, file)
