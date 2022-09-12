import numpy as np
import pandas as pd
import time
import datetime as dt

import matplotlib.pyplot as plt

import copy as cp

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

import shap

import pickle as rick

from sklearn.metrics import accuracy_score, confusion_matrix

seed = 124653


def undummify(df, prefix_sep="_"):
    cols2collapse = {
        item.split(prefix_sep)[0]: (prefix_sep in item) for item in df.columns
    }
    series_list = [df["place_i"], df["place_r"]]
    for col, needs_to_collapse in cols2collapse.items():
        if col == "place":
            continue
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


# Import and classify data (classification should maybe be done in "Datenaufbereitung.py")
folder = "./data"

new_big_data_set = pd.read_csv(folder + "/cleaned/new_big_data_set.csv", delimiter=";", decimal=".")


def make_cat(df):
    df.loc[df.deltatimecor >= 0, "cat"] = 0
    df.loc[df.deltatimecor < 0, "cat"] = 1
    time = df.pop("deltatimecor")


for df in [new_big_data_set]:
    make_cat(df)


def create_extensive(length):
    model = keras.Sequential()
    model.add(layers.Dense(length, input_dim=length, activation="elu"))
    model.add(layers.Dense(50, activation="elu"))  # , activity_regularizer=keras.regularizers.l1(0.01))) 1000
    model.add(layers.Dense(10, activation="elu"))  # 500
    model.add(layers.Dense(5, activation="tanh"))
    model.add(layers.Dense(9, activation="elu"))
    model.add(layers.Dense(1, activation="sigmoid"))
    # opt = keras.optimizers.Adam(learning_rate=0.0001, amsgrad=True)
    model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
    return model


p_big = new_big_data_set.pop("prog")


# model = create_baseline(36)


def train(d, y):
    model = create_extensive(d.shape[1])
    normalized_df = (d - d.mean()) / (d.std())  # (dat-dat.min())/(dat.max()-dat.min())
    h = model.fit(normalized_df, y, epochs=150, batch_size=1028, validation_split=0.3)
    return model, h


for column in new_big_data_set.columns:
    if column[:6] == "round_":
        rest = column[6:]
        new_big_data_set = new_big_data_set.rename(columns={column: "round" + rest})

dats = undummify(new_big_data_set)

category_columns = ["@", "gender", "style", "roundi", "roundr"]
for column in category_columns:
    dats[column] = dats[column].astype("category")

dats["storder"] = dats["storder"].astype("category")

dat = pd.get_dummies(dats)
Y1 = dat.pop("cat")

mod, hist = train(dat, Y1)
acc = hist.history["val_accuracy"][-1]
dat_normed = (dat - dat.mean()) / (dat.std())
y_pred = mod.predict(dat_normed.fillna(0))
y_pred = [0 if item < 0.50 else 1 for item in y_pred.T[0]]
y_true = Y1.values
print(f"Accuracy after training: {accuracy_score(y_true, y_pred)}")
conf_mat = confusion_matrix(y_true, y_pred)


def f(x):
    # x_loc = pd.get_dummies(x)
    # x_norm = (x_loc - x_loc.mean()) / (x_loc.std())  # (dat-dat.min())/(dat.max()-dat.min())
    # return mod.predict(x_norm.fillna(0))
    x_normed = (x - dat.mean()) / (dat.std())
    return mod.predict(x_normed.fillna(0))

"""
explainer = shap.KernelExplainer(f, dats.sample(n=100))
shap_values50 = explainer.shap_values(dats.sample(n=50))
shap_values = np.array(shap_values50)


plot = shap.force_plot(explainer.expected_value, shap_values[0], show=False)
shap.save_html("diagrams/index.html", plot)

plot1 = shap.summary_plot(shap_values[0], feature_names=dat.columns, max_display=12, plot_size=(14, 8), show=False)
plt.savefig("diagrams/shap_summary_1000.png")

plot2 = shap.force_plot(explainer.expected_value[0], shap_values[0][0, :], matplotlib=True,
                        feature_names=dat.columns)

"""

exp = shap.Explainer(f, dat.sample(n=100))
shp_val = exp(dat.sample(n=12000))
plot3 = shap.plots.beeswarm(shp_val, max_display=40, plot_size=(8, 12), show=False)
plt.tight_layout()
plt.savefig("diagrams/shap_summary_12000.png")
plt.close()

f = open("data/shap_values_12000", "ab")
rick.dump(shp_val, f)
f.close()

# f = open("data/shap_values_3500", "rb")
# shapley_values =  rick.load(f)
# f.close()
