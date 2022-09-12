import pandas as pd
import matplotlib.pyplot as plt

import shap
from sklearn.metrics import accuracy_score, confusion_matrix

import pickle as rick

from methods import import_new_data, create_extensive_model, undummify

seed = 124653

_, new_big_data_set, _, _ = import_new_data()

p_big = new_big_data_set.pop("prog")


def train(d, y):
    model = create_extensive_model(d.shape[1])
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
    x_normed = (x - dat.mean()) / (dat.std())
    return mod.predict(x_normed.fillna(0))


exp = shap.Explainer(f, dat.sample(n=100))
shp_val = exp(dat.sample(n=12000))
plot3 = shap.plots.beeswarm(shp_val, max_display=40, plot_size=(8, 12), show=False)
plt.tight_layout()
plt.savefig("diagrams/shap_summary_12000.png")
plt.close()

f = open("data/shap_values_12000", "ab")
rick.dump(shp_val, f)
f.close()
