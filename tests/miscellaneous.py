import numpy as np
import pandas as pd
from tqdm import tqdm  # for progress bar in loop
import time
import datetime as dt

import copy as cp

import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# import h2o4gpu as hg

from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.pipeline import Pipeline

seed = 124653

# This file essentially only is for testing purposes before integrating this functionality into
# "train_neural_network.py" and subsequently into "Datenauswertung".

# Import swim set data and fit a NN to classify whether an athlete improves (1), worsens (-1) or roughly stays
# the same between individual and relay performance.

# Import and classify data (classification should maybe be done in "Datenaufbereitung.py")
folder = "./data"

data = pd.read_csv(folder + "/cleaned/full_data_set.csv", delimiter=";", decimal=".")

# convert dates to seconds since epoch (1970-1-1) should be moved to "Datenaufbereitung" in the future
date_columns = ["BIRTHDATE", "DATE_I", "DATE_R", "MEETDATE_A", "MEETDATE_O"]
for column in date_columns:
    data[column] = pd.to_datetime(data[column])
    data[column] = (data[column] - dt.datetime(1970, 1, 1)).dt.total_seconds()

category_columns = ["gender", "NATION", "style", "round_i", "round_r", "MEETCITY", "MEETNATION"]
for column in category_columns:
    data[column] = data[column].astype("category")


new_selected_data_set = pd.read_csv(folder + "/cleaned/new_selected_data_set.csv", delimiter=";", decimal=".")
new_big_data_set = pd.read_csv(folder + "/cleaned/new_big_data_set.csv", delimiter=";", decimal=".")

minimal_data = pd.DataFrame()
for column in ["expout", "valence", "storder", "deltatimecor"]:
    minimal_data[column] = data[column]
minimal_data = pd.get_dummies(minimal_data)

"""
delta = 0.001  # Symmetric interval around zero to be classified as "no change in performance"

class_names = ["even", "worse", "better"]

data.loc[np.abs(data.deltatimecor) <= delta, "cat"] = 0
data.loc[data.deltatimecor > delta, "cat"] = 1
data.loc[data.deltatimecor < -delta, "cat"] = 2
"""


def make_cat(df):
    df.loc[df.deltatimecor >= 0, "cat"] = 0
    df.loc[df.deltatimecor < 0, "cat"] = 1
    time = df.pop("deltatimecor")


for df in [data, new_selected_data_set, new_big_data_set, minimal_data]:
    make_cat(df)

S = set(new_big_data_set.columns) - set(["cat"])

# data["doodoo"] = 0

theory_percent = len(new_big_data_set[(new_big_data_set.prog == new_big_data_set.cat)]) / len(new_big_data_set.prog)
# doodoo_percent = len(data[(data.doodoo == data.cat)]) / len(data.prog)

"""
theory_samples = []
for i in range(100):
    sample = new_big_data_set.sample(100, random_state=seed + i)
    sample_percent = accuracy_score(sample.cat, sample.prog)
    theory_samples.append(sample_percent)
theory_var = np.var(theory_samples)
"""

Y = new_big_data_set["cat"].astype("int64").values


def create_baseline(length):
    model = keras.Sequential()
    model.add(layers.Dense(length, input_dim=length, activation="elu"))
    model.add(layers.Dense(5, activation="elu"))
    # model.add(layers.Dense(2, activation="elu"))
    model.add(layers.Dense(1, activation="sigmoid"))

    model.compile(loss="mean_squared_error", optimizer="adam", metrics=["accuracy"])
    return model


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
p_sel = new_selected_data_set.pop("prog")
data.drop(["valence", "prog", "expout", "YEAR", "MEETDATE", "deltatime", "MEETCITY",  "relay"], axis=1, inplace=True)
# model = create_baseline(36)


def train(d, y):
    model = create_extensive(d.shape[1])
    # model = create_baseline(d.shape[1])
    normalized_df = (d-d.mean())/(d.std()) # (dat-dat.min())/(dat.max()-dat.min())
    h = model.fit(normalized_df, y, epochs=100, batch_size=12000, validation_split=0.3)
    return model, h



addone_accs = []
addone_feats = ["NATION", "age", "points", "individual", "RELAYTOTAL","place_i", "place_r", "DATE_R", "gender", "style",
                "MEETNATION"]

for feature in addone_feats:
    df = cp.deepcopy(minimal_data)
    ydf = df.pop("cat").astype("int64").values

    df[feature] = data[feature]
    df = pd.get_dummies(df)

    _, hist = train(df, ydf)
    addone_accs.append(hist.history["val_accuracy"][-1])


# make baseline
df = cp.deepcopy(minimal_data)
ydf = df.pop("cat").astype("int64").values
_, hist = train(df, ydf)
base_acc = hist.history["val_accuracy"][-1]

addone_feats[addone_feats.index("place_i")] = "Individual Placement"
addone_feats[addone_feats.index("age")] = "Age"
addone_feats[addone_feats.index("MEETNATION")] = "Country of Competition"
addone_feats[addone_feats.index("NATION")] = "Swimmer Nationality"
addone_feats[addone_feats.index("points")] = "FINA Points"


diffs = [(feature, 100*(1 - base_acc/acc)) for feature, acc in zip(addone_feats, addone_accs)]
diffs.sort(key=lambda a: a[1])

fig, ax = plt.subplots(figsize=(6, 3))

# Example data
j = 4
y_pos = np.arange(j)
cmap = plt.get_cmap("Purples")
cs = cmap(np.linspace(0.25, 1, j))
ax.barh(y_pos, [item[1] for item in diffs[-j:]], align='center', color=cs)
ax.set_yticks(y_pos, labels=[item[0] for item in diffs[-j:]])
# ax.invert_yaxis()  # labels read top-to-bottom
ax.set_xlabel('Accuracy increase in percent')
plt.tight_layout()
plt.savefig("diagrams/categorical/addone_feature_re.png")
plt.show()



data = pd.get_dummies(data)

times = []
accs = []
sets = [(dat, dat.pop("cat").astype("int64").values) for dat in [data, new_big_data_set, minimal_data]]
models = []
conf_mats = []

for dat, Y1 in sets:
    t1 = time.time()
    mod, hist = train(dat, Y1)
    t2 = time.time()
    times.append(t2 - t1)
    accs.append(hist.history["val_accuracy"][-1])
    models.append(mod)

    dat_normed = (dat - dat.mean()) / (dat.std())
    y_pred = mod.predict(dat_normed.fillna(0))
    y_pred = [0 if item < 0.50 else 1 for item in y_pred.T[0]]
    y_true = Y1
    print(f"Accuracy after training: {accuracy_score(y_true, y_pred)}")
    conf_mat = confusion_matrix(y_true, y_pred)
    conf_mats.append(conf_mat)



#big_pred = models[0].predict(sets[0][0])
#min_pred = models[1].predict(sets[1][0])
#big_pred = 1*(big_pred.flatten() >= 0.5)
#min_pred = 1*(min_pred.flatten() >= 0.5)

#correct_positive_big = (np.sum((big_pred == 1) & (sets[0][1] == 1)))/np.sum(big_pred)
#correct_positive_min = (np.sum((min_pred == 1) & (sets[1][1] == 1)))/np.sum(min_pred)
#correct_negative_big = (np.sum((big_pred == 0) & (sets[0][1] == 0)))/(np.sum(big_pred == 0))
#correct_negative_min = (np.sum((min_pred == 0) & (sets[1][1] == 0)))/(np.sum(min_pred == 0))


# model.save("./data/decision_network")

"""
nn_samples = []
for i in range(100):
    sample = new_big_data_set.sample(100, random_state=seed + i)
    Y_true = Y[sample.index]
    Y_pred = model.predict(sample)
    Y_pred = 1 * (Y_pred >= 0.5)
    sample_percent = accuracy_score(Y_true, Y_pred)
    nn_samples.append(sample_percent)
nn_var = np.var(nn_samples)
"""
"""
estimator = KerasClassifier(build_fn=create_baseline, epochs=30, batch_size=10000, verbose=0)
kfold = StratifiedKFold(n_splits=7, shuffle=True)
results = cross_val_score(estimator, data.values, Y, cv=kfold)
print(f"Baseline: {results.mean()*100:.2f} % ({results.std()*100:.2f} %)")
"""

kfold = False
if kfold:
    estimators = []
    estimators.append(("standardize", StandardScaler()))
    estimators.append(("mlp", KerasClassifier(build_fn=create_baseline, epochs=30, batch_size=10000, verbose=0)))
    pipeline = Pipeline(estimators)
    kfold = StratifiedKFold(n_splits=10, shuffle=True)
    results = cross_val_score(pipeline, new_big_data_set, Y, cv=kfold)
    print(f"Standardized: {results.mean()*100:.2f} % ({results.std()*100:.2f} %)")

    rfc = hg.RandomForestClassifier(n_estimators=500, max_features="auto", random_state=seed, oob_score=True, verbose=2, n_gpus=1)
    scaler = StandardScaler()
    scaler.fit(new_big_data_set)
    normed = scaler.transform(new_big_data_set)
    rfc.fit(normed, Y)

    estimators = []
    estimators.append(("standardize", StandardScaler()))
    estimators.append(("rfc", rfc))
    pipeline = Pipeline(estimators)
    kfold = StratifiedKFold(n_splits=2, shuffle=True)
    results_rfc = cross_val_score(pipeline, new_big_data_set, Y, cv=kfold)
    print(f"Standardized: {results.mean()*100:.2f} % ({results.std()*100:.2f} %)")


def train_test(N, wo_theory=True, repeats=100, batch_size=None, epochs=100, lr=0.001):
    acc = []
    pbar = tqdm(range(repeats))
    for i in pbar:
        if wo_theory:
            data = new_big_data_set
        else:
            data = new_selected_data_set

        dat = data.sample(N, random_state=seed+i).reset_index()

        labels = dat.pop("cat")
        dat = dat.drop(["index"], axis=1)

        #if wo_theory:
            # prog = data.pop("prog")
            # styles: data.columns[15:22]
            # storder: data.columns[22:26]
            # rounds: data.columns[26:36]
            # @_X: data.columns[3:9]
            #dat = dat.drop(list(S-set(["points", "age", "place_i", "place_r", "prog"])), axis=1)
            #print(dat.columns)

        # normalized_df = (dat-dat.min())/(dat.max()-dat.min())  # Pandas should apply these functions column-wise
        normalized_df = (dat-dat.mean())/(dat.std())

        x_train = normalized_df # .sample(frac=0.7, random_state=seed)
        # x_train, x_test, y_train, y_test = train_test_split(df, labels, test_size=0.25, random_state=seed+i)
        y_train = labels


        """
        # Most basic model first
        model = tf.keras.Sequential()
        model.add(layers.Dense(train_set.shape[1], input_shape=[train_set.shape[1]], activation="relu"))
        # model.add(layers.Dense(256, activation="relu"))
        # model.add(layers.Dense(256, activation="relu"))
        model.add(layers.Dense(1, activation="sigmoid"))


        optimizer = tf.keras.optimizers.Adam(learning_rate=lr, beta_1=0.75, beta_2=0.9, amsgrad=True)
        # optimizer = tf.keras.optimizers.SGD(lr, momentum=0.8, decay=lr/0.8)
        # SparseCategoricalCrossentropy or BinaryCrossentropy
        model.compile(optimizer=optimizer, loss=tf.losses.BinaryCrossentropy(from_logits=False),
                      metrics=["accuracy"])
        """
        model = create_baseline(length=x_train.shape[1])

        if batch_size is None:
            batch_size = int(len(x_train))

            if batch_size == 0:
                batch_size += 1
            elif 32 < batch_size <= 64:
                batch_size = 32
            elif 64 < batch_size <= 128:
                batch_size = 64
            elif batch_size > 128:
                batch_size = 128

        hist = model.fit(x_train, y_train, epochs=epochs, verbose=0, batch_size=batch_size, validation_split=0.3)

        test_acc = hist.history['val_accuracy'][-1]
        # test_loss, test_acc = model.evaluate(test_set, test_labels, verbose=0)
        pbar.set_description(f"N: {N}, Accuracy: {test_acc:.2f}")
        acc.append(test_acc)

    if repeats == 1:
        return model# , acc

    return acc

"""
# model1 = train_test(len(data), wo_theory=False, repeats=1, epochs=30, lr=0.01)
# model2 = train_test(len(data), wo_theory=True, repeats=1, epochs=30, lr=0.01)

lab = data.pop("cat")
this = (data - data.min()) / (data.max() - data.min())
y_pred = model1.predict(this)
pred = 1*(y_pred >= 0.5)

"""
Ns = np.arange(10, 500, 10)
reps = 20
acc_wo_theory = []
acc_w_theory = []
for n in Ns:
    acc_wo_theory.append(train_test(n, repeats=reps))
    acc_w_theory.append(train_test(n, wo_theory=False, repeats=reps))

np.savetxt("./data/categorical/raw_nn_acc_vs_n_wo.csv", acc_wo_theory, delimiter=";")
np.savetxt("./data/categorical/raw_nn_acc_vs_n_w.csv", acc_w_theory, delimiter=";")

# acc_wo_theory = pd.read_csv("./data/categorical/raw_nn_acc_vs_n_wo.csv", delimiter=";", decimal=".", header=None)
# acc_w_theory = pd.read_csv("./data/categorical/raw_nn_acc_vs_n_w.csv", delimiter=";", decimal=".", header=None)

acc_wo_theory = np.genfromtxt("./data/categorical/raw_nn_acc_vs_n_wo.csv", delimiter=";")
acc_w_theory = np.genfromtxt("./data/categorical/raw_nn_acc_vs_n_w.csv", delimiter=";")

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
plt.plot(Ns, vars["w_theory"], label="Selected")
plt.plot(Ns, vars["wo_theory"], label="All")
plt.xlabel("Sample Size")
plt.ylabel("Variance of Accuracy")
plt.title("ANN Performance")
plt.legend()

plt.show()