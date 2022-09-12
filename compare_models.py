import numpy as np
import pandas as pd

from sklearn import linear_model, tree
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import LinearSVR
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix

import matplotlib.pyplot as plt
import seaborn as sns

from methods import undummify, import_new_data, create_extensive_model

_, new_big_data_set, _, _ = import_new_data()

seed = 124653

p_big = new_big_data_set.pop("prog")

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

dat_normed = (dat - dat.mean()) / (dat.std())
y_true = Y1.values

X_train, X_test, y_train, y_test = train_test_split(dat_normed.fillna(0), y_true, test_size=0.33, random_state=seed)


def evaluate_model(model):
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    pred = [0 if item < 0.50 else 1 for item in pred]
    return accuracy_score(y_test, pred), confusion_matrix(y_test, pred)


# Linear Regression
model_lr = linear_model.LogisticRegression()
acc_lr, conf_lr = evaluate_model(model_lr)

# Random Forest
model_rfr = RandomForestRegressor(n_estimators=250, max_features="auto",
                                  random_state=seed, oob_score=True, n_gpus=1)
acc_rfr, conf_rfr = evaluate_model(model_rfr)

# Suppport Vector Machine
model_svr = LinearSVR(random_state=seed, tol=1e-5, max_iter=10000)
acc_svr, conf_svr = evaluate_model(model_svr)

# decision tree
model_dt = tree.DecisionTreeClassifier()
acc_dt, conf_dt = evaluate_model(model_dt)

# Artificial neural network
model_ann = create_extensive_model(X_train.shape[1])
hist = model_ann.fit(X_train, y_train, epochs=150, batch_size=1028, validation_split=0.3)
acc = hist.history["val_accuracy"][-1]
y_pred = model_ann.predict(X_test)
y_pred = [0 if item < 0.50 else 1 for item in y_pred.T[0]]

acc_ann = accuracy_score(y_test, y_pred)
conf_ann = confusion_matrix(y_test, y_pred)

summary = list(zip(["ANN", "LR", "RF", "SVM"], [acc_ann, acc_lr, acc_rfr, acc_svr]))
summary = sorted(summary, key=lambda x: x[1])


# plt.bar([item[0] for item in summary], [item[1] for item in summary])


def colors_from_values(values, palette_name):
    # normalize the values to range [0, 1]
    normalized = (values - min(values)) / (max(values) - min(values))
    # convert to indices
    indices = np.round(normalized * (len(values) - 1)).astype(np.int32)
    # use the indices to get the colors
    palette = sns.color_palette(palette_name, len(values))
    return np.array(palette).take(indices, axis=0)


sns.barplot(x=[item[0] for item in summary], y=[item[1] for item in summary],
            palette=colors_from_values([item[1] for item in summary], "flare"))
plt.ylim([np.min([item[1] for item in summary]) - 0.01, np.max([item[1] for item in summary]) + 0.01])
plt.ylabel("Accuracy")
plt.title("Model Comparison")
plt.savefig("diagrams/compare_models.png")
plt.show()
