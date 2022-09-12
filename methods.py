import numpy as np
import pandas as pd

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as be
from tensorflow.keras import layers

# import tensorflow_docs as tfdocs
# import tensorflow_docs.modeling

from sklearn.metrics import mean_squared_error, r2_score


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



def create_baseline_model(length):
    # Helper function that creates a "simple" neural network for classification
    model = keras.Sequential()
    model.add(layers.Dense(length, input_dim=length, activation="elu"))
    model.add(layers.Dense(5, activation="elu"))
    # model.add(layers.Dense(2, activation="elu"))
    model.add(layers.Dense(1, activation="sigmoid"))

    model.compile(loss="mean_squared_error", optimizer="adam", metrics=["accuracy"])
    return model


def create_extensive_model(length):
    # Helper function that creates a more "extensive" neural network with more layers and paramters
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


def import_new_data(dummies=True):
    # Import and classify data (classification should maybe be done in "data_cleanup.py")
    folder = "./data"

    data = pd.read_csv(folder + "/cleaned/full_data_set.csv", delimiter=";", decimal=".")

    category_columns = ["gender", "NATION", "style", "round_i", "round_r", "MEETCITY", "MEETNATION"]
    for column in category_columns:
        data[column] = data[column].astype("category")

    new_selected_data_set = pd.read_csv(folder + "/cleaned/new_selected_data_set.csv", delimiter=";", decimal=".")
    new_big_data_set = pd.read_csv(folder + "/cleaned/new_big_data_set.csv", delimiter=";", decimal=".")

    minimal_data = pd.DataFrame()
    for column in ["expout", "valence", "storder", "deltatimecor"]:
        minimal_data[column] = data[column]
    if dummies:
        minimal_data = pd.get_dummies(minimal_data)

    def make_cat(df):
        df.loc[df.deltatimecor >= 0, "cat"] = 0
        df.loc[df.deltatimecor < 0, "cat"] = 1
        time = df.pop("deltatimecor")

    for df in [data, new_selected_data_set, new_big_data_set, minimal_data]:
        make_cat(df)

    return data, new_big_data_set, new_selected_data_set, minimal_data


class NeuralWrapper:
    seed = 1337

    def __init__(self, dataset, var, epochs=1000, seed=seed, batch_size=None, split=True):
        # var = "deltatime"
        self.pred = None
        self.dependant = var
        np.random.seed(seed)
        if split:
            self.train_set = dataset.sample(frac=0.7, random_state=self.seed)
            # use remaining indices as validation
            # read data from data_set into variable using corresponding set of indices
            self.valid_set = dataset.drop(self.train_set.index)
        else:
            self.train_set = dataset
            self.valid_set = None

        self.train_stats = self.train_set.describe()
        self.train_stats.pop(var)
        self.train_stats = self.train_stats.transpose()

        train_labels = self.train_set.pop(var)

        normed_train_data = self.norm(self.train_set)

        def coeff_determination(y_true, y_pred):
            ss_res = be.sum(be.square(y_true - y_pred))
            ss_tot = be.sum(be.square(y_true - be.mean(y_true)))

            return 1 - ss_res / (ss_tot + be.epsilon())

        def r2_loss(y_true, y_pred):
            ss_res = be.sum(be.square(y_true - y_pred))
            ss_tot = be.sum(be.square(y_true - be.mean(y_true)))

            return ss_res / (ss_tot + be.epsilon())

        def build_my_model():
            model = keras.Sequential()

            model.add(layers.Dense(500, input_shape=[self.train_set.shape[1]], activation="elu"))
            model.add(layers.Dense(200, activation='elu'))
            model.add(layers.Dropout(0.1, seed=self.seed))
            model.add(layers.Dense(100, activation='elu'))
            model.add(layers.Dense(50, activation='elu'))
            # model.add(layers.Dense(4, activation="linear"))
            model.add(layers.Dense(1, activation="linear"))

            optimizer = tf.keras.optimizers.SGD(0.0008, momentum=0.8)
            model.compile(loss=r2_loss, optimizer=optimizer, metrics=["mae", "mse", coeff_determination])

            return model

        self.model = build_my_model()

        self.epochs = epochs

        if batch_size is None:
            self.batch_size = int(len(normed_train_data) / 1)

            if self.batch_size == 0:
                self.batch_size += 1
            elif 100 < self.batch_size <= 10000:
                self.batch_size = int(len(normed_train_data) / 10)
            elif self.batch_size > 10000:
                self.batch_size = 10000
        else:
            self.batch_size = batch_size

        self.model.fit(normed_train_data, train_labels, epochs=self.epochs, batch_size=self.batch_size,
                       validation_split=0.0, verbose=0)  # , callbacks=[tfdocs.modeling.EpochDots()])

    def test_model(self, valid_set=None):
        if valid_set is None:
            test_set = self.valid_set
        else:
            test_set = valid_set
            if self.valid_set is None:
                self.valid_set = valid_set

        pred = self.model.predict(self.norm(test_set.loc[:, test_set.columns != self.dependant])).flatten()
        # self.pred = pred
        print(f"NaN in pred: {np.isnan(pred).any()}")
        print(f"NaN in valid set: {test_set[self.dependant].isnull().values.any()}")
        # generate RMSE and R^2 metrics for prediction using actual data and sklearn
        y_true = test_set[self.dependant][np.invert(np.isnan(pred))]
        pred = pred * (self.train_set.max() - self.train_set.min()) + self.train_set.min()
        pred = pred[np.invert(np.isnan(pred))]
        try:
            rmse = np.sqrt(mean_squared_error(y_true, pred))
        except ValueError:
            rmse = np.nan

        try:
            r2 = r2_score(y_true, pred)
        except ValueError:
            r2 = np.nan
        return rmse, r2

    def norm(self, x):
        # n = (x - self.train_stats["mean"]) / self.train_stats["std"]
        n = (x - self.train_set.min()) / (self.train_set.max() - self.train_set.min())
        return n.fillna(0)
