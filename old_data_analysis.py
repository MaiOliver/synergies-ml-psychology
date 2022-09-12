import pandas as pd
import numpy as np
import copy as cp
import time
import datetime as dt
from tabulate import tabulate

from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score

import h2o4gpu as hg
import methods as tnn

# import h2o4gpu.util.metrics as metrics

# set name for folder containing csv data files
folder = "./data"
# set seed for numpys random number generator
glob_seed = 1337
np.random.seed(glob_seed)
valid_size = 100000
# read in csv files prepared in script "Datenaufbereitung.py" as pandas dataframe
# for old data set
big_data_set = pd.read_csv(folder + "/cleaned/big_data_set.csv", delimiter=";", decimal=".")
selected_data_set = pd.read_csv(folder + "/cleaned/selected_data_set.csv", delimiter=";", decimal=".")
# for new data set
new_big_data_set = pd.read_csv(folder + "/cleaned/new_big_data_set.csv", delimiter=";", decimal=".")
new_selected_data_set = pd.read_csv(folder + "/cleaned/new_selected_data_set.csv", delimiter=";", decimal=".")


def test_model(model, data_set, valid_data=None, seed=None, n_samples=1000, dependant="deltatimecor"):
    # Function/Method that fits a model to a data set and returns its R^2 and RMSE values scored on a validation set.
    # Arguments:
    #   model: model object (as created by sklearn or h2o4gpu)
    #   data_set: pandas dataframe object, if no validation set is provided this set is split into 70 % training and
    #             30 % validation data
    #   valid_data: pandas dataframe object, data set used for validation
    #   seed: float, used to seed sampling method
    #   n_samples: integer, size of the sample taken from data_set
    #   dependant: string, column name of dependant variable
    # Outputs:
    #   RMSE: float, Root mean squared error of fitted model on validation data
    #   R2: float, R^2 value of fitted model on validation data
    # generate a sub-sample of data_set of size n_samples
    sample = data_set.sample(random_state=seed, n=n_samples, replace=False).reset_index()
    if valid_data is None:
        # np.random.seed(seed)
        # If no validation data is given do:
        # split sample into train and validation set by index
        # generate floor(70 % of length of data_set) random indices to be used for training
        train_set = sample.sample(frac=0.7, random_state=seed)
        valid_set = sample.drop(train_set.index)
    else:
        # If validation data is given:
        # set all training data to sample of data_set
        train_set = sample
        # set all validation data to sample of valid_data, for now half the size of data_sets sample
        valid_set = valid_data.sample(random_state=seed, n=valid_size, replace=False).reset_index()

    # fit model object using training data. first argument is a dataframe w/o dependant variable second is the same
    #  dataframe w/ only dependant variable
    model.fit(train_set.loc[:, train_set.columns != dependant], train_set[dependant])

    # generate model prediction using validation data
    pred = model.predict(valid_set.loc[:, valid_set.columns != dependant])

    # generate RMSE and R^2 metrics for prediction using actual data and sklearn
    rmse = np.sqrt(mean_squared_error(valid_set[dependant], pred))
    r2 = r2_score(valid_set[dependant], pred)

    return rmse, r2


# The following methods are formulated in bit of a redundant way.
def make_hypothesis_one_three(n_repeats, n_samples=None, validation_sets=[None, None, None, None, None],
                              nn_split=False):
    # Method that uses "test_model" to test the prediction skill of different models.
    # Arguments:
    #       n_repeats: integer, number of times each model is tested
    #       n_samples: integer, size of the sample taken for testing
    #       validation_sets: list-like of dataframes used to pass to "test_model", if none this method creates
    #       hypothesis one if not it should create hypothesis three
    # Outputs:
    #       hypot_one: pandas dataframe, generated testing scores with columns: "Procedure", "R2test", "RMSEtest".

    # generate output for now as python dictionary, with keys corresponding to later column names, and empty arrays
    hypot_one = {"Procedure": [], "R2test": [], "RMSEtest": [], "N_samples": []}

    for N in range(n_repeats):
        # do n_repeats number of times:
        # set new seed for random number generator
        seed = 137 + N
        # set sample size to argument (a bit useless)
        samples = n_samples
        print(f"Iteration: {N}; Samples: {samples}")
        print("Testing linear model")
        # generate linear model object using sklearn
        model_lr = linear_model.LinearRegression()
        # train and test linear model object on (old) preselected data set
        rmse, r2 = test_model(model_lr, selected_data_set, seed=seed, n_samples=samples, valid_data=validation_sets[0])
        print(f"Done with rmse: {rmse} and r2: {r2}")
        # append results to dictionary entries
        hypot_one["Procedure"].append("Theory Model (H)")
        hypot_one["RMSEtest"].append(rmse)
        hypot_one["R2test"].append(r2)
        hypot_one["N_samples"].append(samples)

        print("Testing hybrid random forest")
        # generate random forest regression (rfr) model using h2o4gpu, here number of trees are n_estimators, maximum
        # features are the maximum number of features to consider at each split, n_gpu sets number of gpus to be used
        model_rfr = hg.RandomForestRegressor(n_estimators=500, max_features="auto",
                                             random_state=seed, oob_score=True, n_gpus=1)
        # train and test rfr model on (old) preselected data set
        rmse, r2 = test_model(model_rfr, selected_data_set, seed=seed, n_samples=samples, valid_data=validation_sets[1])
        # append results to dictionary entries
        hypot_one["Procedure"].append("Random Forest (H)")
        hypot_one["RMSEtest"].append(rmse)
        hypot_one["R2test"].append(r2)
        hypot_one["N_samples"].append(samples)
        print(f"Done with rmse: {rmse} and r2: {r2}")

        print("Testing big data random forest")
        # generate another rfr model
        model_rfr = hg.RandomForestRegressor(n_estimators=500, max_features="auto",
                                             random_state=seed, oob_score=True, n_gpus=1)
        # this time test and train rfr model on (old) extended data set
        rmse, r2 = test_model(model_rfr, big_data_set, dependant="deltatimecor", seed=seed, n_samples=samples,
                              valid_data=validation_sets[2])
        # append results to dictionary entries
        hypot_one["Procedure"].append("Random Forest (B)")
        hypot_one["RMSEtest"].append(rmse)
        hypot_one["R2test"].append(r2)
        hypot_one["N_samples"].append(samples)
        print(f"Done with rmse: {rmse} and r2: {r2}")

        print("Testing big data neural network")
        model_nn = tnn.NeuralWrapper(
            selected_data_set.sample(n=samples, random_state=seed, replace=False).reset_index(),
            "deltatimecor", epochs=500, split=nn_split)
        rmse, r2 = model_nn.test_model(valid_set=validation_sets[3])
        hypot_one["Procedure"].append("Neural Network (H)")
        hypot_one["RMSEtest"].append(rmse)
        hypot_one["R2test"].append(r2)
        hypot_one["N_samples"].append(samples)
        print(f"Done with rmse: {rmse} and r2: {r2}")

        print("Testing hybrid neural network")
        model_nn = tnn.NeuralWrapper(big_data_set.sample(n=samples, random_state=seed, replace=False).reset_index(),
                                     "deltatimecor", epochs=500, split=nn_split)
        rmse, r2 = model_nn.test_model(valid_set=validation_sets[4])
        hypot_one["Procedure"].append("Neural Network (B)")
        hypot_one["RMSEtest"].append(rmse)
        hypot_one["R2test"].append(r2)
        hypot_one["N_samples"].append(samples)
        print(f"Done with rmse: {rmse} and r2: {r2}")

        # print("Testing big data support vector regression")
        # generate another rfr model
        # model_svr = hg.svm.LinearSVR(random_state=seed, tol=1e-5, max_iter=100000)
        # this time test and train rfr model on (old) extended data set
        # rmse, r2 = test_model(model_svr, big_data_set, dependant="deltatime", seed=seed, n_samples=samples,
        #                      valid_data=validation_sets[2])
        # append results to dictionary entries
        # hypot_one["Procedure"].append("Support Vector Model")
        # hypot_one["RMSEtest"].append(rmse)
        # hypot_one["R2test"].append(r2)
        # hypot_one["N_samples"].append(samples)
        # print(f"Done with rmse: {rmse} and r2: {r2}")

    # convert dictionary to pandas dataframe for ease of use
    hypot_one = pd.DataFrame.from_dict(hypot_one)

    return hypot_one


def make_hypothesis_two(samples_list, n_repeats):
    # Method that essentially repeats "make_hypothesis_one" for a given list of sample sizes and returns variance and
    # average values of RMSE and R^2 metrics.
    # Note: In principle one should just chuck in "make_hypothesis_one" in that major loop, but the data structure has
    #       been changed a bit here and this leaves a bit more flexibility so this is the way it is for now.
    #       Reading through it now, no clue why I did it the way I did it.
    # Arguments:
    #       samples_list: list-like of integers, list of samples to iterate over
    #       n_repeats: integer, number of times models are tested for each iteration
    # Outputs:
    #       hypot_two: pandas dataframe, congregated results over all sample sizes

    # generate output for now as python dictionary, with keys corresponding to later column names, and empty arrays
    hypot_two = {"Procedure": [], "Mean_R2test": [], "Mean_RMSEtest": [], "Variance_R2test": [],
                 "Variance_RMSEtest": [], "n_samples": [], "n_repeats": []}

    for n_samples in samples_list:
        # for every sample in the list of samples do:
        # create python dictionary for storing results for each model type
        proc = {"R2test": [], "RMSEtest": []}
        # generate a dictionary for this iteration, storing all test results for each model type

        hypot_two_a = {"Random Forest (H)": cp.deepcopy(proc),
                       "Random Forest (B)": cp.deepcopy(proc), "Neural Network (H)": cp.deepcopy(proc),
                       "Neural Network (B)": cp.deepcopy(proc), "Linear Regression (H)": cp.deepcopy(proc)}

        for N in range(n_repeats):
            # do n_repeats number of times:
            # set new seed
            seed = 137 + int(N / 2)
            # set sample size
            samples = n_samples

            # generate linear model object using sklearn
            model_lr = linear_model.LinearRegression()
            # train and test linear model object on (old) preselected data set
            rmse, r2 = test_model(model_lr, selected_data_set, seed=seed, n_samples=samples)
            hypot_two_a["Linear Regression (H)"]["RMSEtest"].append(rmse)
            hypot_two_a["Linear Regression (H)"]["R2test"].append(r2)

            # generate random forest regression (rfr) model using h2o4gpu
            model_rfr = hg.RandomForestRegressor(n_estimators=500, max_features="auto",
                                                 random_state=seed, oob_score=True, n_gpus=1)
            # train and test rfr model using (old) preselected data set
            rmse, r2 = test_model(model_rfr, selected_data_set, seed=seed, n_samples=samples)
            # append results to output for current sample size
            hypot_two_a["Random Forest (H)"]["RMSEtest"].append(rmse)
            hypot_two_a["Random Forest (H)"]["R2test"].append(r2)

            # generate new rfr model
            model_rfr = hg.RandomForestRegressor(n_estimators=500, max_features="auto",
                                                 random_state=seed, oob_score=True, n_gpus=1)
            # train and test rfr model on (old) extended data set
            rmse, r2 = test_model(model_rfr, big_data_set, dependant="deltatimecor", seed=seed, n_samples=samples)
            # append results to output of this iteration
            hypot_two_a["Random Forest (B)"]["RMSEtest"].append(rmse)
            hypot_two_a["Random Forest (B)"]["R2test"].append(r2)

            model_nn = tnn.NeuralWrapper(
                selected_data_set.sample(n=samples, random_state=seed, replace=False).reset_index(),
                "deltatimecor", epochs=500)
            rmse, r2 = model_nn.test_model()
            hypot_two_a["Neural Network (H)"]["RMSEtest"].append(rmse)
            hypot_two_a["Neural Network (H)"]["R2test"].append(r2)

            model_nn = tnn.NeuralWrapper(big_data_set.sample(n=samples, random_state=seed, replace=False).reset_index(),
                                         "deltatimecor", epochs=500)
            rmse, r2 = model_nn.test_model()
            hypot_two_a["Neural Network (B)"]["RMSEtest"].append(rmse)
            hypot_two_a["Neural Network (B)"]["R2test"].append(r2)

        for procedure in hypot_two_a.keys():
            # for every model do:
            # append the model name to list of procedures
            hypot_two["Procedure"].append(procedure)
            # calculate average values of R^2 and RMSE for current sample size
            mean_r2test = np.mean(hypot_two_a[procedure]["R2test"])
            mean_rmsetest = np.mean(hypot_two_a[procedure]["RMSEtest"])
            # calculate variance of R^2 and RMSE for current sample size
            var_r2test = np.var(hypot_two_a[procedure]["R2test"])
            var_rmsetest = np.var(hypot_two_a[procedure]["RMSEtest"])
            # append means and variances to overall output
            hypot_two["Variance_R2test"].append(var_r2test)
            hypot_two["Variance_RMSEtest"].append(var_rmsetest)
            hypot_two["Mean_R2test"].append(mean_r2test)
            hypot_two["Mean_RMSEtest"].append(mean_rmsetest)
            hypot_two["n_samples"].append(n_samples)
            hypot_two["n_repeats"].append(n_repeats)

        # generate commandline output to follow along
        print(f"\nFor {n_samples} samples and {n_repeats} repeats:")
        # should've used list comprehension
        print(tabulate([[hypot_two["Procedure"][-5], hypot_two["Mean_R2test"][-5], hypot_two["Mean_RMSEtest"][-5],
                         hypot_two["Variance_R2test"][-5], hypot_two["Variance_RMSEtest"][-5]],
                        [hypot_two["Procedure"][-4], hypot_two["Mean_R2test"][-4], hypot_two["Mean_RMSEtest"][-4],
                         hypot_two["Variance_R2test"][-4], hypot_two["Variance_RMSEtest"][-4]],
                        [hypot_two["Procedure"][-3], hypot_two["Mean_R2test"][-3], hypot_two["Mean_RMSEtest"][-3],
                         hypot_two["Variance_R2test"][-3], hypot_two["Variance_RMSEtest"][-3]],
                        [hypot_two["Procedure"][-2], hypot_two["Mean_R2test"][-2], hypot_two["Mean_RMSEtest"][-2],
                         hypot_two["Variance_R2test"][-2], hypot_two["Variance_RMSEtest"][-2]],
                        [hypot_two["Procedure"][-1], hypot_two["Mean_R2test"][-1], hypot_two["Mean_RMSEtest"][-1],
                         hypot_two["Variance_R2test"][-1], hypot_two["Variance_RMSEtest"][-1]]
                        ], headers=["Procedure", "Mean R2", "Mean RMSE", "Var R2", "Var RMSE"]))


    # convert output into dataframe for ease of use
    hypot_two = pd.DataFrame.from_dict(hypot_two)

    return hypot_two


hypothesis_one = make_hypothesis_one_three(10, 20000, nn_split=True)
hypothesis_one.to_csv(folder + "/lr_rf_nn_hybrid_big_hypot_1.csv", index=False, sep=";")

samples = list(range(50, 1500, 200)) + \
          list(range(1500, 10000, 500)) + list(range(10000, 110000, 10000))
samples = list(range(50, 10150, 150))
# samples = list(range(100, 1000, 100))
samples = [10, 100, 1000, 5000, 10000, 100000]
hypothesis_two = make_hypothesis_two(samples, n_repeats=10)
hypothesis_two.to_csv(folder + "/lr_rf_nn_hybrid_big_hypot_2.csv", index=False, sep=";")

hypothesis_three = make_hypothesis_one_three(10, n_samples=20000, validation_sets=[new_selected_data_set,
                                                                                     new_selected_data_set,
                                                                                     new_big_data_set,
                                                                                     new_selected_data_set.sample(
                                                                                         random_state=glob_seed,
                                                                                         n=valid_size,
                                                                                         replace=False).reset_index(),
                                                                                     new_big_data_set.sample(
                                                                                         random_state=glob_seed,
                                                                                         n=valid_size,
                                                                                         replace=False).reset_index()])
hypothesis_three.to_csv(folder + "/lr_rf_nn_hybrid_big_hypot_3.csv", index=False, sep=";")

# rmse_lr = []
# rmse_rfr = []
# r2_lr = []
# r2_rfr = []

# N_range = range(1000, 100001, 1000)
##N_range = [100]

# for N in N_range:
# seed = 137+int(N/2)
# np.random.seed(seed)

# hdatas = selected_data_set.sample(random_state=seed, n=N, replace=False).reset_index()
# train_indices = np.random.choice(range(0, len(hdatas)), int(0.7*len(hdatas)), replace=False)
# valid_indices = np.delete(range(0, len(hdatas)), train_indices)

# TrainSet = hdatas.iloc[train_indices].reset_index()
# ValidSet = hdatas.iloc[valid_indices].reset_index()

# model_lr = linear_model.LinearRegression()
# model_lr.fit(TrainSet.loc[:, TrainSet.columns != "deltatime"], TrainSet.deltatime)

##model_lr = hg.solvers.linear_regression.LinearRegression()
##model_lr.fit(TrainSet.loc[:, TrainSet.columns != "deltatime"], TrainSet.deltatime)


# pred_lr = model_lr.predict(ValidSet.loc[:, TrainSet.columns != "deltatime"])
# rmse_lr.append(mean_squared_error(ValidSet.deltatime, pred_lr, squared=False))
# r2_lr.append(r2_score(ValidSet.deltatime, pred_lr))


# model_rfr = hg.RandomForestRegressor(n_estimators=250, max_features=30, random_state=seed, oob_score=False,
# n_gpus=1)
# model_rfr.fit(TrainSet.loc[:, TrainSet.columns != "deltatime"], TrainSet.deltatime)

# pred_rfr = model_rfr.predict(ValidSet.loc[:, TrainSet.columns != "deltatime"])
# rmse_rfr.append(mean_squared_error(ValidSet.deltatime, pred_rfr, squared=False))
# r2_rfr.append(r2_score(ValidSet.deltatime, pred_rfr))

# plt.figure(1)
# plt.plot(N_range, r2_lr, color="r", label="Linear Regression")
# plt.plot(N_range, r2_rfr, color="b", label="Random Forest")
# plt.legend()
# plt.ylabel("R^2")
# plt.xlabel("Sample Size")
# plt.savefig("./diagrams/lr_vs_rfr_r2.png")

# plt.figure(2)
# plt.plot(N_range, mse_lr, color="r", label="Linear Regression")
# plt.plot(N_range, mse_rfr, color="b", label="Random Forest")
# plt.legend()
# plt.ylabel("MSE")
# plt.xlabel("Sample Size")
# plt.savefig("./diagrams/lr_vs_rfr_mse.png")
# plt.show()
