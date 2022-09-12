import numpy as np
from tqdm import tqdm  # for progress bar in loop

from methods import import_new_data, create_baseline_model

seed = 124653

data, new_big_data_set, new_selected_data_set, minimal_data = import_new_data()


def train_test(N, wo_theory=True, repeats=100, batch_size=None, epochs=100):
    acc = []
    pbar = tqdm(range(repeats))
    for i in pbar:
        if wo_theory:
            data_loc = new_big_data_set
        else:
            data_loc = new_selected_data_set

        dat = data_loc.sample(N, random_state=seed + i).reset_index()

        labels = dat.pop("cat")
        dat = dat.drop(["index"], axis=1)

        # normalized_df = (dat-dat.min())/(dat.max()-dat.min())  # Pandas should apply these functions column-wise
        normalized_df = (dat - dat.mean()) / (dat.std())

        x_train = normalized_df  # .sample(frac=0.7, random_state=seed)
        # x_train, x_test, y_train, y_test = train_test_split(df, labels, test_size=0.25, random_state=seed+i)
        y_train = labels

        model = create_baseline_model(length=x_train.shape[1])

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

    return acc


Ns = np.arange(10, 500, 10)
reps = 20
acc_wo_theory = []
acc_w_theory = []
for n in Ns:
    acc_wo_theory.append(train_test(n, repeats=reps))
    acc_w_theory.append(train_test(n, wo_theory=False, repeats=reps))

np.savetxt("./data/categorical/raw_nn_acc_vs_n_wo.csv", acc_wo_theory, delimiter=";")
np.savetxt("./data/categorical/raw_nn_acc_vs_n_w.csv", acc_w_theory, delimiter=";")
np.savetxt("./data/categorical/raw_nn_acc_vs_n_ns.csv", Ns, delimiter=";")
