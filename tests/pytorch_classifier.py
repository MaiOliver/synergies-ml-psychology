import torch
import pandas as pd
import datetime as dt
import math

seed = 1334

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


def make_cat(df):
    df.loc[df.deltatimecor >= 0, "cat"] = 0
    df.loc[df.deltatimecor < 0, "cat"] = 1
    time = df.pop("deltatimecor")


for df in [data, new_selected_data_set, new_big_data_set, minimal_data]:
    make_cat(df)

S = set(new_big_data_set.columns) - set(["cat"])

y = torch.tensor(new_big_data_set["cat"].astype("int64").values)

device = torch.device("cuda:0")  # "cpu" else
data = torch.tensor(new_big_data_set.drop("cat", axis=1).values)

model = torch.nn.Sequential(
    torch.nn.Linear(data.shape[1], 5),
    torch.nn.ELU(),
    torch.nn.Linear(5, 1),
    torch.nn.Sigmoid()
)

loss_fn = torch.nn.MSELoss()
lr = 1e-5

optimizer = torch.optim.Adam(model.parameters(), lr=lr)

for t in range(100):
    y_pred = model(data)

    loss = loss_fn(y_pred, y)

    if t % 10 == 9:
        print(t, loss.item())

    loss.backward()
    optimizer.step()


