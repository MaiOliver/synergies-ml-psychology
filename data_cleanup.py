import pandas as pd
import numpy as np

import datetime as dt

import glob

# location of data files
folder = "./data"

# read selected new files
# get file names
files = glob.glob(folder + "/new_data/*20[1][0-9].csv")
# read csv files into list of pandas dataframes
df_each_file = (pd.read_csv(f, delimiter=";", dtype={"MEETTYPE": 'str'}) for f in files)
# concatenate all dataframes to one dataframe
new_data = pd.concat(df_each_file, ignore_index=True)
# delete the list of dataframes to at least save some memory
del df_each_file

# read old csv file to dataframe
hdata = pd.read_csv(folder + "/CSV Version Effort.csv", delimiter=";", decimal=",")

# prepare old data
# replace "#NULL!" entries with numpys Not a Number
hdata = hdata.replace(r"#NULL!", np.NaN)
# set datatype of columns which had "#NULL!" entries to numeric
hdata["place_r"] = pd.to_numeric(hdata["place_r"])
hdata["place_i"] = pd.to_numeric(hdata["place_i"])
# create relay time
hdata["relay"] = hdata["deltatime"] + hdata["individual"]
# create outcome expectation: "high" for every placement better than 4 and "low" otherwise
hdata["expout"] = np.where(hdata["place_r"] <= 4, "high", "low")

# correct athletes time in relay with reaction offset  // this may have broken stuff soooo yeah
# read offset table into dataset

cor_data = pd.read_csv(folder + "/reactiontimedifference_hueffmeier.csv", delimiter=";")

delta_cor = []
# do for every entry in column "deltatime"
for i, delta in enumerate(hdata["deltatime"]):
    # create column-name of offset dataframe using gender of corresponding athlete
    cor_column = "RTDiff_" + hdata["gender"][i]
    # get the starting position of corresponding athlete and convert it to the index in offset dataframe
    pos = min(hdata["storder"][i] - 1, 3)
    # create corrected time using correct offset
    cor = delta + cor_data[cor_column][pos]
    delta_cor.append(cor)
# write corrected time to dataframe
hdata["deltatimecor"] = delta_cor

# hdata["deltatimecor"] = hdata["deltatimepct"]
"""
# convert some meta-data to high or low valence  // this as alredy been encoded into the "major1" column sans points
# valence is high when swimmer has more than 500 points or if the event is any of the listed international events
points = hdata["points"] >= 500
event = hdata["@_world"] | hdata["@_olympic"] | hdata["@_european"] | hdata["@_universiade"] |\
        hdata["@_panpac"] | hdata["@_commonwealth"]
valence = points | event
hdata["valence"] = valence
hdata.loc[hdata.valence == True, "valence"] = "high"
hdata.loc[hdata.valence == False, "valence"] = "low"
"""
hdata.loc[hdata.major1 == 1, "valence"] = "high"
hdata.loc[hdata.major1 == 0, "valence"] = "low"

hdata["prog"] = (hdata["valence"] == "high") & (hdata["expout"] == "high") & (hdata["storder"] > 1)
hdata["prog"] = hdata["prog"].astype(int)

# create list of column names which are supposed to be treated as categorical data (similar to factors)
categorical = ["style", "gender", "round_i", "round_r", "expout", "storder", "agegroup", "meetid", "athleteid",
               "athletes_team", "valence"]  # place_i/r
# set data type for all listed columns
for column in categorical:
    hdata[column] = hdata[column].astype("category")

# create different selection of columns for different models
big_data_columns = ["gender", "style", "points", "storder", "place_i", "place_r", "round_i", "round_r", "@_world",
                    "@_olympic", "@_european", "@_universiade", "@_panpac", "@_commonwealth", "age", "schedule",
                    "deltatimecor", "individual", "prog"]
selected_data_columns = ["deltatimecor", "expout", "storder", "style", "gender", "points", "age", "valence", "prog"]

columns_cut = set(big_data_columns) - set(selected_data_columns)
all_data_columns = big_data_columns + list(columns_cut)  # unused

# create new dataframes from selected columns and drop any entries with NaN in them
big_data_set = pd.get_dummies(hdata[big_data_columns].dropna())
selected_data_set = pd.get_dummies(hdata[selected_data_columns].dropna())

# save all dataframes to csv files for later use
big_data_set.to_csv(folder+"/cleaned/big_data_set.csv", index=False, sep=";")
selected_data_set.to_csv(folder+"/cleaned/selected_data_set.csv", index=False, sep=";")

# prepare new data
# transcribe "MEETDATE" into two columns formatted as datetime (not really useful as of now)
meetdate_a, meetdate_o = [], []
for date in new_data["MEETDATE"]:
    ao = date.split(" - ")
    if len(ao) <= 1:
        a = ao[0]
        o = ao[0]
    else:
        a = ao[0]
        o = ao[-1]

        points_at = [i for i in range(len(o)) if o.startswith(".", i)]
        count = a.count(".") - 1
        a = a[:-1] + o[points_at[count]:]

    meetdate_a.append(a)
    meetdate_o.append(o)
new_data["MEETDATE_A"] = pd.to_datetime(meetdate_a, format="%d.%m.%Y")
new_data["MEETDATE_O"] = pd.to_datetime(meetdate_o, format="%d.%m.%Y")

# set data types for datetime columns (again not really used as of now)
for column in ["BIRTHDATE", "DATE_I", "DATE_R"]:
    new_data[column] = pd.to_datetime(new_data[column])

# create a column for the year (redundant but for comparability)
new_data["YEAR"] = [x.year for x in new_data["DATE_I"]]
# forgot why this is here
new_data["DELTAABS"] = (new_data["RELAY"] - new_data["INDIVIDUAL"])

# correct time same as old data set  // again this may break stuff

delta_cor = []
for i, delta in enumerate(new_data["DELTAABS"]):
    cor_column = "RTDiff_" + new_data["GENDER"][i]
    pos = min(new_data["ORDER"][i] - 1, 3)
    cor = delta + cor_data[cor_column][pos]
    delta_cor.append(cor)

new_data["DELTACOR"] = delta_cor

new_data = new_data.replace({"Butterfly": "Fly"}, regex=True)
# bulk rename most of the column names to match old data
rename_col = {"DELTAABS": "deltatime", "DELTACOR": "deltatimecor", "ORDER": "storder"}
for column in list(hdata.columns):
    rename_col[column.upper()] = column
new_data = new_data.rename(columns=rename_col)

# create outcome expectation: "high" for every placement better than 4 and "low" otherwise
new_data["expout"] = np.where(new_data["place_r"] <= 4, "high", "low")
new_data = new_data[new_data.storder <= 4]  # some are set to 999 for some reason

new_data["MEETTYPE"] = new_data["MEETTYPE"].astype("category")
new_data = pd.get_dummies(new_data, columns=["MEETTYPE"])
old_meettype_columns = ["@_world", "@_olympic", "@_european", "@_universiade", "@_panpac", "@_commonwealth"]
new_meettype_columns_suffix = ["WC", "OG", "EC", "UR", "PPC", "CG"]
rename_col = {}
for i, column_name in enumerate(old_meettype_columns):
    try:
        new_data["MEETTYPE_"+new_meettype_columns_suffix[i]]
    except KeyError:
        new_data["MEETTYPE_" + new_meettype_columns_suffix[i]] = 0

    rename_col["MEETTYPE_"+new_meettype_columns_suffix[i]] = column_name
new_data = new_data.rename(columns=rename_col)

# convert some meta-data to high or low valence
# valence is high when swimmer has more than 500 points or if the event is any of the listed international events
points = new_data["points"] >= 500
event = new_data["@_world"] | new_data["@_olympic"] | new_data["@_european"] | new_data["@_universiade"] |\
        new_data["@_panpac"] | new_data["@_commonwealth"]
# valence = points | event
valence = event
new_data["valence"] = valence
new_data.loc[new_data.valence == True, "valence"] = "high"  # this could probably be done more elegantly
new_data.loc[new_data.valence == False, "valence"] = "low"

new_data["prog"] = (new_data["valence"] == "high") & (new_data["expout"] == "high") & (new_data["storder"] > 1)
new_data["prog"] = new_data["prog"].astype(int)

# create list of column names which are supposed to be treated as categorical data (similar to factors)
categorical = ["style", "gender", "round_i", "round_r", "expout", "storder", "meetid", "athleteid", "valence"]
# set data type for all listed columns
for column in categorical:
    new_data[column] = new_data[column].astype("category")

# create new dataframes from selected columns
new_big_data_set = pd.get_dummies(new_data[big_data_columns]).drop(columns=["round_i_QUA"])
new_selected_data_set = pd.get_dummies(new_data[selected_data_columns])

try:
    new_big_data_set["round_r_FHT"]
except KeyError:
    new_big_data_set["round_r_FHT"] = 0
new_big_data_set = new_big_data_set[big_data_set.columns]


# convert dates to seconds since epoch (1970-1-1)
date_columns = ["BIRTHDATE", "DATE_I", "DATE_R", "MEETDATE_A", "MEETDATE_O"]
for column in date_columns:
    new_data[column] = pd.to_datetime(new_data[column])
    new_data[column] = (new_data[column] - dt.datetime(1970, 1, 1)).dt.total_seconds()

# save all dataframes to csv files for later use
new_big_data_set.to_csv(folder+"/cleaned/new_big_data_set.csv", index=False, sep=";")
new_selected_data_set.to_csv(folder+"/cleaned/new_selected_data_set.csv", index=False, sep=";")
new_data.to_csv(folder+"/cleaned/full_data_set.csv", index=False, sep=";")
# pd.get_dummies(new_data.sample(100000)).to_csv(folder+"/cleaned/new_complete_data_set.csv", index=False, sep=";")


