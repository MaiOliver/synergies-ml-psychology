from sklearn import tree

import copy as cp
import graphviz

from methods import import_new_data, undummify

seed = 124653


_, _, _, minimal_data = import_new_data(dummies=False)


def make_dt(d):
    df = cp.deepcopy(d)
    df.loc[df.expout == "high", "expout"] = 1
    df.loc[df.expout == "low", "expout"] = 0
    df.loc[df.valence == "high", "valence"] = 1
    df.loc[df.valence == "low", "valence"] = 0
    y = df.pop("cat").astype("int64").values
    clf = tree.DecisionTreeClassifier()
    clf = clf.fit(df, y)
    return clf


decision_tree = make_dt(minimal_data)
dot_data = tree.export_graphviz(decision_tree, out_file=None, filled=True, rounded=True, feature_names=["expout", "valence",
                                                                                               "storder"])
graph = graphviz.Source(dot_data)
graph.render(directory="diagrams/")

