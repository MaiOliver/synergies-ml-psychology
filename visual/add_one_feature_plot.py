import numpy as np
import json
import matplotlib.pyplot as plt

with open("./data/categorical/addone_feature.json", "r") as file:
    addone = json.load(file)

diffs = [(k, v) for k, v in addone.items()]
diffs.sort(key=lambda a: a[1])

# Plot the four features that by themselves helped improve the prediction the most
fig, ax = plt.subplots(figsize=(6, 3))
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
