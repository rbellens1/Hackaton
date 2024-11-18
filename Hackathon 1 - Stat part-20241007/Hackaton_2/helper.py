import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import warnings

COLORS = {"linear": "red", "logistic": "green", "knn": "blue"}
SATISFACTORY = {"recall": .95, "f1_score": .75}
VALIDATION_INDEX = {"recall": 0, "f1_score": 2}

def plot_result(result, thresold, to_show = "recall"):
    if to_show not in ["recall", "f1_score"]:
        warnings.warn("You can only show recall of f1_score", UserWarning)        
    plt.axhline(y=SATISFACTORY[to_show], color="black", linestyle="--")
    for m in result: 
        for k in result[m]:
            plt.scatter([k], result[m][k][VALIDATION_INDEX[to_show]], color=COLORS[m], s = 10)
    plt.legend(handles = [mpatches.Patch(color=COLORS[m], label=m) for m in COLORS])
    plt.xlabel("Number of features")
    plt.title("{} with threshold = {}".format(to_show, thresold))
    plt.show()
