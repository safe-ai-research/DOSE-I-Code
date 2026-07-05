#
# ██████╗  ██████╗ ███████╗███████╗     ██╗
# ██╔══██╗██╔═══██╗██╔════╝██╔════╝     ██║
# ██║  ██║██║   ██║███████╗█████╗       ██║
# ██║  ██║██║   ██║╚════██║██╔══╝       ██║
# ██████╔╝╚██████╔╝███████║███████╗     ██║
# ╚═════╝  ╚═════╝ ╚══════╝╚══════╝     ╚═╝
#
# A Project by
# SAFE-AI Consortium | UNIVERSITÄTSMEDIZIN HALLE
# https://safe-ai-research.github.io
#
# Code Author: Quang Vu Nguyen
# Licence: GNU GPLv3
#

import numpy as np
import helper_functions as hf
import time
from pathlib import Path

from sklearn.preprocessing import LabelEncoder

from src import helper_functions
from src.summarize import summarize
from src.summarize_mult import summarize as sm
np.set_printoptions(legacy='1.25') # so values display without np.float32(x) etc. in PyCharm's sci-view

folder = Path("timelag+rollingAvg") # path where the input files are
#folder = Path("joined")  # path where the input files are
#Optional

all_files = folder.glob(
    "*.csv")
file_ids = [file.stem for file in all_files]
clfs = ["dummy", "decision-tree", "forest", "NN"]
experimentname = "LocalModel1-noLag-noexclude-70-30-Multi"
startcol = int(2)
labelcol = 1 # important!!!! 1 = multiclass, 0 = binary
for clf in ["dummy", "decision-tree", "forest", "NN"]:
    for file in file_ids:
        losses = []
        aucs = []
        accuracies = []
        precisions = []
        recalls = []
        f1s = []
        confusions = []
        executions = []
        RFImportances = []
        monLabels = []

        subfolder = f"{experimentname}\\{file}"  # dir for output (format is clf + subfolder, e.g. forest/500
        data = f"{folder.name}\\" + file + ".csv"

        temp = np.genfromtxt(data, delimiter=",")
        tscv = hf.non_overlapping_train_test_splits(temp.shape[0], int(temp.shape[0]*0.70), int(temp.shape[0]*0.30), gap_size=0)
        for i, (train_index, test_index) in enumerate(tscv):
            start = time.time()
            # test_index = test_index[::5] # enable to activate non overlapping test
            train_data = temp[train_index]
            test_data = temp[test_index]
            train_X = train_data[:, startcol:]
            train_Y = train_data[:, labelcol]
            test_X = test_data[:, startcol:]
            test_Y = test_data[:, labelcol]
            le = LabelEncoder()

            # Fit and transform
            if labelcol == 1:
                train_Y = le.fit_transform(train_Y)
                test_Y = le.fit_transform(test_Y)
            onlyOneClasslabel = 0
            if (test_Y.max() < 1 or test_Y.min() > 0) and labelcol != 1: #TODO Find a better way to check for testsets with only one class
                onlyOneClasslabel = 1
                #continue

            results, bestmodel = helper_functions.train_and_eval(clf, train_X, train_Y, test_X, test_Y, nout=1,
                           loss_fun="binary_crossentropy",
                           labelcol=labelcol, onlyOneClasslabel=onlyOneClasslabel)
        if labelcol==0:
            losses.append(results["loss"])
            aucs.append(results["auc-roc"])
            accuracies.append(results["accuracy"])
            precisions.append(results["precision"])
            recalls.append(results["recall"])
            f1s.append(results["f1"])
            confusions.append(results["confusion"])
            monLabels.append(results["monotoneClasslabels"])
            end = time.time()
            length = end - start
            results["Execution Time"] = length
            executions.append(length)

            if clf == "forest":
                RFImportances.append(bestmodel.feature_importances_)

            filename = f"\\{subfolder}\\" + "_" + str(train_index[0]) + "-" + str(test_index[0]) + "-" + str(test_index[-1]) + "_startcol" + str(
                startcol) + ".txt"
            Path(f"{clf}\\{subfolder}").mkdir(parents=True, exist_ok=True)
            hf.create_modelresults(clf, filename, results)

            if len(aucs) == 0:
                continue
            filename = f"\\{subfolder}\\" + "ModelFinalResults-" + (str(clf)) + ".txt"

            Path(f"{clf}\\{subfolder}").mkdir(parents=True, exist_ok=True)
            confusion_median = np.median(np.array(confusions), axis=0)
            medianresults = {
                "Median Cross-Entropy Loss": np.median(losses),
                "Median AUC-ROC": np.median(aucs),
                "Median Accuracy": np.median(accuracies),
                "Median Precision": np.median(precisions),
                "Median Recall": np.median(recalls),
                "Median F1 Score": np.median(f1s),
                "Median Confusion Matrix": confusion_median.tolist(),
                "Total Execution Time": np.sum(executions),
                "Median Percentage monotone Class labels": np.mean(monLabels)
            }

            hf.create_modelresults(clf, filename, medianresults)

            if clf == "forest":
                filename = f"\\{subfolder}\\" + "FeatureImportances-" + (str(clf)) + ".txt"
                # hf.plotRFFeatureImportance(clf, filename, RFImportances)
        else:
            accuracies.append(results["accuracy"])
            precisions.append(results["precision"])
            recalls.append(results["recall"])
            f1s.append(results["f1"])
            confusions.append(results["confusion"])
            monLabels.append(results["monotoneClasslabels"])

            # if bestmodel is not None:
            # tscv = TimeSeriesSplit(n_splits=3, max_train_size=10)
            # cv_scores = cross_val_score(bestmodel, train_X, train_Y, cv=tscv, scoring='roc_auc')
            # print(cv_scores)
            end = time.time()
            length = end - start
            results["Execution TIme"] = length
            executions.append(length)

            if clf == "forest":
                RFImportances.append(bestmodel.feature_importances_)

            filename = f"\\{subfolder}\\" + "_" + str(train_index[0]) + "-" + str(test_index[0]) + "-" + str(
                test_index[-1]) + "_startcol" + str(
                startcol) + ".txt"
            Path(f"{clf}\\{subfolder}").mkdir(parents=True, exist_ok=True)
            hf.create_modelresults(clf, filename, results)

            filename = f"\\{subfolder}\\" + "ModelFinalResults-" + (str(clf)) + ".txt"

            Path(f"{clf}\\{subfolder}").mkdir(parents=True, exist_ok=True)
            confusion_median = np.median(np.array(confusions), axis=0).tolist()
            precision_median_per_class = np.median(precisions, axis=0).tolist()
            recall_median_per_class = np.median(recalls, axis=0).tolist()
            f1_median_per_class = np.median(f1s, axis=0).tolist()
            medianresults = {
                "Median Cross-Entropy Loss": np.median(losses),
                "Median Accuracy": np.median(accuracies),
                "Median Precision": precision_median_per_class,
                "Median Recall": recall_median_per_class,
                "Median F1 Score": f1_median_per_class,
                "Median Confusion Matrix": confusion_median,
                "Total Execution Time": np.sum(executions),
                "Median Percentage monotone Class labels": np.mean(monLabels)
            }
            hf.create_modelresults(clf, filename, medianresults)

            if clf == "forest":
                filename = f"\\{subfolder}\\" + "FeatureImportances-" + (str(clf)) + ".txt"
                # hf.plotRFFeatureImportance(clf, filename, RFImportances)
    if labelcol == 0:
        summarize(experimentname, clf)
    else:
        sm(experimentname,clf, all_classes=[0, 1, 2, 3, 4])
