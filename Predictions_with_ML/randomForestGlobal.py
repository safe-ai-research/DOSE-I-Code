import keras
import sklearn
import numpy as np
import helper_functions as hf
from keras.api import regularizers
import glob
from sklearn.metrics import log_loss, roc_auc_score, accuracy_score, confusion_matrix, recall_score, precision_score, \
    f1_score, roc_curve, mean_absolute_error, auc
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import TimeSeriesSplit, KFold, cross_val_score
import time
from pathlib import Path
from imblearn.ensemble import BalancedRandomForestClassifier
from src.summarize import summarize

np.set_printoptions(legacy='1.25') # so values display without np.float32(x) etc. in PyCharm's sci-view
# so the script runs reliably on a single core on the cluster
#  tf.config.threading.set_intra_op_parallelism_threads(1)
#  tf.config.threading.set_inter_op_parallelism_threads(1)

folder = Path("timelag+rollingAvg") # path where the input files are
#Optional
# subfolder = "withWindows100" # dir for output (format is clf + subfolder, e.g. forest/500
all_files = folder.glob(
    "*.csv")
file_ids = [file.stem for file in all_files]
clfs = ["dummy", "decision-tree", "forest", "NN"]
experimentname = "GlobalModel1-Lag5s-roliingAvg10-30-noexclude"
kf = KFold(n_splits=10, random_state=None, shuffle=True)

for clf in ["dummy", "decision-tree", "forest", "NN"]:

    losses = []
    aucs = []
    accuracies = []
    precisions = []
    recalls = []
    f1s = []
    confusions = []
    executions = []
    RFImportances = []

    for i, (train_index, test_index) in enumerate(kf.split(file_ids)):  # ID of the test file
        start = time.time()
        #id = id
        # boolean: lr = logistic regression (via NN), otherwise "real" NN (see below)
        lr = "lr"  # (sys.argv[2] == "lr")
        # loss function: bce = binary crossentropy, otherwise DOUBLELOSS from above
        bce = "bce"
        # column from which values are considered; always set to 2 in experiments (the "real" data starts there)
        # column 0: sedation state, column 1: sedation depth
        startcol = int(2)
        # fraction of data to use, e.g. 8 -> 1/8 of data is used
        subsamp = int(1)

        # file with test data, see example 0003.tsv
        #test_file = "joined\\" + id + ".csv"  # super important to use \\ and not / when using windows! else "joined/" -> "joined/003.csv, however:
        test_file = [f"{folder.name}\\" + file_ids[i] + ".csv" for i in test_index]
        # all data
        all_files = list(glob.glob(
            f"{folder.name}/*.csv"))  # here the result is "joined\\003.csv", this difference would lead to the test file  not being excluded from the training data which is no good
        # exclude test data
        train_files = [f for f in all_files if f not in test_file]

        all = "None"
        first = True
        # read and merge training data
        for file in train_files:
            temp = np.genfromtxt(file, delimiter=",")
            if first:
                all = temp
                first = False
            else:
                all = np.vstack((all, temp))

        testall = "None"
        first = True
        for file in test_file:
            temp = np.genfromtxt(file, delimiter=",")
            if first:
                testall = temp
                first = False
            else:
                testall = np.vstack((testall, temp))

        train_data = all

        # extract features
        train_X = train_data[:, startcol:]
        #print("dimensions", train_X.shape)

        # select columns depending on the loss function
        # column 0: sedation state, column 1: sedation depth
        if bce:
            train_Y = train_data[:, 0]
            loss_fun = "binary_crossentropy"
            nout = 1
        else:
            train_Y = train_data[:, 0:2]
            #loss_fun = DOUBLELOSS
            nout = 2

        #test_data = genfromtxt(test_file, delimiter=",")
        test_data = testall
        test_X = test_data[:, startcol:]
        test_Y = test_data[:, 0]
        n_jobs = -1

        # time_window = series_to_supervised(train_data, n_in=10, n_out=1, dropnan=True)

        best_loss = 1000000
        best_auc = 0
        best_acc = 0
        bestmodel = "None"
        bestY = 0
        for i in range(1):
            if clf == "dummy":
                model = DummyClassifier(strategy="stratified")
            elif clf == "decision-tree":
                model = DecisionTreeClassifier()
            elif clf == "forest":
                # model = RandomForestClassifier(n_estimators=200, n_jobs=-1)
                model = BalancedRandomForestClassifier(n_estimators=250, n_jobs=-1)
            else:
                model = keras.models.Sequential()
                model.add(keras.layers.Dense(units=10, input_dim=train_X.shape[1], activation='relu',
                                             kernel_regularizer=regularizers.L2(0.001)))
                model.add(keras.layers.Dropout(0.1))
                model.add(keras.layers.Dense(units=5, activation='relu', kernel_regularizer=regularizers.L2(0.001)))
                model.add(keras.layers.Dropout(0.1))
                model.add(
                    keras.layers.Dense(units=nout, activation='sigmoid', kernel_regularizer=regularizers.L2(0.001)))
                model.compile(loss=loss_fun, optimizer="adam", metrics=[keras.metrics.AUC()])
                # model = SKLearnClassifier(model=createJanModel, model_kwargs={"loss": loss_fun,}, )
                # with parallel_backend('threading', n_jobs=n_jobs):
                model.fit(train_X, train_Y, epochs=50, batch_size=64, verbose=0, shuffle=True)

            if clf != "NN":
                # with parallel_backend('threading', n_jobs=n_jobs):  # my cpu has 6 physical cores
                model.fit(train_X, train_Y)
                y_prediction = model.predict(test_X)
                if clf == "decision-tree":
                    y_prediction = (y_prediction > 0.5).astype(int)

            else:
                y_prediction = model.predict(test_X).ravel()
                y_prediction = (y_prediction > 0.5).astype(int)

            loss = log_loss(test_Y, y_prediction)
            losses.append(loss)
            if clf == None: #TODO make it compatible with the createresults function
                fpr_keras, tpr_keras, thresholds_keras = roc_curve(test_Y, y_prediction)
                auc = sklearn.metrics.auc(fpr_keras, tpr_keras)
                # acc = accuracy_score(test_Y, y_prediction)
                # print(model.get_metrics_result())
            else:
                fpr_keras, tpr_keras, thresholds_keras = roc_curve(test_Y, y_prediction)
                # auc = roc_auc_score(test_Y, y_prediction)

            # remember best model
            if loss < best_loss:
                if clf != "NN":
                    bestmodel = model
                else:
                    bestmodel = None
                confusion = confusion_matrix(test_Y, y_prediction)
                tn, fp, fn, tp = confusion.ravel()

                results = {
                    "loss": loss,
                    "auc-roc": auc(fpr_keras, tpr_keras),
                    "accuracy": accuracy_score(test_Y, y_prediction),
                    "precision": precision_score(test_Y, y_prediction),
                    "recall": recall_score(test_Y, y_prediction),
                    "f1": f1_score(test_Y, y_prediction),
                    "confusion": [int(tp), int(fp), int(fn), int(tn)]
                }


                aucs.append(results["auc-roc"])
                accuracies.append(results["accuracy"])
                precisions.append(results["precision"])
                recalls.append(results["recall"])
                f1s.append(results["f1"])
                confusions.append(results["confusion"])


        #if bestmodel is not None:
        #tscv = TimeSeriesSplit(n_splits=3, max_train_size=10)
        #cv_scores = cross_val_score(bestmodel, train_X, train_Y, cv=tscv, scoring='roc_auc')
        #print(cv_scores)
        end = time.time()
        length = end - start
        results["Execution Time"] = length
        executions.append(length)

        if clf == "forest":
            RFImportances.append(bestmodel.feature_importances_)

        Path(f"{clf}\\{experimentname}\\{str(test_index[0])}").mkdir(parents=True, exist_ok=True)
        filename = f"\\{experimentname}\\{str(test_index[0])}\\" + "subsample_" + str(subsamp) + "_" + str(test_index[0]) + "-" + str(test_index[-1]) + "_lr" + str(
            lr) + "_bce" + str(bce) + "_startcol" + str(
            startcol) + ".txt"

        hf.create_modelresults(clf, filename, results)



        filename = f"\\{experimentname}\\{str(test_index[0])}\\" + "ModelFinalResults-" + (str(clf)) + ".txt"
        confusion_median = np.median(np.array(confusions), axis=0)
        medianresults = {
            "Median Cross-Entropy Loss": np.median(losses),
            "Median AUC-ROC": np.median(aucs),
            "Median Accuracy": np.median(accuracies),
            "Median Precision": np.median(precisions),
            "Median Recall": np.median(recalls),
            "Median F1 Score": np.median(f1s),
            "Median Confusion Matrix": confusion_median.tolist(),
            "Total Execution Time": np.sum(executions)
        }
        hf.create_modelresults(clf, filename, medianresults)

        if clf == "forest":
            filename = f"\\{experimentname}\\" + "FeatureImportances-" + (str(clf)) + ".txt"
            hf.plotRFFeatureImportance(clf, filename, RFImportances)

    summarize(experimentname, clf)
