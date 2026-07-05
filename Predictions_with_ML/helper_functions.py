import numbers
from pathlib import Path

import keras
import numpy as np
import matplotlib
import sklearn
from keras.api import regularizers
from pandas import DataFrame
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, log_loss, roc_curve, \
    confusion_matrix
from sklearn.model_selection import KFold
from sklearn.tree import DecisionTreeClassifier

matplotlib.use("Agg")
from matplotlib import pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.utils.class_weight import compute_class_weight

def load_feature_names(file_path: str) -> list:
    with open(file_path, 'r') as f:
        feature_names = [line.strip() for line in f if line.strip()]
    return feature_names

def create_rolling_windows(data, window_seconds: int = 5, window_step: int = 5):
    window_size = window_seconds
    num_rows, num_features = data.shape

    X = []
    feature_names = data.columns.to_list()
    #feature_names = load_feature_names("joined\\colnames.txt")
    exclude_cols = ["SOC", "MOAAS", 'GENDER', 'HEIGHT', 'ASA', "AGE", "BMI", "WEIGHT", ]  # or whatever else
    mean_cols = [col for col in feature_names if "mean" in col.lower()]
    for col in mean_cols:
        exclude_cols.append(col)
    data = pd.DataFrame(data=data, columns=feature_names)

    # Step 2: split the data
    target = data["SOC"].to_numpy()
    rolling_features = data.drop(columns=exclude_cols)
    static_features = data[exclude_cols]

    col_names = []
    for t in range(-window_size + 1, 1):  # t = -10 to -1 (past to just before prediction)
        for feature in rolling_features:
            col_names.append(f"{feature}_t{t}")

    for i in range(0, num_rows - window_size, window_step):
        window = rolling_features.values[i:i + window_size].flatten()  # shape: (10, 35)    # shape: (35,)
        X.append(window)
    # Step 4 (optional): Reattach static features (note: needs alignment!)
    # We'll take the corresponding static row at the target time
    #aligned_static = static_features[10:].reset_index(drop=True)  # skip first 10 rows

    X_df = pd.DataFrame(X, columns=col_names)

    final_df = pd.concat([static_features[:X_df.shape[0]], X_df], axis=1)
    return final_df, target[:X_df.shape[0]]


def plotRFFeatureImportance(clf, filename, list_of_importances):
    feature_dict = {}
    median_of_importances = np.median(np.array(list_of_importances), axis=0)
    feature_names = load_feature_names("timelag+rollingAvg+Step1s\\colnames.txt")
    feature_names.remove("SOC")
    feature_names.remove("MOAAS")

    f = open(clf + filename, "w")
    ranked = sorted(enumerate(median_of_importances), key=lambda x: x[1], reverse=True)

    for feature in ranked:
        f.write(f"{feature_names[feature[0]]}: {feature[1]}")
        f.write("\n")
        feature_dict[feature_names[feature[0]]] = feature[1]

    f.close()

    #df = pd.DataFrame({'Category': list(feature_dict.keys()), 'Value': list(feature_dict.values())})
    #sns.barplot(data=df, y='Category', x='Value')
    plt.barh(list(feature_dict.keys())[:50], list(feature_dict.values())[:50], color='skyblue')
    plt.xlabel('Gini Importance')
    plt.title('Feature Importance - Gini Importance (Top 50)')
    plt.gca().invert_yaxis()  # Invert y-axis for better visualization
    plt.tight_layout()
    plt.show()
    plt.close()
    #print(median_of_importances)


def create_modelresults(clf, filename, resultdict):
    # write out results
    f = open(clf + filename, "w")
    for key, value in resultdict.items():
        if isinstance(value, numbers.Number):
            f.write(f"{clf}: {key}: {value:.2f}")
        else:
            f.write(f"{clf}: {key}: {value}")
        f.write("\n")

    # sklearn.metrics.RocCurveDisplay.from_estimator(clf, test_X, test_Y)
    f.close()


def non_overlapping_train_test_splits(data_length, train_size, test_size, gap_size):
    start = 0
    while start + train_size + test_size <= data_length:
        train_start = start
        train_end = train_start + train_size
        test_start = train_end + gap_size
        test_end = test_start + test_size

        train_indices = list(range(train_start, train_end))
        test_indices = list(range(test_start, test_end))

        yield train_indices, test_indices
        start = test_end  # jump forward → no overlap


def train_and_eval(clf, train_X, train_Y, test_X, test_Y, nout=1, loss_fun="binary_crossentropy",
                   labelcol=0, onlyOneClasslabel=0):
    best_loss = 1000000
    best_auc = 0
    best_acc = 0
    bestmodel = "None"
    bestY = 0
    classes = np.unique(train_Y)
    weights = compute_class_weight(class_weight='balanced', classes=classes, y=train_Y)
    class_weights = dict(zip(classes, weights))
    for i in range(1):
        if clf == "dummy":
            model = DummyClassifier(strategy="stratified")
        elif clf == "decision-tree":
            model = DecisionTreeClassifier()
        elif clf == "forest":
            model = RandomForestClassifier(n_estimators=100, n_jobs=-1)
        elif clf == "NN":
            if labelcol==0:
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
            elif labelcol==1:
                model = keras.models.Sequential()
                model.add(keras.layers.Dense(units=10, input_dim=train_X.shape[1], activation='relu',
                                             kernel_regularizer=regularizers.L2(0.001)))
                model.add(keras.layers.Dropout(0.1))
                model.add(keras.layers.Dense(units=5, activation='relu', kernel_regularizer=regularizers.L2(0.001)))
                model.add(keras.layers.Dropout(0.1))
                model.add(
                    keras.layers.Dense(units=5, activation='softmax',
                                       kernel_regularizer=regularizers.L2(0.001)))  # untis = l
                model.compile(loss="sparse_categorical_crossentropy", optimizer="adam")
                # model = SKLearnClassifier(model=createJanModel, model_kwargs={"loss": loss_fun,}, )
                # with parallel_backend('threading', n_jobs=n_jobs):
                model.fit(train_X, train_Y, class_weight=class_weights, epochs=50, batch_size=64, verbose=1,
                          shuffle=True)
            else:
                raise Exception("wrong labelcol inputted !?")


        #binary class
        if labelcol == 0:
            if clf != "NN":
                # with parallel_backend('threading', n_jobs=n_jobs):  # my cpu has 6 physical cores
                model.fit(train_X, train_Y)
                y_prediction = model.predict(test_X)
                # y_prediction = model.predict_proba(test_X)
                if clf == "decision-tree":
                    y_prediction = (y_prediction > 0.5).astype(int)

            else:
                y_prediction = model.predict(test_X).ravel()
                y_prediction = (y_prediction > 0.5).astype(int)
            if labelcol == 0:
                loss = log_loss(test_Y, y_prediction, labels=[0, 1])
            else:
                loss = log_loss(test_Y, y_prediction, labels=[0, 1, 2, 3, 4])

            if clf == None:  # TODO make it compatible with the createresults function
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
                confusion = confusion_matrix(test_Y, y_prediction, labels=[0, 1])
                tn, fp, fn, tp = confusion.ravel()

                results = {
                        "loss": loss,
                        "auc-roc": sklearn.metrics.auc(fpr_keras, tpr_keras),
                        "accuracy": accuracy_score(test_Y, y_prediction),
                        "precision": precision_score(test_Y, y_prediction),
                        "recall": recall_score(test_Y, y_prediction),
                        "f1": f1_score(test_Y, y_prediction),
                        "confusion": [int(tp), int(fp), int(fn), int(tn)],
                        "monotoneClasslabels": onlyOneClasslabel
                }

        # multiclass
        elif labelcol == 1:
            if clf != "NN":
                # with parallel_backend('threading', n_jobs=n_jobs):  # my cpu has 6 physical cores
                model.fit(train_X, train_Y)
                # y_prediction = model.predict(test_X)

                all_classes = np.array([0, 1, 2, 3, 4])  # your expected classes

                # model.classes_ contains only the classes that appeared in training
                pred_classes = model.classes_

                # predict_proba only outputs columns for the classes it knows
                y_pred_proba_raw = model.predict_proba(test_X)  # shape: (n_samples, len(model.classes_))

                # create a full (n_samples, n_classes) matrix with zeros
                y_pred_proba_full = np.zeros((y_pred_proba_raw.shape[0], len(all_classes)))

                # fill in the probabilities for the classes the model has
                for idx, cls in enumerate(pred_classes):
                    full_idx = np.where(all_classes == cls)[0][0]
                    y_pred_proba_full[:, full_idx] = y_pred_proba_raw[:, idx]

                # y_prediction = model.predict_proba(test_X)
                y_discrete_prediction = np.argmax(y_pred_proba_full, axis=1)
                # y_prediction = model.predict_proba(test_X)
                if clf == "decision-tree":
                    y_prediction = (y_discrete_prediction > 0.5).astype(int)

            else:
                # y_prediction = model.predict(test_X).ravel()
                # y_prediction = (y_prediction > 0.5).astype(int)

                all_classes = np.array([0, 1, 2, 3, 4])  # expected classes

                # keras model always outputs probabilities for all outputs (units=nout)
                y_pred_proba_raw = model.predict(test_X, verbose=0)  # shape: (n_samples, nout)

                # If model outputs fewer units than all_classes (rare, but safeguard)
                y_pred_proba_full = np.zeros((y_pred_proba_raw.shape[0], len(all_classes)))
                k = min(y_pred_proba_raw.shape[1], len(all_classes))
                y_pred_proba_full[:, :k] = y_pred_proba_raw[:, :k]

                # discrete prediction from probabilities
                y_discrete_prediction = np.argmax(y_pred_proba_full, axis=1)
                y_prediction = y_discrete_prediction
            if labelcol == 0:
                loss = log_loss(test_Y, y_prediction, labels=[0, 1])
            else:
                loss = log_loss(test_Y, y_pred_proba_full, labels=[0, 1, 2, 3, 4])
            # remember best model
            if loss < best_loss:
                if clf != "NN":
                    bestmodel = model
                else:
                    bestmodel = None
                # dynamically find all classes
                # all_classes = np.unique(np.concatenate([test_Y, y_discrete_prediction]))
                all_classes = np.array([0, 1, 2, 3, 4])  # your expected classes
                confusion = confusion_matrix(test_Y, y_discrete_prediction, labels=all_classes)
                results = {
                    "loss": loss,
                    "accuracy": accuracy_score(test_Y, y_discrete_prediction),
                    "precision": precision_score(test_Y, y_discrete_prediction, average=None, zero_division=0,
                                                 labels=all_classes),
                    "recall": recall_score(test_Y, y_discrete_prediction, average=None, zero_division=0,
                                           labels=all_classes),
                    "f1": f1_score(test_Y, y_discrete_prediction, average=None, zero_division=0,
                                   labels=all_classes),
                    "confusion": confusion.tolist(),
                    "monotoneClasslabels": onlyOneClasslabel
                }

        else:
            raise Exception("wrong labelcol inputted !?")


    return results, bestmodel
