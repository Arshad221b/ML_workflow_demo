import os

import joblib
import pandas as pd
from sklearn import metrics
from sklearn import tree

import config


def run(fold):
    df = pd.read_csv(config.TRAINING_FILE)
    df_train = df[df.kfold != fold].reset_index(drop=True)
    df_valid = df[df.kfold == fold].reset_index(drop=True)

    x_train = df_train.drop(config.LABEL, axis=1).values
    y_train = df_train[config.LABEL].values

    x_valid = df_valid.drop(config.LABEL, axis=1).values
    y_valid = df_valid[config.LABEL].values

    clf = tree.DecisionTreeClassifier()

    clf.fit(x_train, y_train)
    preds = clf.predict(x_valid)

    accuracy = metrics.accuracy_score(y_valid, preds)
    print(f"FOLD = {fold}, Accuracy= {accuracy}")
    joblib.dump(clf,
                os.path.join(config.MODEL_OUTPUT, f"dt_{fold}.bin")
                )


if __name__ == "__main__":
    run(fold=0)
