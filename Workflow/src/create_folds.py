import numpy as np
import pandas as pd
from sklearn import model_selection

import config


def create_folds(data, num_splits):
    data["kfold"] = -1
    data = data.sample(frac=1).reset_index(drop=True)
    num_bins = int(np.floor(1 + np.log2(len(data))))

    data.loc[:, 'bins'] = pd.cut(
        data[config.LABEL], bins=num_bins, labels=False
    )
    kf = model_selection.StratifiedKFold(n_splits=num_splits)

    for f, (t_, v_) in enumerate(kf.split(X=data, y=data.bins.values)):
        data.loc[v_, 'kfold'] = f

    data = data.drop("bins", axis=1)
    return data


if __name__ == '__main__':
    data = pd.read_csv(config.INPUT_TRAINING_FILE)
    num_splits = 5
    output_data = create_folds(data, num_splits)
    output_data.to_csv(config.TRAINING_FILE)
