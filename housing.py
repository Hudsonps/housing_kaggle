import gc
import os
import random
import pathlib
import argparse

import lightgbm as lgb
import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype

from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder
import category_encoders as ce
import yaml
from scipy.special import boxcox1p
from scipy import stats
from scipy.stats import norm, skew


def load_yaml(yaml_file: pathlib.Path):
    with open(yaml_file, 'r') as stream:
        try:
            return yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)


def preprocess_data(X, test=False):

    # drop whatever cols are deemed irrelevant
    # TODO: read drop_cols from config file

    config = load_yaml("./config.yaml")

    target = config["general"]["target_variable"]

    drop_cols = config["preprocess"]["drop_cols"]
    fill_custom = config["preprocess"]["fill_custom"]
    fill_most_frequent_cols = config["preprocess"]["fill_most_frequent"]
    fill_median_groupby = config["preprocess"]["fill_median_groupby"]
    type_str_cols = config["preprocess"]["type_str_cols"]
    categorical_features = config["general"]["categorical_variables"]

    #######
    for col in drop_cols:
        X.drop(col, axis=1, inplace=True)

    for item in fill_custom:
        value = item["value"]
        custom_cols = item["cols"]
        for col in custom_cols:
            X[col] = X[col].fillna(value)

    # TODO: fillna median without groupby
    # TODO: mean without groupby
    # TODO: mean with groupby

    for item in fill_median_groupby:
        groupby_cols = item["groupby_cols"]
        filled_cols = item["cols"]
        for col in filled_cols:
            X["LotFrontage"] = X.groupby(groupby_cols)[col].transform(
                lambda x: x.fillna(x.median()))

    for col in fill_most_frequent_cols:
        X[col] = X[col].fillna(X[col].mode()[0])


    # TODO: Check if target variable also has problems
    # but only when test=False
    # target name should come from config json.

    # columns whose type is to be converted to str
    # TODO: add possibility of converting types to config file
    if type_str_cols:
        for col in type_str_cols:
            X[col] = X[col].apply(str)

    for col in categorical_features:
        X[col] = X[col].astype('category')

    # TODO: Include a check for whether there are still missing values
    for col in X.columns:
        if any(X[col].isna()):
            print("There are still NA's in column " + str(col))
            return -1

    return X


def feature_engineer(X, test=False):
    """
    This function needs to be adjusted for every use case
    It contains steps that cannot be generalized with a simple config file
    """
    X['TotalSF'] = X['TotalBsmtSF'] + X['1stFlrSF'] + X['2ndFlrSF']
    #for col in ['TotalBsmtSF', '1stFlrSF', '2ndFlrSF']:
     #   X.drop(col, axis=1, inplace=True)

    # since the number of skewed features is very large,
    # we will not use the config file at first to take the log of variables
   # numeric_feats = []
   # for col in X.columns:
    #    if is_numeric_dtype(X[col]):
     #       numeric_feats.append(col)
    #skewed_feats = X[numeric_feats].apply(lambda x: skew(x.dropna()))\
    #    .sort_values(ascending=False)
    #skewness = pd.DataFrame({'Skew': skewed_feats})
    #skewness = skewness[abs(skewness) > 0.75]

    #print(skewness)

    #skewed_features = skewness.index
    #lam = 0.15
    #for feat in skewed_features:
     #   X[feat] = boxcox1p(X[feat], lam)

    return X


def transform_data(X, test=False):
    """
    Preparing final dataset with all features.

    Arguments
    ---
    X - dataframe with preprocessed features and target variable
    test - boolean; if false, it means X is the training set
           If true, it means X is the test set

    """
    config = load_yaml("./config.yaml")

    columns = list(X.columns)

    log_cols = config["transform"]["log_cols"]
    log1p_cols = config["transform"]["log1p_cols"]
    onehot_cols = config["transform"]["onehot_cols"]
    target = config["general"]["target_variable"]
    log_target = config["transform"]["log_target"]

    # generate time features (only relevant for time series)
    # TODO: make datetime column identifiable from config file
    if "timestamp" in columns:
        X.timestamp = pd.to_datetime(X.timestamp,
                                    format="%Y-%m-%d %H:%M:%S")
        # adjust the desirable format accordingly
        X["hour"] = X.timestamp.dt.hour
        X["weekday"] = X.timestamp.dt.weekday

        if not test:
            X.sort_values("timestamp", inplace=True)
            X.reset_index(drop=True, inplace=True)

    # TODO: make cols identified from config file

    if log_cols:
        for col in log_cols:
            # this will replace the columns with their log values
            X[col] = np.log(X[col])

    if log1p_cols:
        for col in log1p_cols:
            # this will replace the columns with their log1p values
            X[col] = np.log1p(X[col])

    # one-hot encoding
    onehot_cols = []  # temporarily overwriting the config file
    if onehot_cols:
        if not test:
            # onehot_encoder must be global so it can be used
            # again on the test test
            global onehot_encoder
            onehot_encoder = ce.OneHotEncoder(
                cols=onehot_cols,
                use_cat_names=True,
                handle_unknown=ignore)
            X = onehot_encoder.fit_transform(X)
        else:
            X = onehot_encoder.transform(X)

    if test:
        return X
    else:
        if log_target:
            y = np.log1p(X[target])
        else:
            y = X[target]
        X.drop(target, axis=1, inplace=True)
        return X, y


if __name__ == '__main__':

    MAIN = pathlib.Path('./')
    SUBMISSIONS_PATH = MAIN / 'submissions'
    submission_name = 'submission.csv'
    random.seed(0)

    config = load_yaml("./config.yaml")

    #############
    # Load Data #
    #############
    print("Loading data...")
    train = pd.read_csv(MAIN / 'data' / 'train.csv')

    print(train.shape)

    # TODO: reimplement reduce memory usage

    #######################
    # Reduce Memory Usage #
    #######################

    #########################
    # Prepare Training Data #
    #########################
    train = preprocess_data(train, test=False)
    train = feature_engineer(train, test=False)
    X_train, y_train = transform_data(train, test=False)

    #####################
    # Two-Fold LightGBM #
    #####################
    print("\n==================\n")
    X_half_1 = X_train[:int(X_train.shape[0] / 2)]
    X_half_2 = X_train[int(X_train.shape[0] / 2):]

    y_half_1 = y_train[:int(X_train.shape[0] / 2)]
    y_half_2 = y_train[int(X_train.shape[0] / 2):]

    # TODO: Read categorical features from config file
    categorical_features = config["general"]["categorical_variables"]

    d_half_1 = lgb.Dataset(
        X_half_1,
        label=y_half_1,
        categorical_feature=categorical_features,
        free_raw_data=False
    )
    d_half_2 = lgb.Dataset(
        X_half_2,
        label=y_half_2,
        categorical_feature=categorical_features,
        free_raw_data=False
    )

    # Include both datasets in watchlists
    # to get both training and validation loss
    watchlist_1 = [d_half_1, d_half_2]
    watchlist_2 = [d_half_2, d_half_1]

    params = {
        "objective": "regression",
        "boosting": "gbdt",
        "num_leaves": 40,
        "learning_rate": 0.05,
        "feature_fraction": 0.85,
        "reg_lambda": 3,
        "metric": "rmse"
    }

    print("Building model with first half and validating on second half:")
    model_half_1 = lgb.train(
        params,
        train_set=d_half_1,
        num_boost_round=1000,
        valid_sets=watchlist_1,
        verbose_eval=200,
        early_stopping_rounds=200
    )

    print("Building model with second half and validating on first half:")
    model_half_2 = lgb.train(
        params,
        train_set=d_half_2,
        num_boost_round=1000,
        valid_sets=watchlist_2,
        verbose_eval=200,
        early_stopping_rounds=200
    )

    #####################
    # Prepare Test Data #
    #####################
    print(
        "\n==================\n",
        "Loading test set...",
        sep='\n'
    )

    test = pd.read_csv(MAIN / 'data' / 'test.csv')

    # TODO: Call memory usage on test set

    # TODO: think of a better way to track the test ID's
    # without having to hardcore the name
    test_ids = test['Id']
    test = preprocess_data(test, test=True)
    test = feature_engineer(test, test=True)
    X_test = transform_data(test, test=True)

    ######################
    # Prepare Submission #
    ######################
    print(
        "\n==================\n",
        "Generating predicitons on test set...",
        sep='\n'
    )
    raw_pred_1 = model_half_1.predict(
        X_test,
        num_iteration=model_half_1.best_iteration
    )

    log_target = config["transform"]["log_target"]
    if log_target:
        pred = np.expm1(raw_pred_1) / 2
    else:
        pred = raw_pred_1 / 2
    del model_half_1
    gc.collect()

    raw_pred_2 = model_half_2.predict(
        X_test,
        num_iteration=model_half_2.best_iteration
    )
    # TODO: Same as above
    if log_target:
        pred += np.expm1(raw_pred_2) / 2
    else:
        pred += raw_pred_2 / 2
    del model_half_2
    gc.collect()

    print("Saving predictions as csv...")
    submission = pd.DataFrame(
            {"Id": test_ids, "SalePrice": np.clip(pred, 0, a_max=None)}
        )
    submission.to_csv(
            SUBMISSIONS_PATH / (submission_name), index=False
        )