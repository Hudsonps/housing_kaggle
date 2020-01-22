import gc
import os
import random
import pathlib
import argparse

import lightgbm as lgb
import numpy as np
import pandas as pd
import seaborn as sns

from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder

from utils import *

def preprocess_data(X, test=False):

    # drop whatever cols are deemed irrelevant
    # TODO: read drop_cols from config file
    drop_cols = [
        "Utilities"
    ]
    for col in drop_cols:
        X.drop(col, axis=1, inplace=True)

    # fillna values
    # I don't see a way of making this part automatic
    # Since what to fill and how may change
    # TODO: Consider options for automation

    fill_none = [
        "PoolQC",
        "MisFeature",
        "Alley",
        "Fence",
        "FireplaceQu",
        "GarageType",
        "GarageFinish",
        "GarageQual",
        "GarageCond",
        "BsmtQual",
        "BsmtCond",
        "BsmtExposure",
        "BsmtFinType1",
        "BsmtFinType2",
        "MasVnrType",
        "MSSubClass"
    ]
    for col in fill_none:
        X[col] = X[col].fillna("None")

    # fill with median
    # TODO: think about how to automate this step
    # It is harder because there is a groupby involved...
    X["LotFrontage"] = X.groupby("Neighborhood")["LotFrontage"].transform(
        lambda x: x.fillna(x.median()))

    fill_zero = [
        "GarageYrBlt",
        "GarageArea",
        "GarageCars",
        "BsmtFinSF1",
        "BsmtFinSF2",
        "BsmtUnfSF",
        "TotalBsmtSF",
        "BsmtFullBath",
        "BsmtHalfBath",
        "MasVnrArea"
    ]
    for col in fill_zero:
        X[col] = X[col].fillna(0)

    # TODO: generalize the lines below
    X['MSZoning'] = X['MSZoning'].fillna(X['MSZoning'].mode()[0])
    X['Electrical'] = X['Electrical'].fillna(
        X['Electrical'].mode()[0])
    X['KitchenQual'] = X['KitchenQual'].fillna(X['KitchenQual'].mode()[0])
    X['Exterior1st'] = X['Exterior1st'].fillna(X['Exterior1st'].mode()[0])
    X['Exterior2nd'] = X['Exterior2nd'].fillna(X['Exterior2nd'].mode()[0])
    X['SaleType'] = X['SaleType'].fillna(X['SaleType'].mode()[0])

    # TODO: give config json the possibility of custom fillna
    X["Functional"] = X["Functional"].fillna("Typ")

    # TODO: Include a check for whether there are still missing values

    # TODO: Check if target variable also has problems
    # but only when test=False
    # target name should come from config json.

    return X




def transform_data(X, test=False):
    """
    Preparing final dataset with all features.
    """

    columns = list(X.columns)

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

    # TODO: read the target name from config
    target = "target_variable_name"

    # TODO: make cols identified from config file
    log_cols = []
    log1p_cols = []

    # if true, we will take the log of the target
    # TODO: let's read that from a config file later
    log_target = True

    for col in log_cols:
        # this will replace the columns with their log values
        X[col] = np.log(X[col])

    for col in log_cols:
    # this will replace the columns with their log values
        X[col] = np.log1p(X[col])

    if test:
        return X
    else:
        if log_target:
            y = np.log1p(X[target])
        else:
            y = X[target]
        X.drop(target, axis=1, inplace=True)
        return X, y


parser = argparse.ArgumentParser(
    description="Generate aggregate features")
parser.add_argument(
    '-s', '--sample',
    action='store_true',
    help='Run code on a select subset of ruts '
         'from main dataset for prototyping'
)
parser.add_argument(
    '-sn', '--submission_name',
    type=pathlib.Path,
    required=True,
    help='Name of submission',
)

if __name__ == '__main__':

    MAIN = pathlib.Path('/Users/palermopenano/personal/kaggle_energy')
    SUBMISSIONS_PATH = MAIN / 'submissions'
    p = parser.parse_args()
    sample = p.sample
    submission_name = p.submission_name
    random.seed(0)

    #############
    # Load Data #
    #############
    print("Loading data...")
    df_train = pd.read_csv(MAIN / 'data' / 'train.csv')
    building = pd.read_csv(MAIN / 'data' / 'building_metadata.csv')
    le = LabelEncoder()
    building.primary_use = le.fit_transform(building.primary_use)
    weather_train = pd.read_csv(MAIN / 'data' / 'weather_train.csv')

    # Take only a random sample of n buildings
    randbuilding = None
    if sample:
        print("Taking a random sample of buildings...")
        df_train, randbuilding = \
            df_sample_random_buildings(df_train, 'building_id', n=5)
        print(randbuilding)
    print(df_train.shape)

    #######################
    # Reduce Memory Usage #
    #######################
    print("Reducing memory usage...")
    df_train = reduce_mem_usage(
        df_train,
        use_float16=True,
        cols_exclude=['timestamp']
    )
    building = reduce_mem_usage(
        building,
        use_float16=True,
        cols_exclude=['timestamp']
    )
    weather_train = reduce_mem_usage(
        weather_train,
        use_float16=True,
        cols_exclude=['timestamp']
    )

    #########################
    # Prepare Training Data #
    #########################
    X_train, y_train = prepare_data(df_train, building, weather_train)
    del df_train, weather_train
    gc.collect()

    #####################
    # Two-Fold LightGBM #
    #####################
    print("\n==================\n")
    X_half_1 = X_train[:int(X_train.shape[0] / 2)]
    X_half_2 = X_train[int(X_train.shape[0] / 2):]

    y_half_1 = y_train[:int(X_train.shape[0] / 2)]
    y_half_2 = y_train[int(X_train.shape[0] / 2):]

    categorical_features = [
        "building_id", "site_id",
        "meter", "primary_use",
        "hour", "weekday"
    ]

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
        "reg_lambda": 2,
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
    df_test = pd.read_csv(MAIN / 'data' / 'test.csv')
    weather_test = pd.read_csv(MAIN / 'data' / 'weather_test.csv')

    df_test = reduce_mem_usage(
        df_test,
        use_float16=True,
        cols_exclude=['timestamp']
    )
    weather_test = reduce_mem_usage(
        weather_test,
        use_float16=True,
        cols_exclude=['timestamp']
    )

    if sample:
        df_test = df_test[df_test['building_id'].isin(randbuilding)]
        print("Shape of test data: ", df_test.shape)

    X_test, row_ids = prepare_data(df_test, building, weather_test, test=True)

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
    pred = np.expm1(raw_pred_1) / 2
    del model_half_1
    gc.collect()

    raw_pred_2 = model_half_2.predict(
        X_test,
        num_iteration=model_half_2.best_iteration
    )
    pred += np.expm1(raw_pred_2) / 2
    del model_half_2
    gc.collect()

    if not sample:
        print("Saving predictions as csv...")
        submission = pd.DataFrame(
            {"row_id": row_ids, "meter_reading": np.clip(pred, 0, a_max=None)}
        )
        submission.to_csv(
            SUBMISSIONS_PATH / (submission_name + '.csv'), index=False
        )

        print(
            submission.meter_reading.describe().apply(
                lambda x: format(x, ',.2f')
            )
        )
