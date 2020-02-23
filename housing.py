import gc
import os
import random
import pathlib
import argparse

import lightgbm as lgb
import xgboost as xgb
import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype

from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder, RobustScaler
import category_encoders as ce
import yaml
from scipy.special import boxcox1p
from scipy.stats import zscore
from scipy import stats
from scipy.stats import norm, skew
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.model_selection import KFold, cross_validate
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import ElasticNet, ElasticNetCV, Lasso, LassoCV
from sklearn.linear_model import RidgeCV, LinearRegression
from sklearn.pipeline import make_pipeline
#from sklearn.model_selection import GridSearchCV

from ml_utils import StackingAveragedModels, AveragingModels
from ml_utils import CrossValidateAveragingModels


def load_yaml(yaml_file: pathlib.Path):
    with open(yaml_file, 'r') as stream:
        try:
            return yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)


def preprocess_data(X, y=None, test=False):

    # drop whatever cols are deemed irrelevant
    # TODO: read drop_cols from config file

    config = load_yaml("./config.yaml")
    drop_cols = config["preprocess"]["drop_cols"]
    fill_custom = config["preprocess"]["fill_custom"]
    fill_most_frequent_cols = config["preprocess"]["fill_most_frequent"]
    fill_median_groupby = config["preprocess"]["fill_median_groupby"]
    type_str_cols = config["preprocess"]["type_str_cols"]
    categorical_features = config["general"]["categorical_variables"]

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

    # columns whose type is to be converted to str
    # TODO: add possibility of converting types to config file
    if type_str_cols:
        for col in type_str_cols:
            X[col] = X[col].apply(str)

    for col in categorical_features:
        X[col] = X[col].astype('category')

    #######
    for col in drop_cols:
        X.drop(col, axis=1, inplace=True)

    # TODO: Include a check for whether there are still missing values
    for col in X.columns:
        if any(X[col].isna()):
            print("There are still NA's in column " + str(col))
            return -1

    if test:
        return X
    else:
        return X, y


def feature_engineer(X, y=None, test=False):
    """
    This function needs to be adjusted for every use case
    It contains steps that cannot be generalized with a simple config file
    """

    config = load_yaml("./config.yaml")
    onehot_cols = config["transform"]["onehot_cols"]

    # removing outliers
    if not test:
        X = X[X['GrLivArea'] < 4500]
        y = y[X.index]

    X['TotalSF'] = X['TotalBsmtSF'] + X['1stFlrSF'] + X['2ndFlrSF']
    for col in ['TotalBsmtSF', '1stFlrSF', '2ndFlrSF']:
        X.drop(col, axis=1, inplace=True)
    X['TotalSF'] = np.log1p(X['TotalSF'])

    X['TotalPorch'] = X['OpenPorchSF'] + X['EnclosedPorch']
    for col in ['OpenPorchSF', 'EnclosedPorch']:
        X.drop(col, axis=1, inplace=True)
    X['TotalPorch'] = np.log1p(X['TotalPorch'])

    X['TotalBath'] = X['FullBath'] + X['BsmtFullBath'] +\
        0.5*(X['BsmtHalfBath'] + X['HalfBath'])
    for col in ['FullBath', 'BsmtFullBath', 'BsmtHalfBath', 'HalfBath']:
        X.drop(col, axis=1, inplace=True)
    X['TotalBath'] = np.log1p(X['TotalBath'])

    if not test:
        lr = make_pipeline(ce.OneHotEncoder(
                cols=onehot_cols,
                use_cat_names=True), LinearRegression())
        resids = y - lr.fit(X, y).predict(X)
        outliers = resids[np.abs(zscore(resids)) > 2.5].index
        X.drop(outliers, inplace=True)
        y = y[X.index]

    if test:
        return X
    else:
        return X, y


def transform_data(X, y=None, test=False):
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
    boxcox1p_cols = config["transform"]["boxcox1p_cols"]
    onehot_cols = config["transform"]["onehot_cols"]
    targetencode_cols = config["transform"]["targetencode_cols"]
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

    if boxcox1p_cols:
        for col in boxcox1p_cols:
            if col in columns:
                print("taking the log of "+str(col))
                # this will replace the columns with their boxcox1p values
                X[col] = boxcox1p(X[col], 0.15)

    # robust scaler
    numeric_cols = X.select_dtypes(include=np.number).columns.tolist()
    if not test:
        global robust_scaler
        robust_scaler = RobustScaler()
        robust_scaler.fit_transform(X[numeric_cols])
    else:
        robust_scaler.transform(X[numeric_cols])


    # transforming target
    if log_target and not test:
        y = np.log1p(y)

    # target encoding
    if targetencode_cols:
        if not test:
            global target_encoder
            target_encoder = ce.TargetEncoder(cols=targetencode_cols)
            X = target_encoder.fit_transform(X, y)
        else:
            X = target_encoder.transform(X)

    if test:
        return X
    else:
        return X, y


if __name__ == '__main__':

    MAIN = pathlib.Path('./')
    SUBMISSIONS_PATH = MAIN / 'submissions'
    submission_name = 'submission.csv'
    random.seed(0)

    config = load_yaml("./config.yaml")

    onehot_cols = config["transform"]["onehot_cols"]

    categorical_features = config["general"]["categorical_variables"]

    target = config["general"]["target_variable"]
    #############
    # Load Data #
    #############
    print("Loading data...")
    train = pd.read_csv(MAIN / 'data' / 'train.csv')

    print(train.shape)
    print(train.columns)
    y_train = train[target]
    X_train = train.drop([target], axis=1)


    # TODO: reimplement reduce memory usage

    #######################
    # Reduce Memory Usage #
    #######################

    #########################
    # Prepare Training Data #
    #########################
    X_train, y_train = preprocess_data(X_train, y_train, test=False)
    X_train, y_train = feature_engineer(X_train, y_train, test=False)
    X_train, y_train = transform_data(X_train, y_train, test=False)

    #X_train.reset_index(drop=True, inplace=True)
    #y_train.reset_index(drop=True, inplace=True)

    def rmsle_cv(model):
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        score = np.sqrt(-cross_val_score(
            model,
            X_train,
            y_train,
            scoring="neg_mean_squared_error",
            cv=kf)
            )
        print("\nValidation score: {:.4f} ({:.4f})\n".format(score.mean(),
                                                        score.std()))
        return(score)

    model_list = []

    model_lgb = lgb.LGBMRegressor(
       objective='regression',
       num_leaves=5,
       learning_rate=0.05,
       n_estimators=400,
       feature_fraction=0.2)
      # feature_fraction_seed=9,
      # bagging_seed=9,
      # min_data_in_leaf=6,
      # min_sum_hessian_in_leaf=11)

    model_xgb = make_pipeline(ce.OneHotEncoder(
        cols=onehot_cols,
        use_cat_names=True),
        xgb.XGBRegressor(colsample_bytree=0.4603, gamma=0.0468,
                             learning_rate=0.05, max_depth=3,
                             min_child_weight=1.7817, n_estimators=2200,
                             reg_alpha=0.4640, reg_lambda=0.8571,
                             subsample=0.5213, silent=1,
                             random_state =7, nthread = -1))

    GBoost = make_pipeline(ce.OneHotEncoder(
        cols=onehot_cols,
        use_cat_names=True),
        GradientBoostingRegressor(n_estimators=1000, learning_rate=0.05,
                                   max_depth=4, max_features='sqrt',
                                   min_samples_leaf=15, min_samples_split=10,
                                   loss='huber', random_state =5))

    linear = make_pipeline(ce.OneHotEncoder(
                cols=onehot_cols,
                use_cat_names=True), LinearRegression())

    lasso = make_pipeline(ce.OneHotEncoder(
                cols=onehot_cols,
                use_cat_names=True),
                Lasso(alpha=0.0005, random_state=1))

    Enet = make_pipeline(ce.OneHotEncoder(
                cols=onehot_cols,
                use_cat_names=True),
                ElasticNet(alpha=0.0005, l1_ratio=.5, random_state=3))

    RidgeReg = make_pipeline(ce.OneHotEncoder(
                cols=onehot_cols,
                use_cat_names=True), RidgeCV())

    ensemble = AveragingModels(
        models=(RidgeReg, Enet, lasso, model_lgb, GBoost))

    cross_validate_ensemble = CrossValidateAveragingModels(
        models=[RidgeReg, Enet, lasso, model_lgb, GBoost]
    )

    stacked = StackingAveragedModels(
        base_models=(Enet, lasso, model_lgb, GBoost, RidgeReg),
        meta_model=Lasso(alpha=0.0005, random_state=1)
    )

    #model_list.append(model_lgb)
   #model_list.append(lasso)
   # model_list.append(Enet)
   # model_list.append(RidgeReg)

    #model_list.append(model_lgb)

    model_list.append(ensemble)

    for model in model_list:
        print("A NEW MODEL")

        model.fit(X_train, y_train)
       # rmsle_cv(model)

    #params = {
     #   "objective": "regression",
     #   "boosting": "gbdt",
     #   "num_leaves": 40,
     #   "learning_rate": 0.05,
     #   "feature_fraction": 0.85,
     #   "reg_lambda": 3,
     #   "metric": "rmse"
    #}

    #####################
    # Prepare Test Data #
    #####################
    print(
        "\n==================\n",
        "Loading test set...",
        sep='\n'
    )

    X_test = pd.read_csv(MAIN / 'data' / 'test.csv')

    # TODO: Call memory usage on test set

    # TODO: think of a better way to track the test ID's
    # without having to hardcore the name
    test_ids = X_test['Id']
    X_test = preprocess_data(X_test, test=True)
    X_test = feature_engineer(X_test, test=True)
    X_test = transform_data(X_test, test=True)

    ######################
    # Prepare Submission #
    ######################
    print(
        "\n==================\n",
        "Generating predictions on test set...",
        sep='\n'
    )

    n_models = len(model_list)
    final_pred = np.zeros(X_test.shape[0])
    for model in model_list:
        raw_pred = model.predict(X_test)
        log_target = config["transform"]["log_target"]
        if log_target:
            pred = np.expm1(raw_pred)
        else:
            pred = raw_pred

        final_pred += pred/n_models
    #del model_lgb
    #gc.collect()

    print("Saving predictions as csv...")
    submission = pd.DataFrame(
            {"Id": test_ids, "SalePrice": np.clip(pred, 0, a_max=None)}
        )
    submission.to_csv(
            SUBMISSIONS_PATH / (submission_name), index=False
        )
