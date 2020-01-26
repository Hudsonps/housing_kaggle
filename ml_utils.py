'''General purpose utilities
'''
import pathlib
import subprocess
import json
import datetime
from dateutil.relativedelta import relativedelta
from typing import Dict, List
import re

import pandas as pd
from pandas.api.types import is_datetime64_any_dtype as is_datetime
from pandas.api.types import is_categorical_dtype
import numpy as np
import yaml


def print_full(df, num_rows=100):
    '''Print the first num_rows rows of dataframe in full

    Resets display options back to default after printing
    '''
    pd.set_option('display.max_rows', len(df))
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 2000)
    pd.set_option('display.float_format', '{:20,.2f}'.format)
    pd.set_option('display.max_colwidth', -1)
    display(df.iloc[0:num_rows])
    pd.reset_option('display.max_rows')
    pd.reset_option('display.max_columns')
    pd.reset_option('display.width')
    pd.reset_option('display.float_format')
    pd.reset_option('display.max_colwidth')


def reduce_mem_usage(df: pd.DataFrame,
                     cols_exclude: List[str] = []) -> pd.DataFrame:
    '''Iterate through all the columns of a dataframe and modify
    the data type to reduce memory usage.

    Original code from
    https://www.kaggle.com/gemartin/load-data-reduce-memory-usage by @gemartin
    '''

    start_mem = df.memory_usage().sum() / 1024**2

    cols = [c for c in df.columns if c not in cols_exclude]
    print(
        "Reducing memory for the following columns: ",
        cols,
        sep='\n'
    )

    for col in cols:
        if is_datetime(df[col]) or is_categorical_dtype(df[col]):
            continue

        print(f"Reducing memory for {col}")
        col_type = df[col].dtype

        if col_type != object:

            c_min = df[col].min()
            c_max = df[col].max()

            if str(col_type)[:3] == "int":
                if c_min > np.iinfo(np.int8).min \
                        and c_max < np.iinfo(np.int8).max:

                    df[col] = df[col].astype(np.int8)

                elif c_min > np.iinfo(np.int16).min \
                        and c_max < np.iinfo(np.int16).max:

                    df[col] = df[col].astype(np.int16)

                elif c_min > np.iinfo(np.int32).min \
                        and c_max < np.iinfo(np.int32).max:

                    df[col] = df[col].astype(np.int32)

                elif c_min > np.iinfo(np.int64).min \
                        and c_max < np.iinfo(np.int64).max:

                    df[col] = df[col].astype(np.int64)

            else:
                df[col] = df[col].astype(np.float32)
        else:
            df[col] = df[col].astype("category")

    end_mem = df.memory_usage().sum() / 1024**2
    print(
        f"Memory usage before: {start_mem:.2f} MB",
        f"Memory usage after: {end_mem:.2f} MB "
        f"({100 * (start_mem - end_mem) / start_mem:.1f}% decrease)",
        sep='\n'
    )

    return df


def add_datepart(df: pd.DataFrame,
                 fldnames: List[str],
                 datetimeformat: str,
                 drop: bool = True,
                 time: bool = False,
                 errors: str = "raise") -> None:
    '''add_datepart converts a column of df from a datetime64 to
    many columns containing the information from the date. This applies
    changes inplace.
    Parameters:
    -----------
    df: A pandas data frame. df gain several new columns.
    fldname: A string or list of strings that is the name of the date column
    you wish to expand. If it is not a datetime64 series, it will be
    converted to one with pd.to_datetime.

    drop: If true then the original date column will be removed.
    time: If true time features: Hour, Minute, Second will be added.

    Original code taken from FastAI library
    '''
    if isinstance(fldnames, str):
        fldnames = [fldnames]
    for fldname in fldnames:
        fld = df[fldname]
        fld_dtype = fld.dtype
        if isinstance(fld_dtype, pd.core.dtypes.dtypes.DatetimeTZDtype):
            fld_dtype = np.datetime64

        if not np.issubdtype(fld_dtype, np.datetime64):
            df[fldname + '_orig'] = df[fldname].copy()
            df[fldname] = fld = pd.to_datetime(
                fld, format=datetimeformat, errors=errors)
        targ_pre = re.sub('[Dd]ate$', '', fldname)
        attr = ['Year', 'Month', 'Week', 'Day', 'Dayofweek', 'Dayofyear',
                'Is_month_end', 'Is_month_start', 'Is_quarter_end',
                'Is_quarter_start', 'Is_year_end', 'Is_year_start']
        if time:
            attr = attr + ['Hour', 'Minute', 'Second']
        for n in attr:
            df[targ_pre + n] = getattr(fld.dt, n.lower())
        df[targ_pre + 'Elapsed'] = fld.astype(np.int64) // 10 ** 9
        if drop:
            df.drop(fldname + '_orig', axis=1, inplace=True)
    return None
