# Pipeline for generating a quick baseline model for a Kaggle Dataset

The code in this repository is designed to apply the most typical data transformation steps to a Kaggle dataset and generate a submission by adjusting the transformation steps mostly through a config file.

## Data Transformation

The data transformation steps are currently divided into three major functions within the main code:

* preprocess_data
* feature_engineer
* transform_data

### preprocess_data

This function focuses on steps related to data cleaning. It currently includes the following functionalities:

* Filling NA's with various values -- see more details below
* Converting categorical features to type 'category' within a pandas dataframe
* Converting certain features to type 'str'within a pandas dataframe
* Dropping certain columns/features completely

### feature_engineer

Under the assumption that there are no NA's anymore and that a certain number of columns were dropped, the user can generate any desirable features inside this function.
Since these features will not appear on the config file (unless the user simply transforms an existing feature rather than creating a new one), the user must make sure to do all the required transformations (such as normalization, taking logs, etc.)
The goal of this function is to implement steps that do not generalize from a Kaggle competition to another, so that will require the user to modify code for each use case.

### transform_data

This function takes of care of general transformation steps dictated by the config file. A few of the transformations included:

* taking the log (or log1p or boxcox1p) of a set of variables
* target-encoding a set of variables
* taking the log1p of the target variable
* Applying a robust scaler to numeric features (**remark: currently this is hardcoded; ideally this should come from the config file** )

Note: originally, this function also one-hot encoded certain variables. However, certain models do not require one-hot encoding (for example, lightGBM), so we opted for allowing one-hot encoding explicility during the definition of a model in training steps. The variables to one-hot encode can still be decided through the config file, but the one-hot encoder must be explicitly added with a scikit-learn pipeline

## Training and Creating a Submission from the Test Set

This portion of the code contains the actual training and submission generation.
It is not currently modularized, but we may consider that in the future. Here are the major steps within the code:

* Apply the three functions described above on the training data
* A set of models is defined within the code. Currently, this step is not coming from the config file, but we may consider that in the future
* A set of models is appended to a model list
* Fit and cross-validation is done to all the models on the model list
* Test set goes through the same functions as the training set
* Models within the model list are called to generate predictions, which are averaged over to produce a single submission (**remark: within ml_utils.py, a class for averaging that does not require a list is already included**)
* Predictions are saved on the submissions folder

## Config File

The config file, config.yaml, is where the user specifies the variables for the transformations discussed above. It can be thought of as a dictionary, where the key is associated with some transformation, and the value (or list of variables) usually contains variables that will go through that transformation.

Here is the current config file structure, and what each part does:

### general

Here the user gets to specify some general parameters

* **target_variable**: string should contain the name of the target variable in the traning and test sets.
* **categorical_variables**: list of variables that will be explicitly treated as type 'category'. This is useful for models such as LightGBM

### preprocess

Here are parameters associated with preprocessing the training and test sets

* **drop_cols**: list of features that will be dropped altogether
* **fill_most_frequent**: list of features whose NA's will be filled with the most common observation for the feature
* **fill_custom**: list of objects, where the first object field, "value", is the value to fill a NA entry; and the second object field, "cols", is the list of features associated with that custom value; multiple objects can be passed
* **fill_median_groupby** : Much like the parameter above, this parameter supports objects with the keys "groupby_cols" and "cols"; the first key specifies which columns to group by when computing the median; the second key specifies the columns to be filled with the medians
* **type_str_cols**: takes a list of features, and enforces them to be type str; this could be useful for a function that demands their arguments to be a string

More parameters will be implemented as the need arises.

### transform

Here are the parameters associated with the variable transformation steps

* **log_cols**: list of features that will be replaced with their log
* **log1p_cols**: list of features that will be replaced with their log1p
* **boxcox1p_cols**: list of features that will be replaced with their boxcox1p; this function has a parameter lambda that is currently hardcoded in the main pipeline; in the future, this value should also come from the config file, perhaps allowing for different lambdas for different sets of variables
* **onehot_cols**: list of features that can be given to a one-hot encoder when a one-hot encoder is made part of a model pipeline; notice that merely giving the features to this parameter is not enough to one-hot encode them; we opted for that approach for the moment because not every model requires one-hot encoding
* **targetencode_cols**: list of features to be encoded using target encoding
* **log_target**: can be set to True or False; if True, it will convert the target variable to its log1p; we should modify this to allow in the future for different choices

## Build virtual environment

First, check if you have the conda package manager by invoking `conda` in the command line. If it is not installed, download the version of miniconda appropriate for your operating system [here](https://docs.conda.io/en/latest/miniconda.html).

Once installed, `cd` in the repository and build the virtual environment using the dependencies listed in `environment.yaml` by running

```shell

conda env create -f environment.yaml
```

Activate the environment by

```shell
conda activate ml
```

## Challenges

To be included later.

## TODO

* ???
