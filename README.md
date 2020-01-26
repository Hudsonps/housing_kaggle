# Pipeline for ML Model Development

## Build virtual environment

First, check if you have the conda package manager by invoking `conda` in the command line. If it is not installed, download the version of miniconda appropriate for your operating system [here](https://docs.conda.io/en/latest/miniconda.html).

Once installed, `cd` in the repository and build the virtual environment using the dependencies listed in `environment.yaml` by running

```
conda env create -f environment.yaml
```

Activate the environment by

```
conda activate ml
```

## !!! Ideas

Thoughts on how to split procedures:

Have two separate scripts that will prepare the features and the labels

prepare_features.py --> features.csv
prepare_label.py    --> labels.csv

Using features.csv and labels.csv, have a script that merges these two files and trains a model

train.py --> model\_<RUN DATE>\_<PRIMARY PARAMS>

With the model produced by train.py, generate predictions on test set

gen_predictions.py --> predictions.csv

## TODO
* ???
