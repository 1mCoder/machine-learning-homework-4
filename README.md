# Machine Learning Homework 4

This repository contains code and notebooks for a text classification project focusing on toxic comment classification.

## Overview

This project focuses on classifying toxic comments using machine learning techniques. The repository includes exploratory data analysis (EDA) notebooks, code for data preprocessing, model training, evaluation, and a testing suite.

## Project Structure

The repository structure is organized as follows:

- **.github**: Contains GitHub Actions workflows for CI/CD.
- **EDA**: Contains an exploratory data analysis notebook (`EDA.ipynb`) exploring the dataset.
- **kmaml223**: Contains dataset files (`sample_submission.csv`, `test.csv`, `train.csv`).
- **models**: Contains trained models structured by different approaches (`baseline`, `logistic_regression_accuracy`, `logistic_regression_f1`).
- **src**: Contains the project's source code divided into subdirectories:
  - **code**: Holds Python scripts for different functionalities (`baseline.py`, `clean_text.py`, `data.py`, `logreg_acc.py`, `logreg_f1.py`, `main.py`, `make_submission.py`).
    - `main.py`: Contains the Inference Script that utilizes selected models to make predictions for any input.
  - **notebooks**: Holds Jupyter notebooks for different project stages (`text_cleanup.ipynb`, `baseline.ipynb`, `logistic_regression_accuracy.ipynb`, `logistic_regression_f1.ipynb`, `make_submission.ipynb`).
- **tests**: Contains unit tests structured by functionalities.
  - **code**: Holds test scripts (`conftest.py`, `test_clean_text.py`, `test_data.py`, `test_make_submission.py`, `test_models.py`).
  - **data**: Holds data-related tests (`conftest.py`, `test_test_dataset.py`, `test_train_dataset.py`).
- **.gitignore**: Specifies intentionally untracked files to ignore.
- **requirements.txt**: Lists required Python libraries and their versions.

The `main.py` file within the `src/code` directory serves as the Inference Script. It leverages the selected trained models from the `models` directory to make predictions for any input data.


## Notebooks

The `src/notebooks` directory contains Jupyter notebooks detailing different stages of the project, including data cleaning, model training, and result analysis.

## Code

The `src/code` directory contains Python scripts responsible for various project functionalities. These scripts handle baseline models, data cleaning, data processing, different model approaches (`logreg_acc.py`, `logreg_f1.py`), main execution, and submission generation.

## Models

The `models` directory stores trained models categorized by different approaches (`baseline`, `logistic_regression_accuracy`, `logistic_regression_f1`). Each subdirectory contains trained models for toxic comment classification.

## Tests

The `tests` directory contains unit tests for different functionalities (`code`) and data-related tests (`data`). These tests ensure the correctness of code functionalities and data preprocessing.
