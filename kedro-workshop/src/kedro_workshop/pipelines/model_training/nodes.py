import pandas as pd
from sklearn.model_selection import train_test_split
from typing import Tuple, List

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, MinMaxScaler
from sklearn import set_config

# Set transformer output to be a pandas DataFrame instead of numpy array
set_config(transform_output = "pandas")

from lightgbm import LGBMClassifier
from sklearn.pipeline import Pipeline
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)


def split_training_data(df: pd.DataFrame, test_size: float, val_size:float, random_state: int) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, pd.DataFrame]:
    """
    Split the data into training, validation, and test sets.

    Parameters:
    df (pd.DataFrame): Input DataFrame.
    test_size (float): Size of the test set.
    val_size (float): Size of the validation set.
    random_state (int): Random seed for reproducibility.

    Returns:
    X_train, y_train, X_val, y_val: Training and validation datasets.
    test_df : Test data to be saved for use later.
    """

    X = df.drop("Transported", axis=1)
    y = df["Transported"]

    X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=val_size, random_state=random_state)

    test_df = X_test
    test_df["Transported"] = y_test

    return X_train, y_train, X_val, y_val, test_df


def make_feature_pipeline(cat_columns : List[str], num_columns: List[str]) -> ColumnTransformer:
    """
    Create a feature engineering pipeline.

    Parameters:
    cat_columns (List): List of categorical column names.
    num_columns (List): List of numerical column names.

    Returns:
    ColumnTransformer: Feature engineering pipeline.
    """
    feature_engineering_pipeline = ColumnTransformer(
        transformers=[
            # (step name, transformer, column list to apply transformation to)
            ("onehot_encoding", OneHotEncoder(sparse_output=False), cat_columns),
            ("minmax_scaling", MinMaxScaler(), num_columns)
        ],
        remainder = "passthrough"
    )
    
    return feature_engineering_pipeline

def make_pipeline(cat_columns : List[str], num_columns: List[str]) -> Pipeline:
    """
    Create a pipeline with feature engineering and a LightGBM model.

    Parameters:
    cat_columns (List): List of categorical column names.
    num_columns (List): List of numerical column names.

    Returns:
    Pipeline: Model pipeline.
    """
    feature_engineering_pipeline = make_feature_pipeline(cat_columns, num_columns)
    
    model_pipeline = Pipeline(
        steps = [
            # (step name, transformer)
            ("feature_engineering", feature_engineering_pipeline),
            ("model", LGBMClassifier())
        ]
    )
    
    return model_pipeline


def objective(
    trial: optuna.Trial,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    cat_columns: List[str],
    num_columns: List[str]
):
    """
    Objective function for Optuna optimization.

    Parameters:
    trial (optuna.Trial): Optuna trial object.
    X_train (pd.DataFrame): Training features.
    y_train (pd.Series): Training target.
    X_val (pd.DataFrame): Validation features.
    y_val (pd.Series): Validation target.
    cat_columns (List): List of categorical column names.
    num_columns (List): List of numerical column names.

    Returns:
    float: Model score on the validation set.
    """
    params = {
        # meta parameters
        'model__objective': trial.suggest_categorical('model__objective', ['binary']),
        'model__metric': trial.suggest_categorical('model__metric', ['binary_logloss']),
        'model__boosting_type': trial.suggest_categorical('model__boosting_type', ['gbdt']),
        'model__verbosity': trial.suggest_categorical('model__verbosity', [-1]),
        'model__random_state': trial.suggest_categorical('model__random_state', [42]),
        
        # hyperparameters
        'model__num_leaves': trial.suggest_int('model__num_leaves', 10, 100),
        'model__learning_rate': trial.suggest_float('model__learning_rate', 0.01, 0.1, log=True),
        'model__feature_fraction': trial.suggest_float('model__feature_fraction', 0.1, 1.0),
        'model__bagging_fraction': trial.suggest_float('model__bagging_fraction', 0.1, 1.0),
        'model__bagging_freq': trial.suggest_int('model__bagging_freq', 1, 10),
        'model__min_child_samples': trial.suggest_int('model__min_child_samples', 1, 50),
        
    }
    
    # Create pipeline
    model_pipeline = make_pipeline(cat_columns, num_columns)
    
    # Set pipeline hyperparameters for trial
    model_pipeline = model_pipeline.set_params(**params)
    
    # Fit model and evaluate
    model_pipeline.fit(X_train, y_train)
    
    return model_pipeline.score(X_val, y_val)


def train_model(X_train: pd.DataFrame, y_train: pd.Series, X_val: pd.DataFrame, y_val:pd.Series, cat_columns: List[str], num_columns: List[str]) -> optuna.Study:
    """
    Train the model using Optuna for hyperparameter optimization.

    Parameters:
    X_train (pd.DataFrame): Training features.
    y_train (pd.Series): Training target.
    X_val (pd.DataFrame): Validation features.
    y_val (pd.Series): Validation target.
    cat_columns (List): List of categorical column names.
    num_columns (List): List of numerical column names.

    Returns:
    optuna.Study: Optuna study object.
    """
    study = optuna.create_study(direction="maximize")

    study.optimize(
        func = lambda trial: objective(trial, X_train, y_train, X_val, y_val, cat_columns, num_columns),
        n_trials = 100,
        show_progress_bar=True
    )

    return study

def reproduce_best_run(study: optuna.Study, X_train: pd.DataFrame, y_train: pd.Series, X_val: pd.DataFrame, y_val: pd.Series, cat_columns: List[str], num_columns: List[str]) -> Pipeline:
    """
    Reproduce the best run using the study's best parameters.

    Parameters:
    study (optuna.Study): Optuna study object.
    X_train (pd.DataFrame): Training features.
    y_train (pd.Series): Training target.
    X_val (pd.DataFrame): Validation features.
    y_val (pd.Series): Validation target.
    cat_columns (List): List of categorical column names.
    num_columns (List): List of numerical column names.

    Returns:
    model_pipeline (Pipeline): Model pipeline trained with the study's best parameters.
    """
    model_pipeline = make_pipeline(cat_columns, num_columns)

    model_pipeline.set_params(**study.best_params)

    model_pipeline.fit(X_train, y_train)

    # Check if we get exact same score
    assert study.best_value == model_pipeline.score(X_val, y_val), "Reproducibility Error : Could not reproduce best score using the study's best params."

    return model_pipeline