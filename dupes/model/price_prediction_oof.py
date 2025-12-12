import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import KFold, train_test_split, cross_val_score
from xgboost import XGBRegressor
import pickle
import json
import optuna

from dupes.model.optimiser import load_model_base
from dupes.data.gc_client import load_table_to_df, upload_model
from dupes.model.model_paths import get_price_meta_path, get_price_meta_gcs_blob, ensure_model_dirs
from dupes.model.price_prediction import (
    load_price_model,
    preprocess_data,
    save_price_model,
)


def out_of_fold_prediction(df: pd.DataFrame, manufacturer=False):

    # Preprocess data
    df = preprocess_data(df, manufacturer=manufacturer)

    # Define target and features
    target = df['price_eur'] / df['volume_ml']
    X = df.drop(columns=['price_eur'])

    # Create out of fold variables
    N_FOLDS = 5
    cv = KFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
    oof_predictions = np.zeros(len(X))
    fold_scores = []

    # Dynamically load model parameters from hypertuned xgb model
    with open("best_params.json") as file:
        best_params = json.load(file)

    model_params = {**best_params,
                    "random_state": 42,
                    "verbosity": 0,
                    "objective": "reg:squarederror",
                    "eval_metric": "rmse",
                    "tree_method": "hist"}

    # Do out of fold predictions
    for fold, (train_idx, val_idx) in enumerate(cv.split(X)):

        X_train_fold = X.iloc[train_idx]
        X_val_fold = X.iloc[val_idx]
        y_train_fold = target.iloc[train_idx]
        y_val_fold = target.iloc[val_idx]

        model = XGBRegressor(**model_params,
                            early_stopping_rounds=20,
                            enable_categorical=True,
                            num_boost_round=5000)

        model.fit(X_train_fold,
                  y_train_fold,
                  verbose=1,
                  eval_set=[(X_train_fold, y_train_fold), (X_val_fold, y_val_fold)])

        fold_predictions = model.predict(X_val_fold)
        oof_predictions[val_idx] = fold_predictions

    # Use out-of-fold predictions as a new feature
    df_meta = pd.DataFrame({"xgb_1":oof_predictions, "target":target})
    df_meta.to_csv('meta.csv', index = False)

    X_meta = df_meta[["xgb_1"]]
    target = df_meta["target"]

    # Train meta model
    meta_model = XGBRegressor(**model_params,num_boost_round=5000)

    # Save meta model as pickle
    best_model = meta_model.fit(X_meta, target)

    ensure_model_dirs()
    model_path = get_price_meta_path(manufacturer)
    gcs_blob = get_price_meta_gcs_blob(manufacturer)

    print('...writing to pickle file...')
    with open(model_path, "wb") as f:
        pickle.dump(best_model, f)
    upload_model(model_path, gcs_blob)

    # Evaluate meta model (best validation score: -0,0035)
    score = np.mean(cross_val_score(meta_model, X_meta, target, cv=5, scoring="neg_mean_squared_error"))

    return score

def objective(trial):

    df_meta = pd.read_csv('meta.csv')

    X_meta = df_meta[["xgb_1"]]
    target = df_meta["target"]

    # Define Optuna hyperparameters
    params = {
        "objective": "reg:squarederror",
        "eval_metric": "rmse",
        "learning_rate": trial.suggest_float("learning_rate", 1e-4, 0.5, log=True),
        "enable_categorical": True,
        "nthread": -1,
        "max_bin": trial.suggest_int("max_bin", 256, 1024),
        "max_delta_step": trial.suggest_int("max_delta_step", 1, 20),
        "lambda": trial.suggest_float("lambda", 1e-3, 10.0, log=True),
        "alpha": trial.suggest_float("alpha", 1e-3, 10.0, log=True),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 20),
        "max_depth": trial.suggest_int("max_depth", 3, 15),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "scale_pos_weight": trial.suggest_float("scale_pos_weight", 0.1, 10.0),
    }

    # Instantiate model
    model_xgb = XGBRegressor(**params)

    # Evaluate model
    score = np.mean(
        cross_val_score(model_xgb, X_meta, target, cv=5, scoring="neg_mean_squared_error")
    )

    return score

# Load pickle with fitted model
def ensure_meta_model(manufacturer=False):
    """Ensure the meta model exists, downloading if necessary."""
    from dupes.data.gc_client import download_model

    model_path = get_price_meta_path(manufacturer)
    if not model_path.exists():
        ensure_model_dirs()
        gcs_blob = get_price_meta_gcs_blob(manufacturer)
        download_model(gcs_blob, model_path)
    return model_path


def load_model_meta(manufacturer=False):
    # Ensure model exists first
    ensure_meta_model(manufacturer)
    model_path = get_price_meta_path(manufacturer)
    with open(model_path, "rb") as f:
        loaded_model = pickle.load(f)
    return loaded_model


if __name__ == '__main__':
    manufacturer = True

    # Run the models
    df = load_table_to_df()
    score = out_of_fold_prediction(df, manufacturer=manufacturer)
    print(f'loss of baseline meta model: {score}')

    # Create and run the optimization process with 1000 trials
    study_meta = optuna.create_study(study_name="xgboost_meta_study", direction="maximize")
    study_meta.optimize(objective, n_trials=1000, show_progress_bar=True)

    # Retrieve the best parameter values
    best_params_meta = study_meta.best_params
    print(f"\nBest parameters: {best_params_meta}")

    # Save best parameters as json
    with open("best_params_meta.json", "w") as f:
        json.dump(best_params_meta, f)

    # Save best model as pickle
    with open("best_params_meta.json", "r") as f:
        best_params_meta = json.load(f)

    model_xgb_meta = XGBRegressor(
        **best_params_meta,
        objective="reg:squarederror",
        eval_metric="rmse",
        nthread=-1
    )

    df_meta = pd.read_csv('meta.csv')
    X_meta = df_meta[["xgb_1"]]
    target = df_meta["target"]

    best_model_meta = model_xgb_meta.fit(X_meta, target)

    print("...writing to pickle file...")
    save_price_model(best_model_meta, manufacturer=manufacturer)
