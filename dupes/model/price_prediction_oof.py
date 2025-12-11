import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import KFold, train_test_split, cross_val_score
from xgboost import XGBRegressor
import pickle

from dupes.model.price_prediction import preprocess_data
from dupes.model.optimiser import load_model_base
from dupes.data.gc_client import load_table_to_df

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

    # Get model parameters from hypertuned xgb model (see best_params.json)
    model_params = {
        "objective": "reg:squarederror",
        "eval_metric": "rmse",
        "tree_method": "hist",
        "learning_rate": 0.05782733575986101,
        "max_bin": 351,
        "max_delta_step": 6,
        "min_child_weight": 18,
        "colsample_bytree": 0.9990386139113363,
        "lambda": 0.00242507240752407,
        "alpha": 0.06070159380963301,
        "random_state": 42,
        "verbosity": 0,
        "max_depth": 9,
        "subsample": 0.7332731104196174,
        "scale_pos_weight": 5.163943066355587
    }

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
    df_meta = pd.DataFrame()
    df_meta["xgb_1"] = oof_predictions
    df_meta["target"] = target

    X_meta = df_meta[["xgb_1"]]
    target = df_meta["target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X_meta,
        target,
        train_size=0.8,
        random_state=42)

    # Train meta model
    meta_model = XGBRegressor(**model_params,
                              enable_categorical=True,
                              num_boost_round=5000)

    # Save meta model as pickle
    best_model = meta_model.fit(X_meta, target)

    if manufacturer:
        file_name = "xgb_meta_manu.pkl"
    else:
        file_name = "xgb_meta.pkl"

    print('...writing to pickle file...')
    pickle.dump(best_model, open(file_name, "wb"))

    # Evaluate meta model (best validation score: -0,0026)
    score = np.mean(cross_val_score(meta_model, X_test, y_test, cv=5, scoring="neg_mean_squared_error"))

    return score

# Load pickle with fitted model
def load_model_meta(manufacturer = False):

    if manufacturer:
        file_name = "xgb_meta_manu.pkl"
    else:
        file_name = "xgb_meta.pkl"
    loaded_model = pickle.load(open(file_name, "rb"))

    return loaded_model


if __name__ == '__main__':

    # Run previous methods
    df = load_table_to_df()
    score = out_of_fold_prediction(df, manufacturer=True)
    print(score)
