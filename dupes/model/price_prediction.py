import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from dupes.data.gc_client import load_table_to_df

import os
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor

from dupes.data.gc_client import download_model, load_table_to_df, upload_model

CACHE_ROOT = Path(os.getenv("MODELS_CACHE_DIR", "models_cache"))
PRICE_DIR = CACHE_ROOT / "price"
MODEL_PATH = PRICE_DIR / "xgb_best.pkl"
GCS_PRICE_BLOB = os.getenv("PRICE_MODEL_BLOB", "price/xgb_best.pkl")

def _ensure_dirs() -> None:
    PRICE_DIR.mkdir(parents=True, exist_ok=True)


def ensure_price_model() -> Path:
    """
    Ensure the price model exists locally, downloading it from GCS if not.
    """
    _ensure_dirs()
    if not MODEL_PATH.is_file():
        download_model(GCS_PRICE_BLOB, MODEL_PATH)
    return MODEL_PATH


def save_price_model(model) -> Path:
    """
    Persist the trained model locally and upload it to GCS.
    """
    _ensure_dirs()
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)
    upload_model(MODEL_PATH, GCS_PRICE_BLOB)
    return MODEL_PATH


def load_price_model():
    path = ensure_price_model()
    with open(path, "rb") as f:
        return pickle.load(f)

def preprocess_data(df: pd.DataFrame, manufacturer = False):

    # Create data frame
    cols_to_keep = ['price_eur', 'volume_ml', 'formula']
    if manufacturer:
        cols_to_keep.append('manufacturer_name')

    df_new = df[cols_to_keep]
    df_new = df_new.dropna()
    df_new['formula'] = df_new['formula'].str.replace('[','')

    split_ingr_threshold = 50
    # note: from ingredient 50 onwards, we have a significant amount of missing values

    # Create ingredient features with preserved order
    df_split = df_new['formula'].str.split(",", expand=True)
    df_split = df_split.replace('[','')

    # Drop ingredient columns with too many missing values
    num_ingr = len(df_split.columns)
    for index in range(split_ingr_threshold,num_ingr):
        df_split = df_split.drop(columns=index)

    df_split.columns = [f'_{col}' if type(col) == int else col for col in df_split.columns]

    df_split = df_split.select_dtypes('object').astype('category')

    # Concatenate once more
    df_new = df_new.drop(columns='formula')
    df_new = pd.concat([df_new,df_split], axis=1)

    if manufacturer:
        df_new['manufacturer_name'] = df_new['manufacturer_name'].astype('category')

    return df_new

def train_model(df: pd.DataFrame):

    # Define target and features
    target = df['price_eur'] / df['volume_ml']
    X = df.drop(columns=['price_eur'])

    # Create validation split for early-stopping purposes
    X_train, X_eval, y_train, y_eval = train_test_split(
        X,
        target,
        train_size=0.8,
        random_state=42)

    # Instantiate model
    model_xgb_early_stopping = XGBRegressor(
        max_depth=10,
        n_estimators=300,
        objective="reg:squarederror",
        eval_metric=["rmse"],
        learning_rate=0.1,
        early_stopping_rounds=20,
        enable_categorical=True,
        random_state=42,
        num_boost_round=5000)

    # Fit model
    model_xgb_early_stopping.fit(
        X_train,
        y_train,
        verbose=2,
        eval_set=[(X_train, y_train), (X_eval, y_eval)])

    return model_xgb_early_stopping

def preprocess_prediction_input(df: pd.DataFrame, manufacturer = False):

    # Create data frame
    cols_to_keep = ['price_eur', 'volume_ml', 'formula']
    if manufacturer:
        cols_to_keep.append('manufacturer_name')

    df_new = df[cols_to_keep]
    df_new = df_new.dropna()
    df_new['formula'] = df_new['formula'].str.replace('[','')

    # Create ingredient features with preserved order
    df_split = df_new['formula'].str.split(",", expand=True)
    df_split = df_split.replace('[','')

    # Fix: add empty columns until split_ingr_threshold
    # note: from ingredient 50 onwards, we have a significant amount of missing values
    split_ingr_threshold = 50
    num_ingr = len(df_split.columns)

    if num_ingr <=split_ingr_threshold:
        for i in range(num_ingr, split_ingr_threshold):
            df_split[i] = ' '

    for index in range(split_ingr_threshold,num_ingr):
        df_split = df_split.drop(columns=index)

    df_split.columns = [f'_{col}' if type(col) == int  else col for col in df_split.columns]
    df_split = df_split.select_dtypes('object').astype('category')

    # Concatenate once more
    df_new = df_new.drop(columns='formula')
    df = pd.concat([df_new,df_split], axis=1)

    if manufacturer:
        df_new['manufacturer_name'] = df_new['manufacturer_name'].astype('category')

    return df

if __name__ == '__main__':

    # Run previous methods
    df = load_table_to_df()
    preprocess = preprocess_data(df)
    model = train_model(preprocess)
    save_price_model(model)

    # Retrieve performance metrics
    results = model.evals_result()
    epochs = len(results['validation_0']["rmse"])
    x_axis = range(0, epochs)

    # Plot RMSLE loss
    fig, ax = plt.subplots()

    ax.plot(x_axis, results['validation_0']['rmse'], label='Train')
    ax.plot(x_axis, results['validation_1']['rmse'], label='Val')
    ax.legend(); plt.ylabel('RMSE (of log)'); plt.title('XGBoost Log Loss')
    # plt.show()

    # Show prediction scores
    best_val_score = min(results['validation_1']['rmse'])**2
    mean_price = np.mean(df['price_eur'])/np.mean(df['volume_ml'])

    print("Best Validation Score (price/ml)", round(best_val_score, 3))
    print("Mean product (price/ml)", round(mean_price, 3))
    print("Error margin vs. mean (%)", round((best_val_score/mean_price)*100, 3))
