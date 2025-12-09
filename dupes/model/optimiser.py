import pandas as pd
import numpy as np
import pickle
import json
from dupes.model.price_prediction import preprocess_data
from xgboost import XGBRegressor
import optuna
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split, cross_val_score

# Create data
file = '/Users/panamas/code/marili/dupes/raw_data/products_0812.csv'
df = pd.read_csv(file)
preprocess = preprocess_data(df)

target = preprocess['price_eur'] / preprocess['volume_ml']
X = preprocess.drop(columns=['price_eur'])

# Optimise the model with hyper parameter tuning
def objective(trial):

    # Define Optuna hyper parameters
    params = {
        'objective': 'reg:squarederror',
        'eval_metric': 'rmse',
        'learning_rate': trial.suggest_float('learning_rate', 1e-4, 0.5,log=True),
        'enable_categorical': True,
        'nthread': -1,
        'max_bin': trial.suggest_int('max_bin', 256, 1024),
        'max_delta_step': trial.suggest_int('max_delta_step', 1, 20),
        'lambda': trial.suggest_float('lambda', 1e-3, 10.0, log=True),
        'alpha': trial.suggest_float('alpha', 1e-3, 10.0, log=True),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 20),
        'max_depth': trial.suggest_int('max_depth', 3, 15),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'scale_pos_weight': trial.suggest_float('scale_pos_weight', 0.1, 10.0)
    }

    # Instantiate model
    model_xgb = XGBRegressor(**params)

    # Evaluate model
    score = np.mean(cross_val_score(model_xgb, X, target, cv=5, scoring="neg_mean_squared_error"))

    return score

# Load pickle with fitted model
def load_model():

    file_name = "xgb_best.pkl"
    loaded_model = pickle.load(open(file_name, "rb"))

    return loaded_model


if __name__ == '__main__':

    # Create and run the optimization process with 100 trials
    study = optuna.create_study(study_name="xgboost_study", direction='maximize')
    study.optimize(objective, n_trials=1000, show_progress_bar=True)

    # Retrieve the best parameter values
    best_params = study.best_params
    print(f"\nBest parameters: {best_params}")

    # Save best parameters as jso
    with open('best_params.json', 'w') as f:
        json.dump(best_params, f)

    # Save best model as pickle
    with open('best_params.json', 'r') as f:
        best_params = json.load(f)

    model_xgb = XGBRegressor(**best_params,
                             objective = 'reg:squarederror',
                             eval_metric = 'rmse',
                             enable_categorical =  True,
                             nthread = -1)
    best_model = model_xgb.fit(X, target)

    file_name = "xgb_best.pkl"
    pickle.dump(best_model, open(file_name, "wb"))
