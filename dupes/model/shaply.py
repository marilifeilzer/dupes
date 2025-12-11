import shap
import numpy as np
import pandas as pd
from dupes.model.optimiser import load_model_base
from dupes.model.price_prediction import preprocess_data, preprocess_prediction_input, train_model
shap.initjs()

def get_local_shaply_values(df: pd.DataFrame, index: int, manufacturer = False):

    # Preprocess data and load model
    preprocess = preprocess_data(df, manufacturer)
    target = df['price_eur'] / df['volume_ml']
    X = preprocess.drop(columns=['price_eur'])

    model = load_model_base(manufacturer=manufacturer)

    # Calculate the prediction
    feature_values = X.iloc[[index]]
    volume = feature_values['volume_ml']

    prediction = model.predict(feature_values).item()
    pred = volume * prediction

    # Calculate Shaply values
    explainer = shap.Explainer(model)
    shap_values_loc = explainer(feature_values)

    base_values = shap_values_loc.base_values * volume

    sum_shap_values = float(shap_values_loc.values.sum()) * volume
    shap_values_loc.values = np.array([shap_values_loc.values[0]]) * volume.values[0]


    print(f"Base value: {base_values}")
    print(f"Sum of SHAP values: {sum_shap_values}")
    print(f"The prediction for this instance: {pred}")

    shap.plots.bar(shap_values_loc[0])
    #shap.plots.waterfall(shap_values_one[0])
    #shap.plots.force(shap_values_one[0])

def get_global_shaply_values(df: pd.DataFrame, manufacturer = True):

    # Preprocess data and load model
    preprocess = preprocess_data(df, manufacturer)
    target = df['price_eur'] / df['volume_ml']
    X = preprocess.drop(columns=['price_eur'])
    mean_volume = np.mean(X['volume_ml'])

    model = load_model(manufacturer=manufacturer)

    # Calculate Shaply values
    explainer = shap.Explainer(model)
    shap_values = explainer(X)

    shap_values.values = np.array([shap_values.values[0]]) * mean_volume

    shap.plots.bar(shap_values)

if __name__ == '__main__':
    file = '/Users/panamas/code/marili/dupes/raw_data/products_data_1012.csv'
    df = pd.read_csv(file)
    get_global_shaply_values(df, manufacturer=True)
