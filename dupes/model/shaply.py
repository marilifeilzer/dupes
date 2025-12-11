import shap
import numpy as np
import pandas as pd
from dupes.model.optimiser import load_model
from dupes.model.price_prediction import preprocess_data, preprocess_prediction_input, train_model
shap.initjs()

def get_shaply_value(df: pd.DataFrame, index: int, manufacturer = False):

    # Preprocess data and load model
    preprocess = preprocess_data(df)
    target = df['price_eur'] / df['volume_ml']
    X = preprocess.drop(columns=['price_eur'])

    model = load_model(manufacturer=manufacturer)

    # Calculate the prediction
    feature_values = X.iloc[[index]]
    volume = feature_values['volume_ml']

    prediction = model.predict(feature_values).item()
    pred = volume * prediction

    # Calculate Shaply values
    explainer = shap.Explainer(model)
    shap_values_one = explainer(feature_values)

    base_value = shap_values_one.base_values * volume

    sum_shap_values = float(shap_values_one.values.sum()) * volume

    shap_values_one.values = np.array([shap_values_one.values[0]]) * volume.values[0]


    print(f"Base value: {base_value}")
    print(f"Sum of SHAP values: {sum_shap_values}")
    print(f"The prediction for this instance: {pred}")

    shap.plots.bar(shap_values_one[0])
    #shap.plots.waterfall(shap_values_one[0])
    #shap.plots.force(shap_values_one[0])

if __name__ == '__main__':
    file = '/Users/panamas/code/marili/dupes/raw_data/products_data_1012.csv'
    df = pd.read_csv(file)
    get_shaply_value(df, 200)
