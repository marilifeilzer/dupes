import shap
import pandas as pd
from dupes.model.optimiser import load_model
from dupes.model.price_prediction import preprocess_data, preprocess_prediction_input, train_model
shap.initjs()

def get_shaply_value(df: pd.DataFrame, index: int):

    # Preprocess data and load model
    preprocess = preprocess_data(df)
    target = df['price_eur'] / df['volume_ml']
    X = preprocess.drop(columns=['price_eur'])
    model = load_model()

    # Calculate the prediction
    feature_values = X.iloc[[index]]
    volume = feature_values['volume_ml']
    prediction = model.predict(feature_values).item()
    pred = volume * prediction

    # Calculate Shaply values
    explainer = shap.Explainer(model)
    shap_values_one = explainer(feature_values)

    base_value = shap_values_one.base_values * volume
    sum_shap_values = float(shap_values_one.values.sum) * volume

    print(f"Base value: {base_value}")
    print(f"Sum of SHAP values: {sum_shap_values}")
    print(f"The prediction for this instance: {pred}")


if __name__ == '__main__':
    file = '/Users/panamas/code/marili/dupes/raw_data/products_data_1012.csv'
    df = pd.read_csv(file)
    get_shaply_value(df, 5)
