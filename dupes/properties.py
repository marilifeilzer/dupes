import pandas as pd
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv("/Users/lewagon/code/marilifeilzer/dupes/raw_data/products_data.csv")

dupes_df = df.dropna(subset=["propiedad"])

def encode_target(dataframe):
    
    target = dataframe["propiedad"]
    
    label_encoder = LabelEncoder()
    
    label_encoder.fit(target)
    
    encoded_target = label_encoder.transform(target)