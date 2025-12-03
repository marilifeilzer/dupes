import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer
from unidecode import unidecode

def clean_categories(dataframe):

    dupes_df = dataframe.dropna(subset=["propiedad"])

    dupes_df["propiedad"] = dupes_df["propiedad"].map(lambda x: x.split(","))

    dupes_df["propiedad"] = dupes_df["propiedad"].map(lambda x: [v.strip() for v in x])

    dupes_df["propiedad"] = dupes_df["propiedad"].map(lambda x: [v.lower() for v in x])

    dupes_df["propiedad"] = dupes_df["propiedad"].map(lambda x: list(set(x)))

    dupes_df["propiedad"] = dupes_df["propiedad"].map(lambda x: list(filter(lambda y: y.strip(), x)))

    dupes_df["propiedad"] = dupes_df["propiedad"].map(lambda x: [unidecode(v) for v in x])

    dupes_df["propiedad"] = dupes_df["propiedad"].map(lambda x: [v.replace("-"," ")for v in x])

    dupes_df.reset_index(drop=True, inplace=True)

    return dupes_df[["product_id","propiedad"]]


def encode_properties(dataframe, col):

    mlb = MultiLabelBinarizer()

    mlb_df = pd.DataFrame(mlb.fit_transform(dataframe[col]),
                 columns=mlb.classes_,
                 index=dataframe.index
                 )

    return pd.concat([dataframe.drop(columns=[col]),mlb_df], axis=1)

def price_and_vol_clean(data):
    price_str = data["price"].astype(str)
    price_first_number = price_str.str.extract(r'(\d+[.,]\d+)')[0]

    data["price_eur"] = (
        price_first_number
        .str.replace(".", "", regex=False)  #remove thousands separator
        .str.replace(",", ".", regex=False)  #convert decimal comma to dot
        .pipe(pd.to_numeric, errors="coerce")
    )

    #Volume: keep only numeric ml value
    volume_str = data["volume"].astype(str)
    volume_number = volume_str.str.extract(r'(?i)([\d\.,]+)\s*ml')[0]

    data["volume_ml"] = (
        volume_number
        .str.replace(".", "", regex=False)
        .str.replace(",", ".", regex=False)
        .pipe(pd.to_numeric, errors="coerce")
    )

    #Drop original columns
    data = data.drop(columns=["price", "volume"])