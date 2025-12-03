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


