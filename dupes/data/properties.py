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
    return data

def encode_hair_colors(data):
    #Fixed set of categories for hair color
    HAIR_COLORS = [
        "Todos los colores de cabello",
        "Cabello rubio",
        "Cabello Rubio Platino",
        "Cabello Blanco-Gris",
        "Cabello gris",
        "Cabello castaño",
    ]

    #Impute NaN in the original column with the mode
    df["color_de_cabello"] = df["color_de_cabello"].fillna("Todos los colores de cabello")

    #Convert strings into lists
    def split_colors(x: str):
        return [p.strip() for p in str(x).split(",") if p.strip()]

    df["color_list"] = df["color_de_cabello"].apply(split_colors)

    #One-hot encode from the lists
    dummies = (
        df["color_list"]
        .explode()
        .str.get_dummies()
        .groupby(level=0)
        .sum()
    )

    dummies = dummies[HAIR_COLORS]
    df = pd.concat([df, dummies], axis=1)
    df = df.drop(columns=["color_list", "color_de_cabello"])
    return df

def encode_hair_type(data):
    HAIR_TYPES = [
        "Todo tipo de cabello",
        "Cabellos teñidos",
        "Cabello fino",
        "Cabello estresado",
        "Cabello seco",
        "Cabello quebradizo",
        "Cabello rizado",
        "Cabello normal",
        "Cabello con volumen",
        "Cabello apagado",
        "Cabello ondulado",
        "Cabello grueso",
        "Cabello graso",
        "Cuero cabelludo sensible",
        "Cuero cabelludo",
        "Cabello rebelde",
        "Cabello dañado por el sol",
        "Cabello liso",
        "Cabello dañado",
    ]

    #Impute NaN in the original column with the mode
    df["tipo_de_cabello"] = df["tipo_de_cabello"].fillna("Todo tipo de cabello")

    #Split values into lists
    def split_hair_types(x: str):
        return [p.strip() for p in str(x).split(",") if p.strip()]

    df["tipo_list"] = df["tipo_de_cabello"].apply(split_hair_types)

    #One-hot encode from the lists
    dummies = (
        df["tipo_list"]
        .explode()
        .str.get_dummies()
        .groupby(level=0)
        .sum()
    )

    dummies = dummies[HAIR_TYPES]
    df = pd.concat([df, dummies], axis=1)
    df = df.drop(columns=["tipo_list", "tipo_de_cabello"])
    
    return df
