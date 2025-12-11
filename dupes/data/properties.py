import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer
from unidecode import unidecode
import pickle


def parse_formula(x):
    if isinstance(x, str):
        if x.startswith("[") and x.endswith("]"):
            try:
                import ast

                return ast.literal_eval(x)
            except (ValueError, SyntaxError):
                return [item.strip().strip("'\"") for item in x.strip("[]").split(",")]
        else:
            return [item.strip() for item in x.split(",")]
    elif isinstance(x, list):
        return x
    else:
        return []


def clean_categories(dataframe):

    dupes_df = dataframe.dropna(subset=["propiedad"])

    dupes_df["propiedad"] = dupes_df["propiedad"].map(lambda x: x.split(","))

    dupes_df["propiedad"] = dupes_df["propiedad"].map(lambda x: [v.strip() for v in x])

    dupes_df["propiedad"] = dupes_df["propiedad"].map(lambda x: [v.lower() for v in x])

    dupes_df["propiedad"] = dupes_df["propiedad"].map(lambda x: list(set(x)))

    dupes_df["propiedad"] = dupes_df["propiedad"].map(
        lambda x: list(filter(lambda y: y.strip(), x))
    )

    dupes_df["propiedad"] = dupes_df["propiedad"].map(
        lambda x: [unidecode(v) for v in x]
    )

    dupes_df["propiedad"] = dupes_df["propiedad"].map(
        lambda x: [v.replace("-", " ") for v in x]
    )

    dupes_df.reset_index(drop=True, inplace=True)

    return dupes_df[["product_id", "propiedad"]]


def encode_properties(dataframe, col):
    dataframe = dataframe.dropna(subset=col)
    mlb = MultiLabelBinarizer()

    column_name = col[0] if isinstance(col, list) else col

    dataframe[column_name] = dataframe[column_name].apply(parse_formula)

    dataframe[column_name] = dataframe[column_name].apply(
        lambda x: [each for each in x if isinstance(each, str) and each.strip()]
    )

    mlb_df = pd.DataFrame(
        mlb.fit_transform(dataframe[column_name]),
        columns=mlb.classes_,
        index=dataframe.index,
    )

    # save
    with open("model.pkl", "wb") as f:
        pickle.dump(mlb, f)

    return pd.concat([dataframe[["product_id"]], mlb_df], axis=1)


def use_encoder_load(dataframe, col):

    with open("model.pkl", "rb") as f:
        mlb = pickle.load(f)

    # Parse formula strings into lists just like in encode_properties
    dataframe[col] = dataframe[col].apply(parse_formula)

    # Filter to keep only string elements in each list
    dataframe[col] = dataframe[col].apply(lambda x: [each for each in x if isinstance(each, str) and each.strip()])

    mlb_df = pd.DataFrame(
        mlb.transform(dataframe[col]),
        columns=mlb.classes_,
        index=dataframe.index,
    )

    return mlb_df


def price_and_vol_clean(data):

    price_str = data["price"].astype(str)
    price_first_number = price_str.str.extract(r"(\d+[.,]\d+)")[0]

    data["price_eur"] = (
        price_first_number.str.replace(
            ".", "", regex=False
        )  # remove thousands separator
        .str.replace(",", ".", regex=False)  # convert decimal comma to dot
        .pipe(pd.to_numeric, errors="coerce")
    )

    # Volume: keep only numeric ml value
    volume_str = data["volume"].astype(str)
    volume_number = volume_str.str.extract(r"(?i)([\d\.,]+)\s*ml")[0]

    data["volume_ml"] = (
        volume_number.str.replace(".", "", regex=False)
        .str.replace(",", ".", regex=False)
        .pipe(pd.to_numeric, errors="coerce")
    )

    # Drop original columns
    data = data.drop(columns=["price", "volume"])
    return data


def encode_hair_colors(df):
    # Fixed set of categories for hair color
    HAIR_COLORS = [
        "Todos los colores de cabello",
        "Cabello rubio",
        "Cabello Rubio Platino",
        "Cabello Blanco-Gris",
        "Cabello gris",
        "Cabello castaño",
    ]

    # Impute NaN in the original column with the mode
    df["color_de_cabello"] = df["color_de_cabello"].fillna(
        "Todos los colores de cabello"
    )

    # Convert strings into lists
    def split_colors(x: str):
        return [p.strip() for p in str(x).split(",") if p.strip()]

    df["color_list"] = df["color_de_cabello"].apply(split_colors)

    # One-hot encode from the lists
    dummies = df["color_list"].explode().str.get_dummies().groupby(level=0).sum()

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

    # Impute NaN in the original column with the mode
    df["tipo_de_cabello"] = df["tipo_de_cabello"].fillna("Todo tipo de cabello")

    # Split values into lists
    def split_hair_types(x: str):
        return [p.strip() for p in str(x).split(",") if p.strip()]

    df["tipo_list"] = df["tipo_de_cabello"].apply(split_hair_types)

    # One-hot encode from the lists
    dummies = df["tipo_list"].explode().str.get_dummies().groupby(level=0).sum()

    dummies = dummies[HAIR_TYPES]
    df = pd.concat([df, dummies], axis=1)
    df = df.drop(columns=["tipo_list", "tipo_de_cabello"])

    return df
