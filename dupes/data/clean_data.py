import pandas as pd
import numpy as np
import re
import unicodedata
from dupes.data.config import MANUFACTURER_MAP, ORDERED_COLS, PRODUCT_NAME_PATTERNS


def load_raw_data(file_path: str) -> pd.DataFrame:
    """
    Load raw CSV data and drop unused columns if they exist.
    Keep only rows where `ingredients_text` is non-empty (when present).
    """
    df = pd.read_csv(file_path)

    cols_to_drop = [
        "zona", "color", "opacidad", "tendencia",
        "regalo_para", "mini_size", "acabado", "fragancia",
        "estuche", "impermeable", "tipo_de_piel", "manufacturer_email",
        "season", "variante_de_recambio_disponible", "disponible_en_version_recargable",
        "textura", "manufacturer_address", "edad", "zona_de_aplicacion",
        "tipo_de_producto", "articulo_no", "tyoe", "page", "variant", "url",
    ]

    # Drop only those that actually exist
    df = df.drop(columns=[c for c in cols_to_drop if c in df.columns], errors="ignore")

    # If ingredients_text exists, keep only non-empty rows (old behaviour)
    if "ingredients_text" in df.columns:
        df = df[
            df["ingredients_text"].notna()
            & (df["ingredients_text"].astype(str).str.strip() != "")
        ]

    return df


def clean_price_volume(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean `price` and `volume` columns and create:
    - price_eur (float, EUR)
    - volume_ml (float, ml)

    If `price` / `volume` are missing (e.g. pre-cleaned DB),
    the function is a no-op for those parts.
    """
    df = df.copy()

    # Price → price_eur
    if "price" in df.columns:
        price_str = df["price"].astype(str)
        price_first_number = price_str.str.extract(r"(\d+[.,]\d+)")[0]

        df["price_eur"] = (
            price_first_number
            .str.replace(".", "", regex=False)   # thousands separator
            .str.replace(",", ".", regex=False)  # decimal comma → dot
            .pipe(pd.to_numeric, errors="coerce")
        )

    # Volume → volume_ml
    if "volume" in df.columns:
        volume_str = df["volume"].astype(str)
        volume_number = volume_str.str.extract(r"(?i)([\d\.,]+)\s*ml")[0]

        df["volume_ml"] = (
            volume_number
            .str.replace(".", "", regex=False)
            .str.replace(",", ".", regex=False)
            .pipe(pd.to_numeric, errors="coerce")
        )

    # Drop raw columns only if they exist
    for col in ["price", "volume"]:
        if col in df.columns:
            df = df.drop(columns=[col])

    return df


def imputer(df: pd.DataFrame) -> pd.DataFrame:
    """
    Impute missing values for:
    - color_de_cabello: "Todos los colores de cabello"
    - tipo_de_cabello: "Todo tipo de cabello"
    - propiedad: "Detergente"
    - description: "" instead of NaN, stripped

    Operates only on columns that actually exist.
    """
    df = df.copy()

    if "color_de_cabello" in df.columns:
        df["color_de_cabello"] = df["color_de_cabello"].fillna(
            "Todos los colores de cabello"
        )

    if "tipo_de_cabello" in df.columns:
        df["tipo_de_cabello"] = (
            df["tipo_de_cabello"]
            .replace("nan", pd.NA)          # handle literal "nan"
            .fillna("Todo tipo de cabello") # real NaN / None
        )

    if "propiedad" in df.columns:
        df["propiedad"] = df["propiedad"].fillna("Detergente")

    if "description" in df.columns:
        df["description"] = df["description"].fillna("").astype(str).str.strip()

    return df


def fill_missing_manufacturer(df: pd.DataFrame) -> pd.DataFrame:
    """
    For rows where manufacturer_name is NaN, infer manufacturer from product_name
    by taking everything before 'Champú capilar' (or its mojibake form).

    If manufacturer_name is missing from schema, it is created.
    If product_name is missing, function is a no-op.
    """
    df = df.copy()

    # Ensure manufacturer_name exists
    if "manufacturer_name" not in df.columns:
        df["manufacturer_name"] = pd.NA

    if "product_name" not in df.columns:
        return df.reset_index(drop=True)

    mask = df["manufacturer_name"].isna()
    if not mask.any():
        return df.reset_index(drop=True)

    # Regex: capture everything BEFORE “Champú capilar” OR “Champ├║ capilar”
    pat = r"^(.*?)\s+Champ(?:ú|├║)\s+capilar"

    df.loc[mask, "manufacturer_name"] = (
        df.loc[mask, "product_name"]
        .astype(str)
        .str.extract(pat, expand=False)
        .fillna("")
        .str.strip()
    )

    return df.reset_index(drop=True)


def normalize_name(s: str) -> str:
    """
    Lowercase, strip, remove accents/diacritics.
    Returns a normalized ascii string suitable for mapping.
    """
    if not isinstance(s, str):
        return ""
    s = s.strip().lower()
    s = unicodedata.normalize("NFKD", s)
    s = s.encode("ascii", "ignore").decode("utf-8", errors="ignore")
    return s.strip()


def clean_manufacturer(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize and canonicalize manufacturer_name into a clean, title-cased column.
    - Uses MANUFACTURER_MAP from config.py
    - Drops original manufacturer_name and replaces it with cleaned version.

    If manufacturer_name is missing, this is a no-op.
    """
    df = df.copy()

    if "manufacturer_name" not in df.columns:
        return df

    # Normalize
    df["manufacturer_norm"] = df["manufacturer_name"].fillna("").apply(normalize_name)

    # Map to canonical key, fallback to normalized original, then Title Case
    df["manufacturer_clean"] = (
        df["manufacturer_norm"]
        .map(MANUFACTURER_MAP)
        .fillna(df["manufacturer_norm"])
        .str.title()
    )

    # Drop helper/original columns, rename clean -> manufacturer_name
    df = df.drop(columns=["manufacturer_norm", "manufacturer_name"])
    df = df.rename(columns={"manufacturer_clean": "manufacturer_name"})

    return df


def clean_product_name(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove generic middle tags like 'Champú capilar' from product_name.
    Uses PRODUCT_NAME_PATTERNS from config.py.

    If product_name is missing, this is a no-op.
    """
    df = df.copy()

    if "product_name" not in df.columns:
        return df

    s = df["product_name"].astype(str)
    for pat in PRODUCT_NAME_PATTERNS:
        s = s.str.replace(pat, " ", regex=True)

    # Collapse multiple spaces and trim
    s = s.str.replace(r"\s{2,}", " ", regex=True).str.strip()
    df["product_name"] = s

    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Full cleaning pipeline on an already-loaded raw DataFrame.

    Steps:
    - clean_price_volume: build price_eur / volume_ml, drop price / volume if present
    - imputer: fill NaNs in color_de_cabello, tipo_de_cabello, propiedad, description
    - fill_missing_manufacturer: infer missing manufacturer_name from product_name
    - clean_manufacturer: normalize + map manufacturer_name using MANUFACTURER_MAP
    - clean_product_name: strip generic middle tags like 'Champú capilar'
    - reorder columns to ORDERED_COLS (from config.py), using only those that exist
    """
    df = df.copy()

    # 1) Price / volume (no-op if missing)
    df = clean_price_volume(df)

    # 2) Impute missing categorical/text fields
    df = imputer(df)

    # 3) Infer missing manufacturer names from product_name
    df = fill_missing_manufacturer(df)

    # 4) Normalize and canonicalize manufacturer_name
    df = clean_manufacturer(df)

    # 5) Clean product_name (remove generic tags like 'Champú capilar')
    df = clean_product_name(df)

    # 6) Reorder columns: keep only ORDERED_COLS that actually exist
    ordered_present = [c for c in ORDERED_COLS if c in df.columns]
    if ordered_present:
        df = df[ordered_present]

    return df.reset_index(drop=True)
