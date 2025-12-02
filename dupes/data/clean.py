import pandas as pd
import re
import numpy as np

def retrieve_formula_ingredients(df_keys: pd.DataFrame, df: pd.DataFrame, col="ingredients_text"):
    # df_keys= pd.read_csv('/home/marili/code/marilifeilzer/dupes/raw_data/products_data_V2.csv')
    # df_keys.head()
    df_keys.name= df_keys.name.apply(lambda x: x.lower())

    dict_key_formula = {}


    for index, row in df_keys[["name", "formula"]].iterrows():
        dict_key_formula.update({row['name']: row["formula"]})

    df["ingredients_raw"] = df[col].apply(
        lambda x: re.sub(r'[^\w\s\-]+', ',', str(x)).split(',')
    )


    df["ingredients_raw"] = df["ingredients_raw"].apply(
        lambda lst: [i.strip().lower() for i in lst if i.strip() != ""]
    )

    df["ingredients_formula"] = df["ingredients_raw"].apply(
        lambda lst: [dict_key_formula.get(i) for i in lst]
    )

    df.ingredients_formula = df.ingredients_formula.apply(set)
    df.ingredients_formula = df.ingredients_formula.apply(list)
    df.ingredients_formula = df.ingredients_formula.apply(
        lambda x: [item for item in x if item not in ['not an element', 'not a chemical']]
    )

    df.ingredients_formula = df.ingredients_formula.apply(
    lambda x: (
        np.nan
        if isinstance(x, list) and x == [None]
        else [item for item in x if item is not None] if isinstance(x, list) else x
    )
    )

    return df



if __name__ == "__main__":
    df = pd.read_csv('/home/marili/code/marilifeilzer/dupes/raw_data/products_raw_600.csv')
    df_keys= pd.read_csv('/home/marili/code/marilifeilzer/dupes/raw_data/products_dict.csv')
    result = retrieve_formula_ingredients(df_keys=df_keys, df=df, col="ingredients_text")
    output_path = "/home/marili/code/marilifeilzer/dupes/raw_data/products_600_v5.csv"
    result.to_csv(output_path, index=False)

    print(result)
    print(f"Saved to: {output_path}")
