import re
import pandas as pd

# import openai
# from openai import OpenAI
# import json
# import os
# from dotenv import load_dotenv


import pandas as pd
import re

def getting_distribution_ingredients(df: pd.DataFrame, col="ingredients_text", distribution_percentage=0.9):

    # Drop rows with null values
    df = df.dropna(subset=[col])

    # Clean and split ingredients
    df["ingredients_raw"] = df[col].apply(lambda x: re.sub(r'[^\w\s\-]+', '/n', x))

    # Split into lists
    ingredients_doubles = []
    for i in range(len(df["ingredients_raw"])):
        ingredients_doubles.append(df["ingredients_raw"].iloc[i].split('/n'))

    # Flatten list
    ingredients_all_double = []
    for x in ingredients_doubles:
        ingredients_all_double.extend(x)

    # Strip whitespace
    ingredients_all_double_stripped = []
    for each in ingredients_all_double:
        stripped = each.strip()
        if stripped != "":
            ingredients_all_double_stripped.append(stripped)

    # Create dataframe for counting
    dfnames = pd.DataFrame()
    dfnames["names"] = ingredients_all_double_stripped

    # Frequency calculation
    valuesdf = pd.DataFrame()
    valuesdf["names"] = dfnames.names.value_counts(normalize=True)

    # Cumulative sum
    valuesdf["csum"] = valuesdf["names"].cumsum()

    # Filter by % coverage
    valuesdf = valuesdf.loc[valuesdf["csum"] < distribution_percentage]

    # Return names as list
    return valuesdf.index


def df_non_active_ingredients():
    i
