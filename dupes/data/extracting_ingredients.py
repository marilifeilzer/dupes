import re
import pandas as pd
import tqdm
import openai
from openai import OpenAI
import json
import os


API_KEY_OPENAI= os.getenv('API_KEY_OPENAI')
client = OpenAI(api_key=API_KEY_OPENAI)

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
    names = valuesdf.index
    return names


def require_element_name(names: list):


    get_matches_declaration = [{
    "type": "function",
    "name": "get_ingredients",
    "description": "Get  molecular formula if its value exits if not return 'not an element'.",
    "parameters": {
        "type": "object",
        "properties": {
            "formula": {
                "type": "string",
                "description": "The molecular formula of an element if exists otherwise return 'not an element''",
            },
            "active": {
                "type": "boolean",
                "description": "True if elements is active in cosmetic products.",
            },
        },

    },
    }]

    result= []

    for name in tqdm.tqdm(names):
        input_list = [
        {"role": "user", "content": f"Value is: {name}"}
        ]

        try:

            response = client.responses.create(
            model="gpt-4.1",
            instructions = """
                    You are a chemistry tool.

                    Your job is to convert a given common name into a correct chemical formula.
                    Always respond using the get_ingredients tool.

                    Rules:
                        1. If the input is a chemical element (e.g., oxygen, carbon, iron):
                        - Return the standard atomic symbol (O, C, Fe).
                        - active: if it is an active ingredient in shampoo return True

                        2. If the input is a chemical compound with a known common name:
                        - Return the correct molecular formula (e.g., water -> H2O, salt -> NaCl, ethanol -> C2H6O).
                        - active: if it is an active ingredient in shampoo return True

                        3. If the input is NOT a chemical (e.g., car, table, banana):
                        - formula: "not a chemical"
                        - active: false

                        4. Never echo the input text as the formula.
                        5. Always call the get_ingredients tool.
                    """,

            tools=get_matches_declaration,
            input=input_list,
            )


            json_respons = json.loads(response.output[0].arguments)

            result.append({
                "name": name,
                "formula": json_respons.get("formula"),
                "active": json_respons.get("active"),
            })
        except:
            print(f'Failed for {name}')



    df = pd.DataFrame(result)

    return df

# if __name__ == "__main__":
#     df = pd.read_csv('/home/marili/code/marilifeilzer/dupes/raw_data/products_raw_500.csv')

#     names = getting_distribution_ingredients(df)
#     result = require_element_name(names=names)

#     # Save result to raw data folder
#     output_path = "/home/marili/code/marilifeilzer/dupes/raw_data/products_data_V2.csv"
#     result.to_csv(output_path, index=False)

#     print(result)
#     print(f"Saved to: {output_path}")
