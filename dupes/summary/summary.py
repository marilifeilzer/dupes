from openai import OpenAI
import os
import pandas as pd
from tqdm import tqdm
tqdm.pandas()
df = pd.read_csv("/Users/jonamoram/code/marilifeilzer/dupes/raw_data/data_0812.csv")

descriptions_df = df[["product_id","description"]]

API_KEY_OPENAI= os.getenv('API_KEY_OPENAI')
client = OpenAI(
  api_key=API_KEY_OPENAI
)

def translator(text:str):

    return client.responses.create(
  model="gpt-5-nano",
  input=f"translate the following text {text} and then sumarize it, return just the summary",
  store=True,
            ).output_text


descriptions_df["en_description"] = descriptions_df["description"].progress_apply(lambda text: translator(text))

translated_df = descriptions_df.drop(columns=["description"])

translated_df.to_csv("data_translated.csv")
