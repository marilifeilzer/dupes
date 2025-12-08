from fastapi import FastAPI
import pandas as pd
from dupes.logic import predict_shampoo
from dupes.model.descriptions_chromadb import embedding_description_query_chromadb, embedding_description_get_recommendation

app = FastAPI()
df = pd.read_csv("/Users/lewagon/code/marilifeilzer/dupes/raw_data/products_data__0412.csv")

@app.get("/")
def index():
    return {"working": True}

@app.get("/recomend")
def get_recomendation(description: str):


    recomendation = embedding_description_query_chromadb(description)


    return recomendation
