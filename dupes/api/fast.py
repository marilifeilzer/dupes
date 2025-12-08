from fastapi import FastAPI
import pandas as pd
from dupes.logic import predict_shampoo
from dupes.model.descriptions_chromadb import embedding_description_query_chromadb, embedding_description_get_recommendation

app = FastAPI()

embedding_description_get_recommendation()


@app.get("/")
def index():
    return {"working": True}

@app.get("/recomend")
def get_recomendation(description: str):


    recomendation = embedding_description_query_chromadb(description)


    return recomendation
