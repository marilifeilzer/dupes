from transformers import pipeline
import pandas as pd

df = pd.read_csv()

summarizer = pipeline(task="summarization")

summary = df["description"].map()
