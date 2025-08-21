import pandas as pd

import os
import openai
from scipy.spatial import distance
import plotly.express as px
from sklearn.cluster import KMeans
from umap import UMAP


from dotenv import load_dotenv
load_dotenv()

openai.api_key = os.getenv("OPENAI_API_KEY")

data_URL =  "https://raw.githubusercontent.com/keitazoumana/Experimentation-Data/main/Musical_instruments_reviews.csv"

review_df = pd.read_csv(data_URL)
review_df.head()


review_df.info()  # To get an overview of the dataset
review_df.describe()  # To get statistical summaries of numerical columns


def get_embedding(text_to_embed, model="text-embedding-ada-002"):
    response = openai.Embedding.create(
        model=model,
        input=text_to_embed
    )
    embedding = response["data"][0]["embedding"]
    return embedding

# Apply the updated function
review_df["embedding"] = review_df["reviewText"].astype(str).apply(get_embedding)


review_df = review_df.sample(100)
review_df["embedding"] = review_df["reviewText"].astype(str).apply(get_embedding)

# Make the index start from 0
review_df.reset_index(drop=True)

review_df.head(10)
