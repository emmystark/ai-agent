import os
import openai
from scipy.spatial import distance
import plotly.express as px
from sklearn.cluster import KMeans
from umap import UMAP



def get_embedding(text_to_embed):
	# Embed a line of text
	response = openai.Embedding.create(
    	model= "text-embedding-ada-002",
    	input=[text_to_embed]
	)
	# Extract the AI output embedding as a list of floats
	embedding = response["data"][0]["embedding"]
    
	return embedding