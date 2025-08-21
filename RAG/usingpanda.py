import pandas as pd

# Example list of texts
texts = ["Emmanuel is a boy who loves to play football.", "Another example sentence."]
embeddings = [get_embedding(text) for text in texts]

# Convert to a DataFrame
df = pd.DataFrame({
    "text": texts,
    "embedding": [embedding.tolist() for embedding in embeddings]
})

# Save to CSV
df.to_csv("embeddings.csv", index=False)