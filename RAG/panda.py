import json

# Example list of texts
texts = ["Emmanuel is a boy who loves to play football.", "Another example sentence."]
embeddings = [get_embedding(text).tolist() for text in texts]

# Create a dictionary
data = [{"text": text, "embedding": embedding} for text, embedding in zip(texts, embeddings)]

# Save to JSON
with open("embeddings.json", "w") as f:
    json.dump(data, f)