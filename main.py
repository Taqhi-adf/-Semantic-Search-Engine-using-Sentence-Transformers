# Semantic Search Engine
# Building an AI-powered Contextual Search APP
# Using sentence Transfomer-streamlit -scikit-learn

#!pip install sentence-transformers -q
import pandas as pd # for data manipulation and analysis
import numpy as np # for mathematical manipulations in Ai
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('all-MiniLM-L6-v2')

movies = [
    "Avengers is a superhero action movie",
    "Iron Man is a superhero movie",
    "Batman fights crime in Gotham",
    "Titanic is a romantic drama",
    "The Notebook is a love story",
    "Interstellar is a science fiction space movie",
    "The Martian is about survival in space"
]

embeddings = model.encode(movies)
print(embeddings)

print(embeddings.shape)

query = "space exploration movie"
# convert the query into an embedding
query_embedding = model.encode([query])
print(query_embedding)


scores = cosine_similarity(query_embedding,embeddings)
print(scores)


best_match = np.argmax(scores)
print("Query:",query)
print("Recommendation:",movies[best_match])


df = pd.DataFrame({
    'Movie':movies,
    'Similarity':scores[0]
})
print(df.sort_values(by='Similarity',ascending=False))