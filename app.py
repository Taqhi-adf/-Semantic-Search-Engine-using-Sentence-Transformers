import streamlit as st
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Page Configuration
st.set_page_config(
    page_title="Semantic Search Engine",
    page_icon="🔍",
    layout="wide"
)

# Title
st.title("🔍 Semantic Search Engine")
st.subheader("AI-Powered Contextual Search using Sentence Transformers")

# Load Model
@st.cache_resource
def load_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

model = load_model()

# Movie Database
movies = [
    "Avengers is a superhero action movie",
    "Iron Man is a superhero movie",
    "Batman fights crime in Gotham",
    "Titanic is a romantic drama",
    "The Notebook is a love story",
    "Interstellar is a science fiction space movie",
    "The Martian is about survival in space"
]

# Generate Embeddings
@st.cache_data
def generate_embeddings(movie_list):
    return model.encode(movie_list)

embeddings = generate_embeddings(movies)

# Sidebar
st.sidebar.header("About")
st.sidebar.info(
    """
    This application demonstrates Semantic Search using:

    - Sentence Transformers
    - Embeddings
    - Cosine Similarity
    - NLP Search

    Model:
    all-MiniLM-L6-v2
    """
)

# User Query
query = st.text_input(
    "Enter your search query:",
    placeholder="Example: space exploration movie"
)

# Search Button
if st.button("Search"):

    if query.strip() == "":
        st.warning("Please enter a query.")
    else:

        # Query Embedding
        query_embedding = model.encode([query])

        # Similarity Scores
        scores = cosine_similarity(query_embedding, embeddings)

        # Best Match
        best_match = np.argmax(scores)

        st.success("Best Recommendation")

        st.write("**Query:**", query)
        st.write("**Recommended Movie:**", movies[best_match])

        # Create DataFrame
        df = pd.DataFrame({
            "Movie": movies,
            "Similarity Score": scores[0]
        })

        df = df.sort_values(
            by="Similarity Score",
            ascending=False
        )

        st.subheader("Similarity Ranking")

        st.dataframe(
            df,
            use_container_width=True
        )

        # Top Results
        st.subheader("Top Matches")

        for index, row in df.head(5).iterrows():
            st.write(
                f"🎬 {row['Movie']}  |  Similarity: {row['Similarity Score']:.4f}"
            )

# Show Dataset
with st.expander("View Movie Dataset"):
    st.write(movies)

# Show Embedding Shape
st.subheader("Embedding Information")
st.write("Embedding Shape:", embeddings.shape)