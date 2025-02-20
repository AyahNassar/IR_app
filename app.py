import streamlit as st
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Load precomputed document embeddings and documents
embeddings = np.load("embeddings.npy")  # Ensure this file exists
with open("documents.txt", "r", encoding="utf-8") as f:
    documents = f.readlines()


# Function to compute similarity
def retrieve_top_k(query_embedding, embeddings, k=10):
    """Retrieve top-k most similar documents using cosine similarity."""
    similarities = cosine_similarity(query_embedding.reshape(1, -1), embeddings)[0]
    top_k_indices = similarities.argsort()[-k:][::-1]
    return [(documents[i], similarities[i]) for i in top_k_indices]


# Function to convert query to embedding (Replace this with your actual embedding model)
def get_query_embedding(query):
    return np.random.rand(embeddings.shape[1])  # Placeholder


# Streamlit UI
st.title("Information Retrieval using Document Embeddings")
query = st.text_input("Enter your query:")

if st.button("Search"):
    query_embedding = get_query_embedding(query)
    results = retrieve_top_k(query_embedding, embeddings)

    st.write("### Top 10 Relevant Documents:")
    for doc, score in results:
        st.write(f"- **{doc.strip()}** (Score: {score:.4f})")
