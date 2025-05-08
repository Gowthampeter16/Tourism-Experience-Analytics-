import streamlit as st
import pandas as pd

# Load pickled data
df_enc = pd.read_pickle("df_encoded.pkl")
user_item_matrix = pd.read_pickle("user_item_matrix.pkl")
user_similarity_df = pd.read_pickle("user_similarity_df.pkl")

# Define recommendation function
def recommend_attractions(user_id, top_n=5):
    if user_id not in user_item_matrix.index:
        return []

    sim_users = user_similarity_df[user_id].sort_values(ascending=False)[1:6]

    weighted_scores = pd.Series(dtype='float64')
    for other_user, similarity in sim_users.items():
        weighted_scores = weighted_scores.add(user_item_matrix.loc[other_user] * similarity, fill_value=0)

    visited = user_item_matrix.loc[user_id]
    recommendations = weighted_scores[visited == 0].sort_values(ascending=False).head(top_n)

    return recommendations.reset_index().rename(columns={0: "Score", "AttractionId": "Recommended Attraction"})
