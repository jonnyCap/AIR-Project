#%% md
# # Retrieval System
# This notebook implementes the retrievel system
#%%
import joblib
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import os
import sys
current_directory = os.getcwd()
# print(current_directory)
# project_root = os.path.abspath(os.path.join(current_directory, "..", "..", "..", "..", ".."))
# print(project_root)
# sys.path.append(project_root)
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from ProjectPipeline import UserInterface
import spacy
import argparse
from RetrievalSystem import RetrievalSystem
# from ..ProjectPipeline import RetrievalAugmentedPredictionModel
from PredictionModel.RetrievalAugmentedPredictionModel_frontend import setRetrievalResult
from ProjectPipeline import UserInterface
IDEA_IDENTIFIER = "NEW_IDEA"

#%% md
# ### Creation of Embedding dataset
# We create this in order for faster execution in our final user pripeline
#%%
# Define paths relative to the current working director

def getTickers():
    return retrieval_result
    

if __name__ == '__main__':
    userinterface = UserInterface()
    
    parser = argparse.ArgumentParser(description="Find similar companies based on an idea.")
    parser.add_argument("--idea", type=str, required=True, help="The idea to search similar companies for.")
    parser.add_argument("--top_n", type=int, default=10, help="Number of similar entries to retrieve.")
    args = parser.parse_args()
    if not args.idea:
        raise ValueError("Please provide some text")
    # Retrieval
    retrieval_result = userinterface.retrieval_model.find_similar_entries_for_batch(texts=[args.idea], top_n=10)
    idea_embedding, retrieved_documents = retrieval_result[0]
    tickers = retrieved_documents["tickers"].values
    documents = userinterface.dataset.copy()
    documents = documents[documents["tickers"].isin(tickers)]
    documents = documents.merge(
    retrieved_documents[["tickers", "similarity", "embedding"]],  # Select only tickers and similarity columns
    on="tickers",  # Merge on the 'tickers' column
    how="left"  # Perform a left join to keep all rows in `documents`
    )
    new_idea_row = {
        'tickers': ['new_idea'],  # Ticker for the new idea
        'similarity': [0],  # Similarity for the new idea
        'business_description': [args.idea],  # The idea itself as the business description
        'embedding': [idea_embedding]  # The embedding of the idea
    }
    
    new_idea_df = pd.DataFrame(new_idea_row)
    documents = pd.concat([documents, new_idea_df], ignore_index=True)
    print(documents[['tickers','similarity', 'business_description','embedding'] + [f'month_{i}_performance' for i in range(1, 13)]].to_json(orient="records"))
