#%% md
# # Retrieval System
# This notebook implementes the retrievel system
#%%
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import os
import spacy

BERT_ENCODING_SIZE = 768

class RetrievalSystem:
    def __init__(self, path: str, retrieval_number: int = 16):
        """
        Constructor to initialize the RetrievalSystem with a CSV file.
        Args:
            path (str): The path to the CSV file to load.
        """
        self.model_type = 'bert-base-nli-mean-tokens'
        self.retrieval_number = retrieval_number

        if os.path.exists(path):
            self.data = pd.read_csv(path)
        self.model = SentenceTransformer(self.model_type)
        self.nlp = spacy.load("en_core_web_sm")  # Load spaCy for preprocessing

    def preprocess_text(self, text: str) -> str:
        """
        Preprocesses the input text by removing stop words and applying lemmatization.
        Args:
            text (str): The text to preprocess.
        Returns:
            str: The preprocessed text.
        """
        doc = self.nlp(text)
        # Remove stop words and punctuation, and apply lemmatization
        preprocessed_text = " ".join(
            [token.lemma_ for token in doc if not token.is_stop and not token.is_punct]
        )
        return preprocessed_text

    def find_similar_entries(self, text: str, top_n: int = None, excluded_tickers=None):
        """
        Embeds the input text using BERT, compares it with the entries in the CSV file,
        and returns the most similar entries based on cosine similarity.
        Args:
            text (str): The input text to embed and compare.
            top_n (int): The number of most similar entries to return.
            excluded_tickers (list): List of tickers to exclude from similarity checks.
        Returns:
            pd.DataFrame: The top-n most similar entries from the CSV.
        """
        # Preprocess the input text
        text = self.preprocess_text(text)

        if not top_n:
            top_n = self.retrieval_number

        # Generate embedding for the preprocessed text
        input_embedding = self.model.encode([text])

        # Load embeddings from the CSV
        if 'embedding' not in self.data.columns:
            raise ValueError("The CSV file must have an 'embedding' column.")

        # Create a copy of self.data to work with
        copied_data = self.data.copy()

        # Exclude rows with tickers in excluded_tickers
        if excluded_tickers:
            copied_data = copied_data[~copied_data['tickers'].isin(excluded_tickers)]

        # Convert strings to lists only if they are strings
        if isinstance(copied_data['embedding'].iloc[0], str):
            copied_data['embedding'] = copied_data['embedding'].apply(eval)

        embeddings = copied_data['embedding'].tolist()

        # Compute cosine similarities
        similarities = cosine_similarity(input_embedding, embeddings)[0]
        copied_data['similarity'] = similarities

        # Sort by similarity and return the top N results
        return input_embedding, copied_data.sort_values(by='similarity', ascending=False).head(top_n)


    def process_and_save_embeddings(self, path: str, output_path: str):
        """
        Embeds the 'business_description' column from a new CSV file, keeps only 'tickers' and 'embedding',
        and saves the results in a new CSV with 'tickers' as the index.
        Args:
            path (str): The path to the CSV file to process.
            output_path (str): The path to save the output CSV.
        """
        # Load new data
        new_data = pd.read_csv(path)

        # Ensure required columns exist
        if 'tickers' not in new_data.columns:
            raise ValueError("The CSV file must have a 'tickers' column.")
        if 'business_description' not in new_data.columns:
            raise ValueError("The CSV file must have a 'business_description' column.")

        # Preprocess and embed the 'business_description' column
        new_data['processed_description'] = new_data['business_description'].apply(self.preprocess_text)
        new_data['embedding'] = new_data['processed_description'].apply(lambda x: self.model.encode([x])[0].tolist())

        # Keep only 'tickers' and 'embedding' columns
        processed_data = new_data[['tickers', 'embedding']]

        # Set 'tickers' as the index
        processed_data.set_index('tickers', inplace=True)

        # Save the processed data
        processed_data.to_csv(output_path)

#%% md
# ### Creation of Embedding dataset
# We create this in order for faster execution in our final user pripeline
#%%
# Define paths relative to the current working directory
INPUT_PATH = "../Dataset/Data/normalized_real_company_stock_dataset_large.csv"
OUTPUT_PATH = "Embeddings/embeddings.csv"

CREATE_DATASET = False
TEST = True

if __name__ == '__main__':
    if CREATE_DATASET:
        retrieval_system = RetrievalSystem(OUTPUT_PATH)
        retrieval_system.process_and_save_embeddings(INPUT_PATH, OUTPUT_PATH)

    if TEST:
        retrieval_system = RetrievalSystem(OUTPUT_PATH)
        idea = "Hello world program that can print hello world"
        idea = "American Assets Trust, Inc. is a full service, vertically integrated and self-administered real estate investment trust ('REIT'), headquartered in San Diego, California. The company has over 55 years of experience in acquiring, improving, developing and managing premier office, retail, and residential properties throughout the United States in some of the nation's most dynamic, high-barrier-to-entry markets primarily in Southern California, Northern California, Washington, Oregon, Texas and Hawaii. The company's office portfolio comprises approximately 4.1 million rentable square feet, and its retail portfolio comprises approximately 3.1 million rentable square feet. In addition, the company owns one mixed-use property (including approximately 94,000 rentable square feet of retail space and a 369-room all-suite hotel) and 2,110 multifamily units. In 2011, the company was formed to succeed to the real estate business of American Assets, Inc., a privately held corporation founded in 1967 and, as such, has significant experience, long-standing relationships and extensive knowledge of its core markets, submarkets and asset classes."
        result = retrieval_system.find_similar_entries(idea, 10)
        print(result)