#%% md
# # Retrieval System
# This notebook implementes the retrievel system
#%%
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import os
import spacy
import argparse
BERT_ENCODING_SIZE = 768

class RetrievalSystem:
    def __init__(self, path: str, retrieval_number: int = 16):
        """
        Constructor to initialize the RetrievalSystem with a CSV file.
        Args:
            path (str): The path to the CSV file to load.
        """
        self.model_type = 'all-MiniLM-L6-v2'
        self.retrieval_number = retrieval_number

        if os.path.exists(path):
            self.data = pd.read_csv(path)
        self.model = SentenceTransformer(self.model_type)
        self.nlp = spacy.load("en_core_web_sm")  # Load spaCy for preprocessing

    def encode(self, text:str):
        preprocessed_text = self.preprocess_text(text)
        return self.model.encode(preprocessed_text)

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

    def find_similar_entries_for_batch(self, texts: list, top_n: int = None, excluded_tickers: dict = None):
        """
        Embeds a batch of texts and finds the most similar entries in the dataset for each.
        Args:
            texts (list): List of input texts to embed and compare.
            excluded_tickers (dict): Dictionary where each key corresponds to the index of a text, and each value
                                     is a list of tickers to exclude for that text.
        Returns:
            list: A list of tuples containing embeddings and DataFrames for each text.
        """

        if not top_n:
            top_n = self.retrieval_number

        # Preprocess all texts
        processed_texts = [self.preprocess_text(text) for text in texts]

        # Generate embeddings for all input texts as a batch
        input_embeddings = self.model.encode(processed_texts)

        # Prepare the dataset
        if 'embedding' not in self.data.columns:
            raise ValueError("The CSV file must have an 'embedding' column.")

        copied_data = self.data.copy()

        # Convert embeddings column to lists if necessary
        if isinstance(copied_data['embedding'].iloc[0], str):
            copied_data['embedding'] = copied_data['embedding'].apply(eval)

        embeddings = copied_data['embedding'].tolist()
        dataset_embeddings = torch.tensor(embeddings, dtype=torch.float32)

        # Compute cosine similarity for all input embeddings
        input_embeddings = torch.tensor(input_embeddings, dtype=torch.float32)
        similarities = torch.matmul(input_embeddings, dataset_embeddings.T)  # Efficient batch cosine similarity

        # Collect top-N similar entries for each input text
        # Collect top-N similar entries for each input text
        results = []
        for i, sim in enumerate(similarities):
            # Add similarity scores directly to copied_data
            copied_data['similarity'] = sim.numpy()  # Replace or add similarity column

            # Filter the dataset for this text
            if excluded_tickers and i in excluded_tickers:
                excluded = excluded_tickers[i]
                filtered_data = copied_data[~copied_data['tickers'].isin(excluded)]
            else:
                filtered_data = copied_data

            # Sort and get top-N similar entries
            top_results = filtered_data.sort_values(by='similarity', ascending=False).head(top_n)

            # Append results
            results.append((input_embeddings[i].numpy(), top_results))

        return results

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
# print(INPUT_PATH)
OUTPUT_PATH = "Embeddings/embeddings.csv"
CREATE_DATASET = False
TEST = False
FRONTEND = True

if __name__ == '__main__':
    if CREATE_DATASET:
        retrieval_system = RetrievalSystem(OUTPUT_PATH)
        retrieval_system.process_and_save_embeddings(INPUT_PATH, OUTPUT_PATH)

    if TEST:
        retrieval_system = RetrievalSystem(OUTPUT_PATH)
        own_idea = "Hello world program that can print hello world"
        idea = "American Assets Trust, Inc. is a full service, vertically integrated and self-administered real estate investment trust ('REIT'), headquartered in San Diego, California. The company has over 55 years of experience in acquiring, improving, developing and managing premier office, retail, and residential properties throughout the United States in some of the nation's most dynamic, high-barrier-to-entry markets primarily in Southern California, Northern California, Washington, Oregon, Texas and Hawaii. The company's office portfolio comprises approximately 4.1 million rentable square feet, and its retail portfolio comprises approximately 3.1 million rentable square feet. In addition, the company owns one mixed-use property (including approximately 94,000 rentable square feet of retail space and a 369-room all-suite hotel) and 2,110 multifamily units. In 2011, the company was formed to succeed to the real estate business of American Assets, Inc., a privately held corporation founded in 1967 and, as such, has significant experience, long-standing relationships and extensive knowledge of its core markets, submarkets and asset classes."
        result = retrieval_system.find_similar_entries(idea, 10)
        result_batch = retrieval_system.find_similar_entries_for_batch(texts=[idea, idea], top_n=10, excluded_tickers={0: ["AAT", "SVC"], 1: []})
        print(result_batch)
    if FRONTEND:
        project_root = os.path.dirname(os.path.abspath(__file__))  # Directory of the script
        OUTPUT_PATH = os.path.join(project_root,"CompromisedEmbeddings", "embeddings.csv")
        INPUT_PATH = os.path.join(project_root,"..","Dataset","Data","normalized_real_company_stock_dataset_large.csv")
        dataset = pd.read_csv(INPUT_PATH)
        
        retrieval_system = RetrievalSystem(OUTPUT_PATH)
        parser = argparse.ArgumentParser(description="Find similar companies based on an idea.")
        parser.add_argument("--idea", type=str, required=True, help="The idea to search similar companies for.")
        parser.add_argument("--top_n", type=int, default=10, help="Number of similar entries to retrieve.")
        args = parser.parse_args()
        _, similar_entries = retrieval_system.find_similar_entries(text=args.idea, top_n=args.top_n)
        # Extract tickers from similar_entries
        tickers = similar_entries['tickers'].tolist()
        # print(type(dataset))

        # Retrieve the business descriptions for the tickers
        descriptions = dataset[dataset['tickers'].isin(tickers)][['tickers', 'business_description']]
        descriptions_dict = descriptions.set_index('tickers')['business_description'].to_dict()
        
        # Add descriptions to similar_entries
        similar_entries['business_description'] = similar_entries['tickers'].map(descriptions_dict)
        # Select the relevant columns for performance data
        performance_columns = ['tickers'] + [f'month_{i}_performance' for i in range(1, 25)]
        performance_data = dataset[performance_columns]

        # Create a dictionary with tickers as keys and performance data as values
        performance_dict = performance_data.set_index('tickers').to_dict(orient='index')

        # Add business description and performance data to similar_entries
        similar_entries['business_description'] = similar_entries['tickers'].map(descriptions_dict)

        # Add month_1_performance to month_12_performance to similar_entries
        for i in range(1, 25):
            similar_entries[f'month_{i}_performance'] = similar_entries['tickers'].map(lambda x: performance_dict.get(x, {}).get(f'month_{i}_performance', None))

        # Print the result
        print(similar_entries[['tickers', 'similarity', 'business_description'] + [f'month_{i}_performance' for i in range(1, 25)]].to_json(orient="records"))
