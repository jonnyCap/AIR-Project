import torch.nn as nn
import sys
import torch
import argparse
import json
import os
FRONTEND = True
TEST = False

if FRONTEND:
    #same as in ipynb
    current_directory = os.getcwd()
    # print(current_directory)
    project_root = os.path.abspath(os.path.join(current_directory, "..", "..", "..", "..", ".."))
    # print(project_root)
    sys.path.append(project_root)

from RetrievalSystem.RetrievalSystem import RetrievalSystem
from PredictionModel.AttentionModel.AttentionModel import AttentionModel
import pandas as pd
from PredictionModel.Layers.Layers import SimilarityLayer, IdeaLayer, StaticFeatureLayer, HistoricalFeatureLayer, FirstFusionLayer, SecondFusionLayer, OutputLayer

INPUT_PATH = "../RetrievalSystem/Embeddings/embeddings.csv"

if FRONTEND:
    current_directory = os.path.dirname(os.path.abspath(__file__))
    parent_directory = os.path.dirname(current_directory)
    # print(parent_directory)
    INPUT_PATH = os.path.join(parent_directory,"RetrievalSystem","CompromisedEmbeddings","embeddings","embeddings.csv")
BERT_DIM = 768

class RetrievalAugmentedPredictionModel(nn.Module):
    def __init__(self, hidden_dim: int = 128, ret_sys: RetrievalSystem = None, static_dim = 34, historical_dim = 72, forecast_steps: int = 6, retrieval_number: int = 16):
        super(RetrievalAugmentedPredictionModel, self).__init__()

        self.forecast_steps = forecast_steps
        self.static_feature_dim = static_dim
        self.historical_feature_dim = historical_dim
        self.historical_idea_dim = historical_dim - forecast_steps
        self.retrieval_number = retrieval_number

        if ret_sys:
            self.retrieval_system = ret_sys
        else:
            self.retrieval_system = RetrievalSystem(INPUT_PATH, retrieval_number)

        # 16 * 768 -> 768 + 16
        self.attention_model = AttentionModel(input_dim=BERT_DIM, hidden_dim=hidden_dim)
        # 16 -> 32
        self.similarity_fc = SimilarityLayer(retrieval_number=retrieval_number)
        # Use Same bert model as for original embeddings
        # 768 -> 4 * 128 -> 128
        self.idea_fc = IdeaLayer(bert_dim=BERT_DIM, hidden_dim=hidden_dim)

        # Static feature layers (deep)
        # 34 * 16 -> 34 * 8 -> 256
        self.static_fc = StaticFeatureLayer(retrieval_number=retrieval_number,hidden_dim=hidden_dim, static_feature_dim=self.static_feature_dim)

        # Historical stock data layers (deep)
        # 72 * 16-> 72 * 8 -> 72 * 8 -> 512
        self.historical_fc = HistoricalFeatureLayer(retrieval_number=retrieval_number, hidden_dim=hidden_dim, historical_feature_dim=self.historical_feature_dim)

        # 34 -> 32
        self.idea_static_fc = nn.Linear(self.static_feature_dim, 32)
        # 72 -> 64
        self.idea_historical_fc = nn.Linear(self.historical_idea_dim, hidden_dim//2)

        # First Fustion Layer, combines:
        # 1. AttentionModel Output -> 768
        # 1.a Attention Scores -> retrievel_numbre (16)
        # 2. Combined Static Layer Output -> 256
        # 2. Combined Static Layer Output -> 512
        # 4. Cosine Simularity Layer -> 32
        # combined = 1184 -> 1024 -> 512
        self.first_fusion_fc = FirstFusionLayer(bert_dim=BERT_DIM, hidden_dim=hidden_dim, retrieval_number=retrieval_number)

        # Attention layer after first fusion
        self.fusion_attention = nn.MultiheadAttention(embed_dim=4 * hidden_dim, num_heads=4, batch_first=True)

        # Second Fusion Layer, combines:
        # 1. Previous Fusion Layer Output: 512
        # 2. Idea Embedding: 256 (ouput of idea layer)
        # 3. Idea Static: 32
        # 4. Idea Historical: 64
        # combined = 992 -> 1024
        self.second_fusion_fc = SecondFusionLayer(hidden_dim=hidden_dim)

        # Second fusion
        self.second_fusion_attention = nn.MultiheadAttention(embed_dim=4 * hidden_dim, num_heads=4, batch_first=True)

        # Multi-layer LSTM with residual connection
        self.lstm = nn.LSTM(4 * hidden_dim, 2 * hidden_dim, num_layers=10, batch_first=True, dropout=0.2)

        # Attention mechanism
        self.attention = nn.MultiheadAttention(embed_dim=2 * hidden_dim, num_heads=4, batch_first=True)

        # Output layer for forecasting
        self.output_fc = OutputLayer(hidden_dim=hidden_dim)


    def forward(self, idea, dataset: pd.DataFrame = None, static_features=None, historical_data=None, use_auxiliary_inputs=True, excluded_tickers=None):
        # Ensure device compatibility
        if excluded_tickers is None:
            excluded_tickers = []
        if dataset is None:
            # print("We need a dataset for retrieval")
            return None

        device = next(self.parameters()).device

        # Get Idea embedding and similar documents
        idea_embedding, retrieved_documents = self.retrieval_system.find_similar_entries(text=idea, top_n=self.retrieval_number, excluded_tickers=excluded_tickers)
        idea_embedding = torch.tensor(idea_embedding, dtype=torch.float32).to(device)

        # Extract embeddings and tickers from retrieved documents
        retrieved_idea_embeddings = retrieved_documents.loc[:, ["embedding"]].values
        # print("Combined embeddings shape: ", retrieved_idea_embeddings.shape)
        retrieved_idea_embeddings = torch.tensor(retrieved_idea_embeddings.tolist(), dtype=torch.float32).to(device)

        retrieved_similarities = retrieved_documents.loc[:, ["similarity"]].values.flatten()
        retrieved_similarities = torch.tensor(retrieved_similarities.tolist(), dtype=torch.float32).to(device)

        # Filter rows from the dataset where the index is in retrieved tickers
        retrieved_tickers = retrieved_documents.loc[:, ["tickers"]].values.flatten()  # Flatten to get a 1D array of tickers
        # print("Retrieved tickers: ", retrieved_tickers)
        dataset = dataset.set_index("tickers")
        filtered_data = dataset[dataset.index.isin(retrieved_tickers)]
        # print("We have these retrieved documents: ", filtered_data.shape)

        # Create a vector with all columns that are not "ticker", "business_description", or starting with "month"
        static_columns = [
            col for col in filtered_data.columns
            if col not in ["tickers", "business_description"] and not col.startswith("month")
        ]
        static_vector = filtered_data[static_columns].values.flatten()  # Convert to NumPy array

        # Create a second vector with all columns starting with "month"
        month_columns = [col for col in filtered_data.columns if col.startswith("month")]
        month_vector = filtered_data[month_columns].values.flatten()  # Convert to NumPy array

        # print(f"Shape of static vector: {static_vector.shape}, Shape of month vector: {month_vector.shape}")

        # Convert vectors to tensors and move to the appropriate device
        combined_static_tensor = torch.tensor(static_vector, dtype=torch.float32).to(device)
        combined_historical_tensor = torch.tensor(month_vector, dtype=torch.float32).to(device)

        # Initialize the historical data tensor (to handle shifting)
        if historical_data is not None and use_auxiliary_inputs:
            historical_tensor = historical_data.clone().to(device)
        else:
            historical_tensor = torch.zeros((1, self.historical_idea_dim), dtype=torch.float32).to(device)

        predictions = []
        for step in range(self.forecast_steps):
            # Put retrieved documents into appropriate input layers
            weighted_sum, attention_weights = self.attention_model(retrieved_idea_embeddings)
            attention_weights = attention_weights.view(1, -1)
            # print(f"Shape of weighted_sum: {weighted_sum.shape}, attention_weights: {attention_weights.shape}")

            similarity_output = self.similarity_fc(retrieved_similarities).unsqueeze(0)
            combined_static_output = self.static_fc(combined_static_tensor).unsqueeze(0)
            combined_historical_output = self.historical_fc(combined_historical_tensor).unsqueeze(0)
            # print(f"Shape of static_output: {combined_static_output.shape}, similarity: {similarity_output.shape}, historical: {combined_historical_output.shape}")

            # 1. FUSION LAYER - Fuse retrieval layers together
            combined_retrieval_input = torch.cat((weighted_sum, attention_weights, combined_static_output, combined_historical_output, similarity_output), dim=1)
            first_fusion_output = self.first_fusion_fc(combined_retrieval_input)

            # Attention layer
            first_fusion_attention_output, _ = self.fusion_attention(first_fusion_output, first_fusion_output, first_fusion_output)

            # Put new ideas data into input layers
            idea_output = self.idea_fc(idea_embedding)
            if use_auxiliary_inputs:
                static_output = self.idea_static_fc(static_features.to(device))
                historical_output = self.idea_historical_fc(historical_tensor)
            else:
                static_input = torch.zeros((1, self.static_feature_dim), dtype=torch.float32).to(device)
                static_output = self.idea_static_fc(static_input)
                historical_input = torch.zeros((1, self.historical_idea_dim), dtype=torch.float32).to(device)
                historical_output = self.idea_historical_fc(historical_input)

            # 2. FUSION LAYER - Fuse combined retrieval documents and new idea together
            # print(f"Shapes of static_output: {static_output.shape}, historical_output: {historical_output.shape}, idea: {idea_output.shape}, attention_output: {first_fusion_attention_output.shape}")
            combined_idea_input = torch.cat((first_fusion_attention_output, idea_output, static_output, historical_output), dim=1)
            second_fusion_output = self.second_fusion_fc(combined_idea_input)

            # Attention layer
            second_fusion_attention_output, _ = self.second_fusion_attention(second_fusion_output, second_fusion_output, second_fusion_output)

            # LSTM
            lstm_output, _ = self.lstm(second_fusion_attention_output.unsqueeze(1))  # Add sequence dimension

            # Attention
            lstm_attention_output, _ = self.attention(lstm_output, lstm_output, lstm_output)

            # OUTPUT
            final_prediction = self.output_fc(lstm_attention_output.squeeze(1))  # Remove sequence dimension

            # Append to predictions
            predictions.append(final_prediction)

            # Update historical tensor for next step
            # print(f"Final prediction: {final_prediction.shape}, historical tensor: {historical_tensor.shape}")
            historical_tensor = torch.cat((historical_tensor[:, 1:], final_prediction), dim=1)
            # print(f"Resulting historical tensor shape: {historical_tensor.shape}")

        # Stack predictions into a single tensor
        predictions = torch.stack(predictions, dim=1)  # Shape: [1, forecast_steps, 1]
        predictions = predictions.squeeze(-1)  # Remove the last dimension, Shape: [1, forecast_steps]
        return predictions

if __name__ == '__main__':
    # Initialize the model - HAVE TO BE ADAPTED TO DATASET (Values are likely correct)
    static_feature_dim_num = 4    # Number of static features
    historical_dim_num = 12       # Number of historical stock performance points
    hidden_dim_num = 128          # Hidden layer size
    forecast_steps_num = 12       # Predict next 12 months

    DATASET_PATH = "../Dataset/Data/normalized_real_company_stock_dataset_large.csv"

    if FRONTEND:
        current_directory = os.path.dirname(os.path.abspath(__file__))
        parent_directory = os.path.dirname(current_directory)

        # Now define the dataset path by joining the parent directory with the relative path
        DATASET_PATH = os.path.join(parent_directory, "Dataset", "Data", "normalized_real_company_stock_dataset_large.csv")
    
    dataset = pd.read_csv(DATASET_PATH)

    retrieval_system = RetrievalSystem(INPUT_PATH, retrieval_number=16)

    model = RetrievalAugmentedPredictionModel(
        forecast_steps=forecast_steps_num,
        ret_sys = retrieval_system,
        retrieval_number=16
    )

    current_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Random entry
    idea_entry = dataset.iloc[4000, :]  # Get a random row (Series)
    if TEST:
    
        idea = idea_entry["business_description"]

        static_columns = [
            col for col in dataset.columns
            if col not in ["tickers", "business_description"] and not col.startswith("month")
        ]
        month_columns = [col for col in dataset.columns if col.startswith("month")]

        static_data = idea_entry[static_columns]
        historical_data = idea_entry[month_columns]

        # Ensure numeric data and handle missing values
        static_data = static_data.apply(pd.to_numeric, errors='coerce').fillna(0).values.astype(float)
        historical_data = historical_data.apply(pd.to_numeric, errors='coerce').fillna(0).values.astype(float)

        # Convert to tensors
        static_data = torch.tensor(static_data, dtype=torch.float32).unsqueeze(0).to(current_device)
        historical_data = torch.tensor(historical_data[forecast_steps_num:], dtype=torch.float32).unsqueeze(0).to(current_device)

        # Make a prediction
        prediction = model(
            idea=idea,
            dataset=dataset,
            static_features=static_data,
            historical_data=historical_data,
            use_auxiliary_inputs=True
        )
        print(prediction.detach().cpu().numpy())  # Co


        # Make a prediction
        prediction = model(
            idea=idea,
            dataset=dataset,
            use_auxiliary_inputs=False
        )
        print(prediction.detach().cpu().numpy())  # Co
    if FRONTEND:
        parser = argparse.ArgumentParser(description="Find stock perfomance based on an idea.")
        parser.add_argument("--idea", type=str, required=True, help="The idea to search similar companies for.")
        args = parser.parse_args()
        idea = args.idea
        static_columns = [  
            col for col in dataset.columns  
            if col not in ["tickers", "business_description"] and not col.startswith("month")   
        ]   
        month_columns = [col for col in dataset.columns if col.startswith("month")] 
        static_data = idea_entry[static_columns]    
        historical_data = idea_entry[month_columns] 
        # Ensure numeric data and handle missing values 
        static_data = static_data.apply(pd.to_numeric, errors='coerce').fillna(0).values.astype(float)  
        historical_data = historical_data.apply(pd.to_numeric, errors='coerce').fillna(0).values.astype(float)  
        # Convert to tensors    
        static_data = torch.tensor(static_data, dtype=torch.float32).unsqueeze(0).to(current_device)    
        historical_data = torch.tensor(historical_data[forecast_steps_num:], dtype=torch.float32).unsqueeze(0).to(current_device)   
        prediction = model(
            idea=idea,
            dataset=dataset,
            use_auxiliary_inputs=False
        )
        prediction_numpy = prediction.detach().cpu().numpy()

        # If it's a single value, you can directly convert it
        if prediction_numpy.ndim == 0:  # If it's a scalar (e.g., a single number)
            prediction_list = prediction_numpy.item()  # Convert to a Python scalar
        else:  # If it's an array
            prediction_list = prediction_numpy.tolist()  # Convert to a list

        # Convert the list (or scalar) to JSON
        prediction_json = json.dumps(prediction_list)

        # Print the JSON output
        print(prediction_json)

        
        

