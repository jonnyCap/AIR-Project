#%% md
# # Attention Optimized Retrieval Augmented Prediction Model
# 
# This Model, specifically created to make Stock Predictions for upcoming Businesses, means this model predicts the market startup of any new business idea.
# 
#%%
import torch.nn as nn
from RetrievalSystem.RetrievalSystem import RetrievalSystem
import pandas as pd
import numpy as np
from PredictionModel.Layers.AttentionOptimizedLayers import IdeaLayer, IdeaStaticLayer, IdeaHistoricalLayer, OutputLayer, FirstFusionLayer, SecondFusionLayer

INPUT_PATH = "../RetrievalSystem/Embeddings/embeddings.csv"
pd.set_option('display.max_columns', None)
BERT_DIM = 384

class RetrievalAugmentedPredictionModel(nn.Module):
    def __init__(self, hidden_dim: int = 128, ret_sys: RetrievalSystem = None, static_dim = 34, historical_dim = 72, forecast_steps: int = 6, retrieval_number: int = 16):
        super(RetrievalAugmentedPredictionModel, self).__init__()

        if forecast_steps % 3 != 0:
            raise ValueError("forecast_steps must be a multiple of 3")

        self.forecast_steps = forecast_steps
        self.static_feature_dim = static_dim
        self.historical_feature_dim = historical_dim
        self.historical_idea_dim = forecast_steps
        self.retrieval_number = retrieval_number

        # Retrieval Model
        if ret_sys:
            self.retrieval_system = ret_sys
        else:
            self.retrieval_system = RetrievalSystem(INPUT_PATH, retrieval_number)

        # Layers for new Idea
        self.idea_fc = IdeaLayer(bert_dim=BERT_DIM, hidden_dim=hidden_dim)
        self.idea_static_fc = IdeaStaticLayer(static_feature_dim=self.static_feature_dim)
        self.idea_historical_fc = IdeaHistoricalLayer(historical_idea_dim=self.historical_idea_dim, hidden_dim=hidden_dim)

        self.document_fusion_fc = FirstFusionLayer(input_dim=self.historical_feature_dim + self.static_feature_dim + BERT_DIM + 1, hidden_dim=hidden_dim)

        self.idea_fusion_fc = SecondFusionLayer(hidden_dim=hidden_dim)

        # Attention mechanism
        self.pre_attention = nn.MultiheadAttention(embed_dim=2 * hidden_dim, num_heads=4, batch_first=True)

        # Multi-layer LSTM with residual connection
        self.lstm = nn.LSTM(input_size=2*hidden_dim, hidden_size=2*hidden_dim, num_layers=4, batch_first=True, dropout=0.2)

        # Attention mechanism
        self.post_attention = nn.MultiheadAttention(embed_dim=2 * hidden_dim, num_heads=2, batch_first=True)

        # Output layer for forecasting
        self.output_fc = OutputLayer(hidden_dim=2*hidden_dim, retrieval_number=self.retrieval_number)


    def forward(self, ideas: list=None, retrieval_result=None, dataset: pd.DataFrame = None, static_features=None, historical_data=None, use_auxiliary_inputs=True, excluded_tickers: dict = None):
        # Ensure device compatibility

        if excluded_tickers is None:
            excluded_tickers = {}

        if dataset is None:
            print("We need a dataset for retrieval")
            return None

        if not ideas and not retrieval_result:
            print("We need either an idea text or a retrieval result")
            return None

        device = next(self.parameters()).device

        # --- Retrieval Model ---
        # Batch retrieve embeddings and documents
        if not retrieval_result:
            retrieval_result = self.retrieval_system.find_similar_entries_for_batch(texts=ideas, top_n=self.retrieval_number, excluded_tickers=excluded_tickers)

        # Define static and month columns
        static_columns = [
            col for col in dataset.columns
            if col not in ["tickers", "business_description", "embedding", "similarity"] and not col.startswith("month")
        ]
        month_columns = [col for col in dataset.columns if col.startswith("month")]

        # Extract embeddings, similarities, and tickers for the batch
        idea_embeddings, retrieved_embeddings, combined_data = [], [], []

        for embedding, documents in retrieval_result:
            idea_embeddings.append(embedding)

            # Convert documents to a DataFrame if necessary
            if isinstance(documents, list):
                documents = pd.DataFrame(documents)

            # Ensure `tickers` column has the same type in both DataFrames
            documents['tickers'] = documents['tickers'].astype(str)
            if dataset.index.name == 'tickers':
                dataset = dataset.reset_index()
            dataset['tickers'] = dataset['tickers'].astype(str)

            # Join dataset on `tickers`
            joined_data = documents.join(dataset.set_index('tickers'), on='tickers', how='left')

            # Convert embedding and similarity columns to PyTorch tensors
            embeddings_tensor = torch.stack(
                [torch.tensor(e, dtype=torch.float32) for e in joined_data['embedding']],
                dim=0
            ).to(device)  # Shape: [num_documents, embedding_dim]

            similarities_tensor = torch.stack(
                [torch.tensor(s, dtype=torch.float32) for s in joined_data['similarity']],
                dim=0
            ).to(device)  # Shape: [num_documents, similarity_dim]

            # Drop `embedding` and `similarity` columns
            joined_data = joined_data.drop(columns=['embedding', 'similarity'])

            # Select and process static and month columns
            numeric_data = joined_data[static_columns + month_columns].apply(pd.to_numeric, errors='coerce').fillna(0).values
            numeric_tensor = torch.tensor(numeric_data, dtype=torch.float32).to(device)  # Shape: [num_documents, static_dim + month_dim]

            # Concatenate tensors along feature dimension (dim=1)
            print("Embeddings Tensor Shape:", embeddings_tensor.shape)
            print("Similarities Tensor Shape:", similarities_tensor.shape)
            print("Numeric Tensor Shape:", numeric_tensor.shape)

            combined_tensor = torch.cat((embeddings_tensor, similarities_tensor.unsqueeze(1), numeric_tensor), dim=1)  # Shape: [num_documents, total_feature_dim]
            combined_data.append(combined_tensor)


        # Convert to tensors for further processing
        combined_tensor = torch.stack(combined_data, dim=0)  # Shape: [batch_size, sequence_length, feature_dim]

        combined_output = self.document_fusion_fc(combined_tensor)

        print(f"Shape of combined_tensor: {combined_output.shape}")

        # Put new ideas data into input layers
        idea_embeddings = torch.tensor(np.array(idea_embeddings, dtype=np.float32), dtype=torch.float32).to(device).squeeze(1)
        idea_output = self.idea_fc(idea_embeddings)

        batch_size = idea_embeddings.size(0)
        if use_auxiliary_inputs:
            static_tensor = static_features.clone().to(torch.float32).to(device)
            historical_tensor = historical_data.clone().to(device)
        else:
            static_tensor = torch.zeros((batch_size, self.static_feature_dim), dtype=torch.float32).to(device)
            historical_tensor = torch.zeros((batch_size, self.historical_idea_dim), dtype=torch.float32).to(device)

        static_output = self.idea_static_fc(static_tensor) # This wont change within the autoregressiv prediction

        # --- Autoregressive prediction ---
        predictions = []
        pre_attention_weights = []
        post_attention_weights = []
        lstm_hidden_states = []  # To store the second output (hidden states) of the LSTM

        for step in range(self.forecast_steps // 3):  # Predict 3 steps at a time
            historical_output = self.idea_historical_fc(historical_tensor)
            combined_input = torch.cat((static_output, historical_output, idea_output), dim=1)
            idea_tensor = self.idea_fusion_fc(combined_input).unsqueeze(1)

            # Pre attention
            combined_tensor_with_idea = torch.cat((combined_output, idea_tensor), dim=1)
            lstm_attention_output, pre_weights = self.pre_attention(
                combined_tensor_with_idea, combined_tensor_with_idea, combined_tensor_with_idea
            )
            pre_attention_weights.append(pre_weights)  # Store pre-attention weights

            # LSTM
            lstm_output, (h_n, c_n) = self.lstm(lstm_attention_output)  # Capture LSTM's second output
            lstm_hidden_states.append((h_n, c_n))  # Store hidden and cell states

            # Post attention
            lstm_attention_output, post_weights = self.post_attention(lstm_output, lstm_output, lstm_output)
            post_attention_weights.append(post_weights)  # Store post-attention weights

            # Aggregate using mean pooling
            aggregated_output = torch.mean(lstm_attention_output, dim=1)  # Shape: [batch_size, hidden_dim]

            # OUTPUT
            final_prediction = self.output_fc(aggregated_output)  # Now returns [batch_size, 3]

            # Append to predictions
            predictions.append(final_prediction)  # Shape: [batch_size, 3]

            # Update historical tensor for next step
            historical_tensor = torch.cat((historical_tensor[:, 3:], final_prediction), dim=1)

        # Stack predictions into a single tensor
        predictions = torch.cat(predictions, dim=1)  # Shape: [batch_size, forecast_steps]

        # Convert pre- and post-attention weights to tensors (optional)
        pre_attention_weights = torch.stack(pre_attention_weights, dim=0)  # [steps, batch_size, num_heads, seq_len, seq_len]
        post_attention_weights = torch.stack(post_attention_weights, dim=0)  # [steps, batch_size, num_heads, seq_len, seq_len]

        # Return predictions, attention weights, and LSTM hidden states
        return predictions, pre_attention_weights, post_attention_weights, lstm_hidden_states



#%% md
# ### Example usage
# Here is an example of how to use our newly created model:
#%%
import torch
# Initialize the model - HAVE TO BE ADAPTED TO DATASET (Values are likely correct)
def example_usage():
    static_feature_dim_num = 4    # Number of static features
    historical_dim_num = 12       # Number of historical stock performance points
    hidden_dim_num = 128          # Hidden layer size
    forecast_steps_num = 12       # Predict next 12 months

    batch_size = 2

    DATASET_PATH = "../Dataset/Data/normalized_real_company_stock_dataset_large.csv"
    dataset = pd.read_csv(DATASET_PATH)
    print(f"Datasetshape: {dataset.shape}")

    print(f"Datasetshape: {dataset.shape}")

    retrieval_system = RetrievalSystem(INPUT_PATH, retrieval_number=10)

    model = RetrievalAugmentedPredictionModel(
        forecast_steps=forecast_steps_num,
        ret_sys = retrieval_system,
        retrieval_number=10
    )

    current_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Random entry
    idea_entries = dataset.iloc[10:10 + batch_size, :]  # Get a batch of rows

    # Removed tickers: Select rows from index 3 onward
    removed_tickers = [] # dataset.iloc[100:, :]["tickers"].tolist()

    # Create excluded_tickers map
    excluded_tickers = {
        i: [ticker] + removed_tickers  # Include the ticker itself and all removed tickers
        for i, ticker in enumerate(dataset["tickers"])
    }

    ideas = idea_entries["business_description"].tolist()

    static_columns = [
        col for col in dataset.columns
        if col not in ["tickers", "business_description"] and not col.startswith("month")
    ]
    month_columns = [col for col in dataset.columns if col.startswith("month")]

    # Prepare static and historical data for the batch
    static_data = idea_entries[static_columns]
    historical_data = idea_entries[month_columns]

    # Ensure numeric data and handle missing values
    static_data = static_data.apply(pd.to_numeric, errors='coerce').fillna(0).values.astype(float)
    historical_data = historical_data.apply(pd.to_numeric, errors='coerce').fillna(0).values.astype(float)

    # Convert to tensors with batch dimension
    static_data = torch.tensor(static_data, dtype=torch.float32).to(current_device)  # [batch_size, static_feature_dim_num]
    historical_data = torch.tensor(historical_data[:, -2 * forecast_steps_num:-forecast_steps_num], dtype=torch.float32).to(current_device)  # [batch_size, historical_dim_num]

    # Make a prediction
    prediction, _, _, _ = model(
        ideas=ideas,
        dataset=dataset,
        static_features=static_data,
        historical_data=historical_data,
        use_auxiliary_inputs=True,
        excluded_tickers=excluded_tickers,
    )
    print(prediction)  # Co
    print(prediction.shape)


    # Make a prediction
    prediction, _, _, _ = model(
        ideas=ideas,
        dataset=dataset,
        use_auxiliary_inputs=False
    )
    print(prediction)  # Co
    print(prediction.shape)



    retrieval_result = retrieval_system.find_similar_entries_for_batch(texts=ideas, top_n=5)
    prediction, _, _, _ = model(
        dataset=dataset,
        retrieval_result=retrieval_result,
        use_auxiliary_inputs=False,
    )

    print(prediction)
    print(prediction.shape)



#%% md
# ### Simple Training Loop
#%%
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt


# Define a PyTorch Dataset
class StockDataset(Dataset):
    def __init__(self, dataset, forecast_steps, static_columns, month_columns):
        self.dataset = dataset
        self.forecast_steps = forecast_steps
        self.static_columns = static_columns
        self.month_columns = month_columns

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        idea_entry = self.dataset.iloc[idx]

        # Extract idea, static data, historical data, and target
        idea = idea_entry["business_description"]
        ticker = idea_entry["tickers"]

        static_data = idea_entry[self.static_columns]
        historical_data = idea_entry[self.month_columns]

        # Handle missing values
        static_data = static_data.apply(pd.to_numeric, errors='coerce').fillna(0).values.astype(float)
        historical_data = historical_data.apply(pd.to_numeric, errors='coerce').fillna(0).values.astype(float)

        # Split target and input
        target = torch.tensor(historical_data[-self.forecast_steps:], dtype=torch.float32)
        historical_data = torch.tensor(historical_data[:-self.forecast_steps], dtype=torch.float32)

        # Return all relevant data
        return idea, static_data, historical_data, target, ticker

def test_training():
    dataset = pd.read_csv("../Dataset/normalized_real_company_stock_dataset_large.csv")
    removed_tickers = []
    current_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Initialize dataset and DataLoader
    static_columns = [
        col for col in dataset.columns
        if col not in ["tickers", "business_description"] and not col.startswith("month")
    ]
    month_columns = [col for col in dataset.columns if col.startswith("month")]

    forecast_steps = 6
    batch_size = 10
    retrieval_number = 10

    stock_dataset = StockDataset(dataset, forecast_steps, static_columns, month_columns)
    data_loader = DataLoader(stock_dataset, batch_size=batch_size, shuffle=True)

    # Initialize the model
    retrieval_system = RetrievalSystem(INPUT_PATH, retrieval_number=10)
    model = RetrievalAugmentedPredictionModel(
        forecast_steps=forecast_steps,
        ret_sys=retrieval_system,
        retrieval_number=10,
    )
    model.to(current_device)

    # Loss function and optimizer
    loss_fn = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    # Initialize storage for loss values
    losses = []

    # Training loop
    epochs = 1
    for epoch in range(epochs):
        total_loss = 0.0

        for batch in data_loader:
            ideas, static_data, historical_data, targets, tickers = batch

            static_data = static_data.clone().detach().to(current_device)
            historical_data = torch.stack([h.clone().detach() for h in historical_data]).to(current_device)
            targets = torch.stack([t.clone().detach() for t in targets]).to(current_device)

            # Remove tickers in the current batch from the dataset for retrieval
            excluded_tickers = list(tickers)

            # Forward pass
            predictions = model(
                ideas=ideas,
                dataset=dataset,
                static_features=static_data,
                historical_data=historical_data,
                use_auxiliary_inputs=True,
                excluded_tickers={i: [ticker] + removed_tickers for i, ticker in enumerate(excluded_tickers)}
            )

            # Compute loss
            loss = loss_fn(predictions, targets)
            total_loss += loss.item()

            # Backpropagation and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Store epoch loss
        losses.append(total_loss)

        # Print epoch summary
        print(f"Epoch [{epoch + 1}/{epochs}] completed. Total Loss: {total_loss:.4f}")

        return epochs, losses, targets, predictions


#%% md
# # Evaluation
#%%
# Plot Training Loss
def visualize_retrieval_augmented_prediction_model(model, epochs, losses, targets, predictions):
    dataset = pd.read_csv("../Dataset/normalized_real_company_stock_dataset_large.csv")
    removed_tickers = []
    current_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    plt.figure(figsize=(10, 6))
    plt.plot(range(1, epochs + 1), losses, marker='o', label="Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss Over Epochs")
    plt.legend()
    plt.grid()
    plt.show()

    # Example post-training prediction
    example_idea = "I want to create a coffee shop that uses digital cups to analyze what's in your coffee and its impact on you."
    prediction = model(
        ideas=[example_idea],
        dataset=dataset,
        use_auxiliary_inputs=False
    )
    print("Prediction after training:", prediction)

    # Plot Predictions vs. Targets
    targets_numpy = targets.cpu().detach().numpy()
    predictions_numpy = predictions.cpu().detach().numpy()

    plt.figure(figsize=(10, 6))
    for i in range(predictions_numpy.shape[0]):  # Loop over batch size
        plt.plot(targets_numpy[i], label=f"Target {i+1}", linestyle='--')
        plt.plot(predictions_numpy[i], label=f"Prediction {i+1}")
    plt.xlabel("Forecast Step")
    plt.ylabel("Value")
    plt.title("Predictions vs. Targets")
    plt.legend()
    plt.grid()
    plt.show()

    # Example post-training prediction
    example_idea = "I want to create a coffee shop that uses digital cups to analyze what's in your coffee and its impact on you."
    prediction = model(
        ideas=[example_idea],
        dataset=dataset,
        use_auxiliary_inputs=False
    )
    print("Prediction after training:", prediction)

#%% md
# # Main
# Here the test functions can be executed
# 
#%%
if __name__ == "__main__":
    example_usage()