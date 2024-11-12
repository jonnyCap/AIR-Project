#%% md
# # Hybrid Stock Prediction Model
# 
# This Model, specifically created to make Stock Predictions for upcoming Businesses, means this model predicts the market startup of any new business idea.
# 
# ### Model Architecture
# To create the most realistic approach possible, we created a hybrid model consisting of the following layers:
# 1. Encodes business ideas using Sentence-BERT.
# 2. Processes static company features using a dense layer.
# 3. Combines both representations in a fusion layer.
# 4. Uses an LSTM to make sequential predictions across a 12-month period.
# 5. Outputs a prediction for each month in the forecast period.
# 
#%%
import torch
import torch.nn as nn
from sentence_transformers import SentenceTransformer

class StockPerformancePredictionModel(nn.Module):
    def __init__(self, text_embedding_dim, static_feature_dim, hidden_dim, forecast_steps):
        super(StockPerformancePredictionModel, self).__init__()

        # Text representation layer (Sentence-BERT)
        self.text_encoder = SentenceTransformer('all-MiniLM-L6-v2')

        # Freeze Sentence-BERT parameters (optional, if you don't want fine-tuning)
        for param in self.text_encoder.parameters():
            param.requires_grad = False

        # Static feature layer
        self.static_fc = nn.Linear(static_feature_dim, hidden_dim)

        # Fusion layer
        self.fusion_fc = nn.Linear(text_embedding_dim + hidden_dim, hidden_dim)

        # Time-series prediction (LSTM)
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
        self.output_fc = nn.Linear(hidden_dim, 1)  # Output single value per time step
        self.forecast_steps = forecast_steps

    def forward(self, idea_text, static_features):
        # Ensure static_features is on the same device as the model
        device = next(self.parameters()).device
        static_features = static_features.to(device)

        # Text embedding
        text_embedding = torch.tensor(
            self.text_encoder.encode(idea_text, convert_to_numpy=True)
        ).to(device)

        # Add batch dimension if processing a single input
        if text_embedding.dim() == 1:
            text_embedding = text_embedding.unsqueeze(0)

        # Static feature embedding
        static_embedding = torch.relu(self.static_fc(static_features))

        # Fusion
        combined_input = torch.cat((text_embedding, static_embedding), dim=-1)
        combined_input = torch.relu(self.fusion_fc(combined_input))

        # Repeat for time-series prediction
        lstm_input = combined_input.unsqueeze(1).repeat(1, self.forecast_steps, 1)
        lstm_out, _ = self.lstm(lstm_input)

        # Generate monthly predictions
        predictions = self.output_fc(lstm_out).squeeze(-1)  # Shape: (batch_size, forecast_steps)
        return predictions


#%% md
# ### Example usage
# Here is an example of how to use our newly created model:
#%%
# Instantiate the model with appropriate dimensions
model = StockPerformancePredictionModel(text_embedding_dim=384, static_feature_dim=10, hidden_dim=128, forecast_steps=12)

# Define input data
idea_text = "Innovative AI-driven approach to personalized medicine."
static_features = torch.randn(1, 10)  # Corrected static feature input for batch_size = 1

# Generate predictions
predictions = model(idea_text, static_features)
print(predictions.shape)  # Expected shape: (1, 12)

