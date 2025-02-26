#%% md
# # Hybrid Stock Prediction RAPModel
# 
# This RAPModel, specifically created to make Stock Predictions for upcoming Businesses, means this model predicts the market startup of any new business idea.
# 
# ### RAPModel Architecture
# To create the most realistic approach possible, we created a hybrid model consisting of the following layers:
# 1. Encodes business ideas using Sentence-BERT.
# 2. Processes static company features using a dense layer.
# 3. Combines both representations in a fusion layer.
# 4. Uses an LSTM to make sequential predictions across a 12-month period.
# 5. Outputs a prediction for each month in the forecast period.
# 
#%%
import torch.nn as nn
import numpy as np
from sentence_transformers import SentenceTransformer

class StockPerformancePredictionModel(nn.Module):
    def __init__(self, static_feature_dim, historical_dim, hidden_dim, forecast_steps, num_lstm_layers=2):
        super(StockPerformancePredictionModel, self).__init__()

        # Text representation layer (Sentence-BERT)
        self.text_encoder = SentenceTransformer('all-MiniLM-L6-v2')

        # Freeze Sentence-BERT parameters (optional)
        for param in self.text_encoder.parameters():
            param.requires_grad = False

        # Static feature layers (deep)
        self.static_fc = nn.Sequential(
            nn.Linear(static_feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU()
        )

        # Historical stock data layers (deep)
        self.historical_fc = nn.Sequential(
            nn.Linear(historical_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU()
        )

        # Fusion layer to combine text, static, and historical embeddings
        self.fusion_fc = nn.Sequential(
            nn.Linear(384 + 2 * hidden_dim, hidden_dim),  # 384 is the fixed text embedding dimension
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU()
        )

        # Text-only layer for inference
        self.text_only_fc = nn.Sequential(
            nn.Linear(384, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU()
        )

        # Multi-layer LSTM with residual connection
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, num_layers=num_lstm_layers, batch_first=True, dropout=0.2)

        # Attention mechanism
        self.attention = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=4, batch_first=True)

        # Output layer for forecasting
        self.output_fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)  # Output single value per timestep
        )

        self.forecast_steps = forecast_steps

    def forward(self, idea, static_features=None, historical_data=None, use_auxiliary_inputs=True, predict_autoregressively=False):
        # Ensure device compatibility
        device = next(self.parameters()).device

        # Text embedding
        encoded_output = self.text_encoder.encode(idea, convert_to_numpy=True)
        text_embedding = torch.from_numpy(encoded_output).float().to(device)

        # Add batch dimension if processing a single input
        if text_embedding.dim() == 1:
            text_embedding = text_embedding.unsqueeze(0)

        if use_auxiliary_inputs:
            # Static feature embedding
            static_embedding = self.static_fc(static_features.to(device))

            # Historical stock data embedding
            historical_embedding = self.historical_fc(historical_data.to(device))

            # Fusion of text + static + historical embeddings
            combined_input = torch.cat((text_embedding, static_embedding, historical_embedding), dim=-1)
            combined_input = self.fusion_fc(combined_input)
        else:
            # Text-only input (for inference)
            combined_input = self.text_only_fc(text_embedding)

        if not predict_autoregressively:
            # Repeat for time-series prediction
            lstm_input = combined_input.unsqueeze(1).repeat(1, self.forecast_steps, 1)

            # Pass through LSTM
            lstm_out, _ = self.lstm(lstm_input)

            # Apply attention mechanism
            attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)

            # Output predictions
            predictions = self.output_fc(attn_out).squeeze(-1)  # Shape: (batch_size, forecast_steps)
            return predictions
        else:
            # Autoregressive prediction
            predictions = []
            hidden_state = None
            input_step = combined_input.unsqueeze(1)

            for _ in range(self.forecast_steps):
                lstm_out, hidden_state = self.lstm(input_step, hidden_state)
                attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
                current_prediction = self.output_fc(attn_out.squeeze(1))
                predictions.append(current_prediction)

                # Use text-only for subsequent steps
                input_step = self.text_only_fc(text_embedding).unsqueeze(1)

            predictions = torch.stack(predictions, dim=1)
            return predictions


#%% md
# ### Example usage
# Here is an example of how to use our newly created model:
#%%
import torch

# Initialize the model - HAVE TO BE ADAPTED TO DATASET (Values are likely correct)
static_feature_dim_num = 4    # Number of static features
historical_dim_num = 12       # Number of historical stock performance points
hidden_dim_num = 128          # Hidden layer size
forecast_steps_num = 12       # Predict next 12 months

model = StockPerformancePredictionModel(
    static_feature_dim=static_feature_dim_num,
    historical_dim=historical_dim_num,
    hidden_dim=hidden_dim_num,
    forecast_steps=forecast_steps_num
)

# Example input data
idea_text = ["AI-powered e-commerce platform targeting luxury goods."]
fake_static_features = torch.tensor([[1e9, 500000, 0.25, 10]])  # Example static features (batch size = 1)
fake_historical_data = torch.tensor([[0.05, 0.08, 0.06, -0.02, 0.07, 0.03, -0.01, 0.04, 0.02, 0.01, -0.03, 0.05]])  # Example historical data

# Move to the same device as the model
current_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(current_device)
fake_static_features = fake_static_features.to(current_device)
fake_historical_data = fake_historical_data.to(current_device)

#%% md
# After setting up the model we can use it like this:
#%%
# Forward pass with simultaneous prediction
first_predictions = model(
    idea=idea_text,
    static_features=fake_static_features,
    historical_data=fake_historical_data,
    use_auxiliary_inputs=True,
    predict_autoregressively=False  # Default mode
)

print("Simultaneous Predictions:", first_predictions)


# Forward pass with autoregressive prediction
predictions_autoregressive = model(
    idea=idea_text,
    static_features=fake_static_features,
    historical_data=fake_historical_data,
    use_auxiliary_inputs=True,
    predict_autoregressively=True  # Autoregressive mode
)

print("Autoregressive Predictions:", predictions_autoregressive)

# Forward pass with text-only input
predictions_text_only = model(
    idea=idea_text,
    use_auxiliary_inputs=False,
    predict_autoregressively=False  # Simultaneous mode with text-only
)

print("Text-Only Predictions:", predictions_text_only)

#%% md
# ### Simple Training Loop
#%%
import torch
import torch.optim as optim

# Example data (replace with your actual dataset)
idea_texts_test = ["AI-powered e-commerce platform", "Blockchain for supply chain management"]
static_features_test = torch.tensor([[1e6, 0.2, 10, 50], [5e5, 0.1, 5, 25]])  # Shape: (batch_size, static_feature_dim)
historical_data_test = torch.tensor([[0.05, 0.08, 0.07, 0.03, 0.04, 0.06, 0.08, 0.09, 0.07, 0.05, 0.02, 0.01],
                                [0.10, 0.09, 0.08, 0.06, 0.07, 0.05, 0.04, 0.03, 0.02, 0.01, 0.00, -0.01]])  # Shape: (batch_size, historical_dim)
targets = torch.tensor([[0.06, 0.07, 0.08, 0.09, 0.08, 0.07, 0.06, 0.05, 0.04, 0.03, 0.02, 0.01],
                        [0.09, 0.08, 0.07, 0.06, 0.05, 0.04, 0.03, 0.02, 0.01, 0.00, -0.01, -0.02]])  # Shape: (batch_size, forecast_steps)

# Define model parameters
static_feature_dim_test = 4
historical_dim_test = 12
hidden_dim_test = 128
forecast_steps_test = 12

# Initialize the model
model = StockPerformancePredictionModel(
    static_feature_dim=static_feature_dim_test,
    historical_dim=historical_dim_test,
    hidden_dim=hidden_dim_test,
    forecast_steps=forecast_steps_test
)

# Move model and data to the same device
test_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(test_device)
static_features_test = static_features_test.to(test_device)
historical_data_test = historical_data_test.to(test_device)
targets = targets.to(test_device)

# Define optimizer and loss function
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.MSELoss()

# Training loop
num_epochs = 10
for epoch in range(num_epochs):
    model.train()

    # Forward pass
    test_predictions = model(
        idea=idea_texts_test,
        static_features=static_features_test,
        historical_data=historical_data_test,
        use_auxiliary_inputs=True
    )

    # Compute loss
    loss = criterion(test_predictions, targets)

    # Backward pass
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")

#%% md
# ### Now we can test again:
#%%
first_predictions = model(
    idea=idea_text,
    static_features=fake_static_features,
    historical_data=fake_historical_data,
    use_auxiliary_inputs=True,
    predict_autoregressively=False  # Default mode
)

print("Simultaneous Predictions:", first_predictions)


# Forward pass with autoregressive prediction
predictions_autoregressive = model(
    idea=idea_text,
    static_features=fake_static_features,
    historical_data=fake_historical_data,
    use_auxiliary_inputs=True,
    predict_autoregressively=True  # Autoregressive mode
)

print("Autoregressive Predictions:", predictions_autoregressive)

# Forward pass with text-only input
predictions_text_only = model(
    idea=idea_text,
    use_auxiliary_inputs=False,
    predict_autoregressively=False  # Simultaneous mode with text-only
)

print("Text-Only Predictions:", predictions_text_only)