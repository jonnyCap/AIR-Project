#%% md
# # AttentionModel
# This model combines multiple Bert layers
#%%
import torch
import torch.nn as nn

class AttentionModel(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(AttentionModel, self).__init__()
        # Fully connected layers for attention scoring
        self.attention_fc = nn.Linear(input_dim, hidden_dim)
        self.context_vector = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, text_embeddings):
        """
        text_embeddings: Tensor of shape [num_inputs, input_dim]
                         (e.g., [10, 384] for 10 BERT-encoded vectors)
        """
        # Compute attention scores
        hidden_representation = torch.tanh(self.attention_fc(text_embeddings))  # Shape: [num_inputs, hidden_dim]
        attention_scores = self.context_vector(hidden_representation).squeeze(-1)  # Shape: [num_inputs]

        # Normalize scores with softmax
        attention_weights = torch.softmax(attention_scores, dim=0)  # Shape: [num_inputs]

        # Apply attention weights to input embeddings
        weighted_sum = torch.sum(attention_weights.unsqueeze(-1) * text_embeddings, dim=0)  # Shape: [input_dim]

        return weighted_sum, attention_weights
