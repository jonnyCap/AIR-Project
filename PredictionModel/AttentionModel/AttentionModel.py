#%% md
# # AttentionModel
# This model combines multiple Bert layers
#%%
import torch
import torch.nn as nn

class AttentionFusion(nn.Module):
    def __init__(self, input_dim = 384, hidden_dim = 128, *args, **kwargs):
        super(AttentionFusion, self).__init__()

        # Fully connected layer to compute attention scores
        self.attention_fc = nn.Linear(input_dim, hidden_dim)
        self.context_vector = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, text_embeddings):
        """
        text_embeddings: Tensor of shape [batch_size, num_inputs, input_dim]
                         (e.g., [batch_size, 10, 384] for 10 BERT-encoded vectors)
        """
        # Compute attention scores
        hidden_representation = torch.tanh(self.attention_fc(text_embeddings))  # Shape: [batch_size, num_inputs, hidden_dim]
        attention_scores = self.context_vector(hidden_representation).squeeze(-1)  # Shape: [batch_size, num_inputs]

        # Normalize scores with softmax
        attention_weights = torch.softmax(attention_scores, dim=1)  # Shape: [batch_size, num_inputs]

        # Apply attention weights
        weighted_sum = torch.sum(attention_weights.unsqueeze(-1) * text_embeddings, dim=1)  # Shape: [batch_size, input_dim]

        return weighted_sum, attention_weights