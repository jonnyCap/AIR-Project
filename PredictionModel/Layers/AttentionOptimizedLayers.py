#%%
import torch.nn as nn
#%% md
# # IdeaLayer
# This layer encodes the textual description of our idea.
#%%
class IdeaLayer(nn.Module):
    def __init__(self, hidden_dim: int, bert_dim: int):
        super(IdeaLayer, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(bert_dim, 2 * hidden_dim),  # Reduced size
            nn.ReLU(),
            nn.Linear(2 * hidden_dim, hidden_dim)  # Reduced size
        )

    def forward(self, x):
        return self.model(x)

#%% md
# # IdeaStaticLayer
# This layer stores gets the input of the to be predicted idea. During training there will be inputs that belong to the idea we want to predict, but during actual predictions this will empty. This should just help the model to further understand the impact of other inputs
#%%
class IdeaStaticLayer(nn.Module):
    def __init__(self, static_feature_dim: int):
        super(IdeaStaticLayer, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(static_feature_dim, 16),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.model(x)

#%% md
# # IdeaHistoricalLayer
# This layer gets the historical data of the to be predicted idea as inputs. This will also be filled with existing data during training and then zerod out during acutal predictions. However, very important is that newly predicted values will be added to the input vector of this layer and the oldest value will be removed (Shift of values).
#%%
class IdeaHistoricalLayer(nn.Module):
    def __init__(self, historical_idea_dim: int, hidden_dim: int):
        super(IdeaHistoricalLayer, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(historical_idea_dim, hidden_dim//2),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.model(x)

#%% md
# # 1.Fusion Layer
# This fusion layer scales then an idea entry with bert encoding, static and historical data for a batch containing multiple retrieved documents.
#%%
class FirstFusionLayer(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int):
        super(FirstFusionLayer, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 3 * hidden_dim),
            nn.LayerNorm(3 * hidden_dim),
            nn.Dropout(0.05),
            nn.ReLU(),
            nn.Linear(3 * hidden_dim, 2 *hidden_dim),
        )


    def forward(self, x):
        return self.model(x)

#%% md
# # 2.Fusion Layer
# This fusion layer combines the inputs from the new idea.
#%%
class SecondFusionLayer(nn.Module):
    def __init__(self, hidden_dim: int):
        super(SecondFusionLayer, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(hidden_dim + 16 + 64, 2 * hidden_dim),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.model(x)

#%% md
# # Output Layer
# This is the final output layer that compromises all nodes to a single output. The overall output will then be a regressiv prediction from this layer.
#%%
class OutputLayer(nn.Module):
    def __init__(self, hidden_dim: int, retrieval_number: int):
        super(OutputLayer, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim//2),
            nn.LayerNorm(hidden_dim//2),
            nn.ReLU(),
            nn.Linear(hidden_dim //2, hidden_dim//4),
            nn.ReLU(),
            nn.Linear(hidden_dim //4, 3),
        )

    def forward(self, x):
        return self.model(x)
