#%% md
# # Similarity Layer
# This layer takes the similarity values from the RetrievalSystem as Inputs, which should gain the model better insights into how correlated the retrieved inputs are.
#%%
import torch.nn as nn

class SimilarityLayer(nn.Module):
    def __init__(self, retrieval_number: int):
        super(SimilarityLayer, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(retrieval_number, retrieval_number),  # Reduced size
            nn.ReLU(),
            nn.LayerNorm(retrieval_number),  # Directly apply LayerNorm here
            nn.ReLU()
        )

    def forward(self, x):
        return self.model(x)

#%% md
# # StaticFeatureLayer
# This layer has all the static features from the retrieved documents as input.
#%%
class StaticFeatureLayer(nn.Module):
    def __init__(self, retrieval_number: int, hidden_dim: int, static_feature_dim: int):
        super(StaticFeatureLayer, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(static_feature_dim * retrieval_number, static_feature_dim * (retrieval_number // 2)),
            nn.ReLU(),
            nn.Linear(static_feature_dim * (retrieval_number // 2), hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU()
        )

    def forward(self, x):
        return self.model(x)

#%% md
# # HistoricalFeatureLayer
# This layer contains all the hisitorical data from the retrieved companies.
#%%
class HistoricalFeatureLayer(nn.Module):
    def __init__(self, retrieval_number: int, hidden_dim: int, historical_feature_dim: int):
        super(HistoricalFeatureLayer, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(historical_feature_dim * retrieval_number, historical_feature_dim * retrieval_number // 2),
            nn.ReLU(),
            nn.Linear(historical_feature_dim * retrieval_number // 2, 2 * hidden_dim),  # Reduced size
            nn.Dropout(0.1),
            nn.LayerNorm(2 * hidden_dim),
            nn.ReLU()
        )

    def forward(self, x):
        return self.model(x)


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
        self.model = nn.Linear(static_feature_dim, 16)  # Reduced size

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
            nn.Linear(hidden_dim//2, hidden_dim//4),
        )

    def forward(self, x):
        return self.model(x)

#%% md
# # 1.Fusion Layer
# This fusion layer combines all the inputs from our retrieved documents.
#%%
class FirstFusionLayer(nn.Module):
    def __init__(self, bert_dim: int, hidden_dim: int, retrieval_number: int):
        super(FirstFusionLayer, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(bert_dim + retrieval_number + hidden_dim + 2 * hidden_dim + retrieval_number, 6 * hidden_dim),
            nn.ReLU(),
            nn.Linear(6 * hidden_dim,  4 * hidden_dim),
            nn.LayerNorm(4 * hidden_dim),
            nn.ReLU(),
            nn.Linear(4 * hidden_dim,  3 * hidden_dim),
            nn.ReLU()
        )

    def forward(self, x):
        return self.model(x)

#%% md
# # 2.Fusion Layer
# This fusion layer combines the output from the first fusion layer and the inputs from the idea we want to predict.
#%%
class SecondFusionLayer(nn.Module):
    def __init__(self, hidden_dim: int):
        super(SecondFusionLayer, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(3 * hidden_dim + hidden_dim + 16 + hidden_dim//4, 4 * hidden_dim),
            nn.ReLU(),
            nn.Linear(4 * hidden_dim, 2 * hidden_dim),
            nn.LayerNorm(2 * hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
        )

    def forward(self, x):
        return self.model(x)

#%% md
# # Output Layer
# This is the final output layer that compromises all nodes to a single output. The overall output will then be a regressiv prediction from this layer.
#%%
class OutputLayer(nn.Module):
    def __init__(self, hidden_dim: int):
        super(OutputLayer, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim//2),
            nn.LayerNorm(hidden_dim//2),
            nn.Linear(hidden_dim//2, hidden_dim //4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, 1), # Final Output
        )

    def forward(self, x):
        return self.model(x)
