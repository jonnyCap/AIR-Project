import argparse
import json
import joblib
import torch
import os

import urllib
from ProjectPipeline import UserInterface
retrieval_result = 0
def setRetrievalResult(retrieval):
    retrieval_result = retrieval

current_directory = os.path.dirname(os.path.abspath(__file__))
parent_directory = os.path.dirname(current_directory)
# print(parent_directory)
INPUT_PATH = os.path.join(parent_directory,"RetrievalSystem","CompromisedEmbeddings","embeddings.csv")

BERT_DIM = 384
if __name__ == '__main__':
    userinterface = UserInterface()
    
    parser = argparse.ArgumentParser(description="Find similar companies based on an idea.")
    parser.add_argument("--ideas", type=str, required=True, help="The idea to search similar companies for.")

    args = parser.parse_args()
    # print(args.ideas)
    decoded_str = urllib.parse.unquote(args.ideas)
    ideas = json.loads(decoded_str)
    while len(ideas) > 1:
        ideas.pop()
    prediction = userinterface.prediction_model(
        ideas = ideas,
        dataset=userinterface.dataset,
        use_auxiliary_inputs=False,
        not_frontend = False
    )
    if os.path.exists(userinterface.historical_scaler_path):
        # Load the historical scaler
        with open(userinterface.historical_scaler_path, "rb") as scaler_file:
            historical_scaler = joblib.load(scaler_file)
        # Prepare a padded tensor for the prediction
        padded_prediction = torch.zeros(1, 72, device='cpu')
        padded_prediction[:, 60:72] = prediction    
        # Convert the tensor to numpy and denormalize only the prediction portion
        padded_prediction_np = padded_prediction.detach().numpy()
        denormalized_values = historical_scaler.inverse_transform(padded_prediction_np)
        # Extract the denormalized prediction portion
        denormalized_prediction = denormalized_values[0, 60:72] 

    else:
        print("Could not load the required scaler.")
        raise FileNotFoundError("Historical Scaler not found")
    print(denormalized_prediction.tolist())