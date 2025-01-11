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
    # print(ideas)
        # Prediction
    prediction = userinterface.prediction_model(
        ideas = ideas,
        dataset=userinterface.dataset,
        use_auxiliary_inputs=False,
        not_frontend = False
    )
    # TODO: SHOULD BE DONE AFTER RANKING MODEL
    month_columns = [col for col in userinterface.dataset.columns if col.startswith("month")]
    if os.path.exists(userinterface.historical_scaler_path) and os.path.exists(userinterface.ranking_historical_scaler_path):
        with open(userinterface.historical_scaler_path, "rb") as scaler_file:
            historical_scaler = joblib.load(scaler_file)
        with open(userinterface.ranking_historical_scaler_path, "rb") as ranking_scaler_file:
            ranking_historical_scaler = joblib.load(ranking_scaler_file)
        norm_predictions = torch.zeros(1, 72)
        norm_predictions[0, 60:72] = prediction
        norm_predictions_np = norm_predictions.cpu().detach().numpy()
        denorm_predictions_np = historical_scaler.inverse_transform(norm_predictions_np)
        
        #TODO: Scale with ranking scaler back down
    # else:
        # print("Couldnt load all required scalers")
        # raise FileNotFoundError("Historical Scaler not found")
    print(prediction[0].tolist())