import argparse
import json
import numpy as np
import torch
from torch.utils.data import Dataset
import pandas as pd
from transformers import BertTokenizer, BertModel
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset
import matplotlib.pyplot as plt
import urllib
import os
from ProjectPipeline import UserInterface
IDEA_IDENTIFIER = "NEW_IDEA"


if __name__ == '__main__':
    userinterface = UserInterface()
    parser = argparse.ArgumentParser()
    parser.add_argument("--company_and_stock", type=str, required=True)
    args = parser.parse_args()
    # print(f"Received company stock JSON: {args.company_and_stock}")
    with open(args.company_and_stock, "r") as file:
        company_stock = json.load(file)
    # print(company_stock)
    companies = []
    ratings ={}
    # Example data for ten companies with realistic stock performance
    for company in company_stock:
        # Extract the relevant data from the company object
        company_name = company["CompanyName"]
        stock_performance = company["StockPerformance"]
        embedding = company["Embedding"]

        # Convert stock performance to a tensor
        stock_performance_tensor = torch.tensor(stock_performance, dtype=torch.float32)

        # Convert embedding to a tensor
        similar_embeddings_tensor = torch.tensor(embedding, dtype=torch.float32)
        # print(stock_performance_tensor.unsqueeze(0).shape)
        # print(similar_embeddings_tensor.unsqueeze(0).shape)
        # Assuming you have a ranking model defined like in the original code
        # Get the rating for this company using the ranking model
        ratings[company_name] = userinterface.ranking_model(idea_encoding=similar_embeddings_tensor.unsqueeze(0), stock_performance=stock_performance_tensor.unsqueeze(0))

        # Store the results in a dictionary for the company
        company_data = {
            "idea": company_name,
            "rating": ratings[company_name].item()
        }

        companies.append(company_data)
    

    # Print the results
    companies.sort(key=lambda x: x["rating"], reverse=True)
    json_data = json.dumps(companies, indent=4)
    print(json_data)
