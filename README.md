# AIR-Project: Information Retrieval based on textual Description of Business Idea
*(Project for the course Advanced Information Retrieval at Technical University of Graz)*
## Description
We build an advanced information retrieval system that allows the user to retrieve comprehensive information about his
new business idea. After providing ones idea, our system provides you with the following information:
1. **Similar Companies**
2. **Stock Closing Prices Prediction**
3. **An evaluation Score and ranking within similar competitors**
## UI and System Pipeline
For each of the above information we implemented an individual System, that takes the input from the model before, so it
is sort of a pipeline:

![SubSystem Pipeline](/Documents/Images/AIR%20-%20User%20Interaction%20Pipeline%20V2.png "System Pipeline")

To sum this up, the **Retrieval System** finds the **n** most similar companies and returns the precomputed embeddings
alongside their IDs ("ticker") and the preprocessed and embedded idea text. These are then used for inputs for the following
models. Additionally, the **Ranking Model** also gets the predicted stock performance from our **RAP-Model** as input.

In this image you can see the following subsystem:
1. [Retrieval System](https://github.com/jonnyCap/AIR-Project/blob/main/RetrievalSystem/RetrievalSystem.ipynb)
2. [Retrieval Augmented Prediction Model (RAP-Model)](https://github.com/jonnyCap/AIR-Project/blob/main/PredictionModel/RetrievalAugmentedPredictionModel.ipynb)
3. [Ranking Model](https://github.com/jonnyCap/AIR-Project/blob/main/RankingModel/RankingModel.ipynb)

that work together to provide you with the most accurate information possible

### GUI
We also created a graphical user interface, to allow for easy interaction with our model:
```
npm run dev
```

## Model Architecture, Results and Evaluation
Fur further information checkout our [Design Document](/Documents/Design%20Document/AIR_DD_G09_V2.pdf), our [Report](/Documents/Report/Report.pdf), and our [Presentation](/Documents/Presentation/Presentation.pdf).
However, here is a short graphical overview over our Model:

![System Architecture Image](/Documents/Images/AIR%20-%20RAPM%20Architecture.png "System architecture")


The first attempt to reliably predict stock performances was a rather simple [**prediction-system**](https://github.com/jonnyCap/AIR-Project/blob/main/PredictionModel/HybridStockPredictionModel.ipynb), based solely on the inputs of a new idea. Later on we extended this with implementation of retrieval augmentation, which lead to the creatoin of our [**Retrieval Augmented Prediction Model (RAP-Model)**](https://github.com/jonnyCap/AIR-Project/blob/main/PredictionModel/RetrievalAugmentedPredictionModel.ipynb).
This was so far our most promising architecture — one that can still be adapted in many variants, depending on the number of retrieved elements, forecast steps, and auxiliary inputs. In an effort to further improve accuracy, we developed an [**Attention-Optimized Retrieval-Augmented Prediction Model**](https://github.com/jonnyCap/AIR-Project/blob/attention_fix/PredictionModel/AttentionOptimizedRetrievalAugmentedPredictionModel.ipynb), designed to better leverage attention mechanisms and LSTM layers. However, as this approach did not yield the expected results, we decided to continue working with the original **RAP Model**.
