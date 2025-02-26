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
The **front-end** was designed and developed with **Windows Forms** built on C#, offering an intuitive and user-friendly interface. 
With this intuitive design, we interact with models without the need of a server or a backend system. We instead leverage **Python scripts** as alternative to a standard back-end, which offers users a stable connection to the model while enabling enough computational power. Ultimately, the end result is an amalgamation of the best of both worlds – a Windows application, along with a user friendly inbuilt Python interface. This enables us to deeply interact with our models without any hassle. To top it off, a GUI was designed which streamlines the whole process, making it a lot more user friendly.
## How to Start the Application

Follow the steps below to set up and run the application using Microsoft Visual Studio.

1. Locate the Solution File
- Navigate to the project directory and find the `.sln` file (e.g., `MyApp.sln`).
- Double-click the `.sln` file to open it in Microsoft Visual Studio.

2. Set Up the Environment
**Create a Virtual Environment** (if required):
   - Run the following command in your terminal to create a virtual environment:
     ```bash
     python -m venv venv
     ```
   - Activate the virtual environment:
     - On Windows:
       ```bash
       venv\Scripts\activate
       ```
     - On macOS/Linux:
       ```bash
       source venv/bin/activate
       ```
**Install Dependencies**:
   - Install all required dependencies using the `requirements.txt` file. Run:
     ```bash
     pip install -r requirements.txt
     ```
3. Prepare the Embeddings File
- Go to the `CompromisedEmbeddings` folder in the project directory.
- Locate the `.csv` file and unzip it into a new folder named `Embeddings` in the project root.
- Ensure the unzipped `.csv` file is placed directly in the `Embeddings` folder and that the .csv file is called embeddings.csv.

4. Build and Run the Application
- Locate the WindowsForm file (`.sln`) in the `AIR` folder. 
- Open Microsoft Visual Studio with the `.sln` file by double-clicking on it.
- Ensure the correct configuration is selected (e.g., `Debug` or `Release`).
- Click the **Run** button (green play button) in the toolbar or press `F5` to start the application.

5. Application Ready
- The application should now be running and ready for use.

In The following images, you can see the design and a walkthrough of how to use the **GUI**. First of all, the user begins at the initial screen, where they can input their idea into a designated text field.
<div style="display: flex; justify-content: space-around; align-items: center;">
    <img src="/Documents/Images/AIR%20-%20Interface%20-%20App_starting.png" alt="Start Screen" title="Start Screen" style="width: 45%;"/>
    <img src="/Documents/Images/AIR%20-%20Loading%20Screen.png" alt="Loading Screen" title="Loading Screen" style="width: 45%;"/>
</div>

Once the idea is submitted, a loading screen appears, represented by a progress bar that tracks the system's processing of the input. Once the processing is complete, a new screen pops up displaying a list of similar companies to the input idea.
Here The Companies which are Similar to the Input-Idea will be shown.
At the top of this screen, there is a menu with four options: **Similar Companies**, **Stock Ranking**, **Stock Prediction**, and the option for a **New Idea**.
Selecting any of these options navigates the user to their respective visualizations.
<div style="display: flex; justify-content: space-around; align-items: center;">
    <img src="/Documents/Images/AIR%20-%20Similar-Companies.png" alt="Similar Companies" title="Similar Companies" style="width: 30%;"/>
    <img src="/Documents/Images/AIR%20-%20Stock-Prediction.png" alt="Stock Prediction" title="Stock Prediction" style="width: 30%;"/>
    <img src="/Documents/Images/AIR%20-%20Ranking-Companies.png" alt="Ranking Companies" title="Ranking Companies" style="width: 30%;"/>
</div>


Once Selecting the option **"Stock Prediction"**, you will be able to see how your idea would perform, by forecasting the stock related to the idea.
The **Ranking Companies** shows how the idea performs against existing companies and ranks them. 

## Model Architecture, Results and Evaluation
Fur further information checkout our [Design Document](/Documents/Design%20Document/AIR_DD_G09_V2.pdf), our [Report](/Documents/Report/AIR_RE_G09.pdf), and our [Presentation](/Documents/Presentation/AIR_PR_G09.pdf).
However, here is a short graphical overview over our Model:

![System Architecture Image](/Documents/Images/AIR%20-%20RAPM%20Architecture.png "System architecture")


The first attempt to reliably predict stock performances was a rather simple [**prediction-system**](https://github.com/jonnyCap/AIR-Project/blob/main/PredictionModel/HybridStockPredictionModel.ipynb), based solely on the inputs of a new idea. Later on we extended this with implementation of retrieval augmentation, which lead to the creatoin of our [**Retrieval Augmented Prediction Model (RAP-Model)**](https://github.com/jonnyCap/AIR-Project/blob/main/PredictionModel/RetrievalAugmentedPredictionModel.ipynb).
This was so far our most promising architecture — one that can still be adapted in many variants, depending on the number of retrieved elements, forecast steps, and auxiliary inputs. In an effort to further improve accuracy, we developed an [**Attention-Optimized Retrieval-Augmented Prediction Model**](https://github.com/jonnyCap/AIR-Project/blob/attention_fix/PredictionModel/AttentionOptimizedRetrievalAugmentedPredictionModel.ipynb), designed to better leverage attention mechanisms and LSTM layers. However, as this approach did not yield the expected results, we decided to continue working with the original **RAP Model**.
