# Breast Cancer Prediction with Streamlit

This project uses a machine learning model to predict whether a tumor is malignant or benign based on various features. The model is built using the Breast Cancer dataset from `sklearn` and is deployed in a Streamlit app for easy user interaction.

## Project Overview

- **Dataset**: The dataset used in this project is the Breast Cancer dataset from the `sklearn.datasets` library. It contains features related to the characteristics of cell nuclei present in breast cancer biopsies.
- **Model**: A Neural Network model (ANN) is used for prediction. Hyperparameter tuning was performed using `GridSearchCV` to improve the model's performance.
- **UI**: A Streamlit web app allows users to input features, which are then scaled and used to make predictions using the trained model.
  
## Files

- `breast_cancer_app.py`: The main Streamlit app for user input and model prediction.
- `ANN_model.ipynb`: Jupyter notebook that covers data cleaning, preprocessing, feature selection, model building, and evaluation.
- `ann_model.joblib`: The trained ANN model.
- `scaler.joblib`: The scaler used to standardize the input features.

## How to Run Locally

Follow the steps below to run the app locally:

1. **Clone the repository**:

2. **Run the Streamlit app**:

The app will open in your default web browser, where you can enter input values to make predictions.

## Hyperparameter Tuning

Hyperparameter tuning was performed on the ANN model using `GridSearchCV`. The following parameters were optimized:
- `hidden_layer_sizes`: [50, 100, 150, 200]
- `activation`: ['tanh', 'relu']
- `solver`: ['adam', 'sgd']
- `alpha`: [0.0001, 0.001, 0.01, 0.1]
- `learning_rate`: ['constant', 'invscaling', 'adaptive']

## Model Evaluation

### ANN Model (Without Hyperparameter Tuning)
- **Accuracy**: 97.37%
- **Classification Report**:
- Precision: 0.98 (for class 0), 0.97 (for class 1)
- Recall: 0.95 (for class 0), 0.99 (for class 1)
- F1-score: 0.96 (for class 0), 0.98 (for class 1)

### ANN Model (With Hyperparameter Tuning)
- **Accuracy**: 96.49%
- **Classification Report**:
- Precision: 0.93 (for class 0), 0.99 (for class 1)
- Recall: 0.98 (for class 0), 0.96 (for class 1)
- F1-score: 0.95 (for class 0), 0.97 (for class 1)

The first model (without tuning) performs slightly better than the model with hyperparameter tuning in terms of accuracy and other metrics.

## Deployment on Streamlit Cloud

Streamlit Cloud allows you to easily deploy your Streamlit app to the web. Here are the steps to deploy your app:

1. **Create a Streamlit account**: If you don’t have one, sign up at [Streamlit Cloud](https://streamlit.io/cloud).

2. **Push your code to GitHub**: Ensure that your code is pushed to a GitHub repository. If not, create a new GitHub repository and push your code there.

3. **Deploy the app**:
- Go to [Streamlit Cloud](https://streamlit.io/cloud) and log in.
- Click on **New app** and link your GitHub repository containing the code.
- Select the branch and main file (`breast_cancer_app.py`), then click **Deploy**.

4. **Access the app**: After deployment, your app will be available at a public URL, which you can share with others.

## Dependencies

The following dependencies are required for this project:

- `streamlit`: For building the web app interface.
- `sklearn`: For machine learning algorithms and data preprocessing.
- `joblib`: For saving and loading the model and scaler.
- `pandas`: For handling the dataset and manipulation.
- `numpy`: For numerical operations.

You can install all the dependencies using:

## Conclusion

This project demonstrates how to build and deploy a machine learning model for breast cancer prediction. By using a trained ANN model and integrating it with a Streamlit app, users can easily input data and get predictions in real-time.
