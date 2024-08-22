# YouTube Comments Spam Detector

This project is a machine learning model designed to detect spam comments on YouTube videos. The model is trained using a dataset containing comments labeled as spam or not spam. The goal is to classify new comments accurately, helping to filter out unwanted content.

## Project Overview

- **Language**: Python
- **Libraries Used**: Pandas, NumPy, Scikit-learn, NLTK, Matplotlib, Seaborn, TensorFlow/Keras
- **Model**: Logistic Regression
- **Dataset**: CSV file containing YouTube comments with columns `COMMENT_ID`, `AUTHOR`, `DATE`, `CONTENT`, `VIDEO_NAME`, and `CLASS`.

## Features

- **Data Preprocessing**: Clean and preprocess the comments by removing punctuation, numbers, and stopwords.
- **Text Vectorization**: Convert comments into numerical features using TF-IDF vectorization.
- **Model Training**: Train a logistic regression model to classify comments as spam or not spam.
- **Model Evaluation**: Evaluate model performance using accuracy, confusion matrix, and classification report.
- **Prediction**: Predict whether new comments are spam or not using the trained model.

## Usage

1. **Data Cleaning**: The content of the comments is cleaned and preprocessed.
2. **Training**: The model is trained using the cleaned and vectorized text data.
3. **Prediction**: New comments are input, cleaned, and classified by the model.
4. **Model Saving**: The trained model and vectorizer are saved for future use.

## How to Run

1. Clone the repository.
2. Upload your dataset to the environment (e.g., Google Colab).
3. Run the provided cells in sequence to preprocess data, train the model, and make predictions.
4. (Optional) Save the trained model and vectorizer for future predictions.

## Example

You can use the saved model to predict whether the following comments are spam or not:

```python
new_comments = ["This is a spam comment!", "Amazing content, keep it up!"]
