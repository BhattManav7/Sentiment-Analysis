# Sentiment Analysis using LSTM on IMDb Reviews

This project involves building a sentiment analysis model using Long Short-Term Memory (LSTM) neural networks. The model was trained on IMDb movie reviews to classify sentiments as positive or negative. Core NLP and deep learning techniques were explored throughout the project.

## Project Summary

- Built a deep learning model using LSTM for binary sentiment classification
- Achieved an accuracy of 87.44% on the IMDb test set
- Performed text preprocessing including tokenization, padding, and cleaning
- Used dropout regularization and binary cross-entropy loss for training
- Studied core machine learning concepts such as backpropagation, hyperparameters, and model tuning

## Objectives

- Apply NLP techniques to real-world text classification
- Build and evaluate an LSTM-based model for sentiment analysis
- Improve model performance using regularization and hyperparameter tuning
- Gain hands-on understanding of sequence modeling and evaluation metrics

## Technologies Used

- Python 3.x
- TensorFlow / Keras
- NumPy, Pandas
- Matplotlib, Seaborn
- NLTK, Regular Expressions

## Preprocessing Steps

- Convert text to lowercase
- Remove punctuation and special characters
- Tokenize and map words to indices
- Pad sequences to ensure equal input length
- Train-validation split

## Model Architecture

- Embedding layer to represent input tokens
- LSTM layer to capture sequence information
- Dropout layers for regularization
- Dense output layer with sigmoid activation
- Optimizer: Adam
- Loss Function: Binary Cross-Entropy

## Results

- Test Accuracy: 87.44%
- Good generalization with minimal overfitting
- Evaluation metrics and training graphs included for analysis

## Key Learnings

- LSTM's ability to handle long-term dependencies in text
- Importance of data preprocessing in NLP workflows
- Regularization techniques such as dropout to prevent overfitting
- Interpreting evaluation metrics in a binary classification setting

## Future Work

- Use pre-trained word embeddings such as GloVe or Word2Vec
- Try Bidirectional LSTM or GRU networks
- Extend to multi-class sentiment analysis or other text datasets
- Deploy as an interactive web app using Flask or Streamlit

