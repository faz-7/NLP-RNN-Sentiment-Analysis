# Sentiment_Analysis_with_RNN

## Overview
This repository contains a Python implementation of a sentiment analysis project using Recurrent Neural Networks (RNNs). The project preprocesses Twitter data, applies tokenization, lemmatization, and uses word embeddings to train a model for classifying sentiments.

## Table of Contents
- [Installation](#installation)
- [Libraries Used](#libraries-used)
- [Data Preprocessing](#data-preprocessing)
- [Vectorization](#vectorization)
- [Word Embedding](#word-embedding)
- [Model Creation](#model-creation)
- [Usage](#usage)
- [Conclusion](#conclusion)

## Installation
To run the code, ensure you have the following libraries installed:

```bash
pip install pandas numpy nltk tensorflow gensim scikit-learn
```

## Libraries Used
- **pandas**: For data manipulation and analysis.
- **numpy**: For numerical computations.
- **nltk**: Natural Language Toolkit for text processing tasks.
- **tensorflow**: For building and training the RNN model.
- **gensim**: For loading pre-trained word embeddings.
- **scikit-learn**: For evaluation metrics.

## Data Preprocessing
The data is read from a CSV file containing tweets. The preprocessing steps include:
- Converting sentiment labels.
- Replacing URLs, mentions, hashtags, and punctuation.
- Tokenizing the text.
- Lemmatizing words.

## Vectorization
The text data is vectorized using the Keras Tokenizer, converting words into numerical representations, followed by padding sequences to ensure uniform input size.

## Word Embedding
Pre-trained Word2Vec embeddings are loaded to create an embedding matrix that maps words to their vector representations.

## Model Creation
An RNN model is built using Keras, which includes:
- An embedding layer initialized with the embedding matrix.
- A Simple RNN layer for sequence processing.
- A dense output layer for sentiment classification.

## Usage
1. Place your `sentiment140.csv` file in the appropriate directory.
2. Run the script to preprocess the data, train the RNN model, and evaluate its performance.
3. Check the output for test loss, accuracy, and classification metrics.

## Conclusion
This project demonstrates the application of RNNs for sentiment analysis on Twitter data. The methods can be further enhanced with more complex models or additional preprocessing techniques.
