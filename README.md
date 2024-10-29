# Yelp Review Prediction

This repository contains a machine learning project that predicts the star ratings of Yelp reviews based on their textual content. The model uses TensorFlow and Keras to build a neural network that performs regression on the star ratings.

## Table of Contents

- [Overview](#overview)
- [Requirements](#requirements)
- [Data Preparation](#data-preparation)
- [Model Architecture](#model-architecture)
- [Usage](#usage)
- [Results](#results)
- [License](#license)

## Overview

The main objective of this project is to analyze Yelp reviews and predict the corresponding star ratings using natural language processing (NLP) techniques and a neural network model. The code utilizes TF-IDF vectorization for feature extraction and a Sequential model from Keras for training.

## Requirements

To run this code, you'll need the following packages:

- Python 3.x
- NumPy
- Matplotlib
- scikit-learn
- TensorFlow

You can install the necessary packages using pip:

```bash
pip install numpy matplotlib scikit-learn tensorflow
```

## Data Preparation

The code expects a JSON file containing Yelp reviews, with each review containing the fields `text` and `stars`. Ensure your dataset is formatted similarly to the Yelp Academic Dataset.

### Example JSON Structure

```json
[
  {
    "text": "The food was amazing!",
    "stars": 5
  },
  {
    "text": "Service was terrible.",
    "stars": 1
  }
]
```

## Model Architecture

The model consists of the following layers:

1. **Input Layer**: 128 neurons with ReLU activation.
2. **Dropout Layer**: Dropout rate of 0.5 to prevent overfitting.
3. **Hidden Layer**: 64 neurons with ReLU activation.
4. **Output Layer**: 1 neuron for regression output (predicted star rating).

The model is compiled with the Adam optimizer and mean squared error (MSE) as the loss function.

## Usage

1. **Load Data**: Specify the path to the JSON file containing Yelp reviews by modifying the `filtered_review_path` variable in the code.
2. **Train Model**: Run the script to perform data loading, vectorization, and model training. The model includes early stopping and checkpoints to save the best-performing weights.
3. **Evaluate Model**: After training, the model is evaluated on the test set, and the mean squared error of predictions is displayed.

## Results

The model outputs the mean squared error after evaluation, which reflects the accuracy of the predictions. The histogram of document frequencies is also plotted to visualize the distribution of terms in the reviews.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
