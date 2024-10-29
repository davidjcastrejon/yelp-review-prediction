# Yelp Review Prediction

This repository contains a machine learning project that predicts the star ratings of Yelp reviews based on their textual content. The model uses TensorFlow and Keras to build a neural network that performs regression on the star ratings.

## Table of Contents

- [Overview](#overview)
- [Requirements](#requirements)
- [Data Preprocessing](#data-preprocessing)
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

## Data Preprocessing

This project involves preprocessing Yelp's academic dataset to filter and refine the data for further analysis. The preprocessing steps are as follows:

1. **Loading the Dataset**:
   - The initial dataset consists of several JSON files, including `yelp_academic_dataset_business.json`, `yelp_academic_dataset_checkin.json`, `yelp_academic_dataset_review.json`, and `yelp_academic_dataset_tip.json`.
   - Each JSON file is read line by line, and each line is parsed into a Python dictionary.

2. **Filtering Business Data**:
   - The business data is filtered to retain only those entries with a review count of 20 or more. This is done to ensure that only businesses with a sufficient amount of feedback are included for analysis.
   - The filtered business data is saved to a new JSON file named `filtered_yelp_academic_dataset_business.json`.

3. **Extracting Business IDs**:
   - The filtered business data is then used to extract the unique business IDs, which will be used for filtering related data in other datasets.

4. **Filtering Path Data**:
   - For each of the other datasets (`checkin`, `review`, `tip`), entries are filtered based on the business IDs obtained from the filtered business dataset. Only entries with business IDs present in the filtered business dataset are retained.
   - The filtered path data is saved to new JSON files named `filtered_yelp_academic_dataset_checkin.json`, `filtered_yelp_academic_dataset_review.json`, and `filtered_yelp_academic_dataset_tip.json`.

5. **Summary of Data After Filtering**:
   - The lengths of the unfiltered and filtered datasets are printed to provide insight into the reduction in data size:
     - Length of unfiltered business data: 150,346 lines
     - Length of filtered business data: 61,919 lines
     - Length of filtered checkin data: 79,945 lines
     - Length of filtered tip data: 60,448 lines

By following these steps, this ensures that the dataset is clean, relevant, and ready for subsequent analyses or machine learning applications.

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
