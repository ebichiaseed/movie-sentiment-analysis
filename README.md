# Movie Sentiment Analysis Project

**Authors:** Tan Qing Zhe, Chia Kai En Erika, Tan Shyan Q’yan Kate

## Introduction

In this project, we perform sentiment analysis on movie reviews to determine whether a review expresses a positive or negative sentiment. Movie sentiment analysis is a valuable tool for understanding audience feedback, predicting box-office success, and tailoring marketing strategies. We explore several methods ranging from traditional machine learning to deep learning approaches to compare their effectiveness on this task.

## Methods

We implemented and evaluated the following approaches:

### Text Preprocessing:
- Clean the text data by removing noise and standardizing the reviews.
- Tokenize the reviews and remove standardized stop words while retaining key negation words.

### Feature Extraction for ML Methods:
- Generate features using TF-IDF vectorization.
- Generate features using Count Vectorizer (Bag of Words).

### Traditional Machine Learning:
- Logistic Regression with TF-IDF vectorization: **Achieved 89.8% accuracy**
- Naïve Bayes  
- Random Forest  
- Support Vector Machines (SVM)

### Deep Learning:
- **DistilBERT for Sequence Classification**: Achieved **93.4% accuracy**, providing deep, context-aware embeddings that capture semantic nuances in movie reviews.

### Rule-Based Approach:
- **VADER**: A rule-based sentiment analyzer tailored for social media text.

### Hyperparameter Tuning:
- We performed hyperparameter tuning using **Bayesian Optimization**, **Randomized Search**, and **Grid Search** to optimize model performance.

## How to Run the Project

To run the project, please execute the provided Google Colab notebooks in the following order:

1. `Movie Sentiment EDA.ipynb`  
   - Explore and visualize the dataset to understand its structure and sentiment distributions.

2. `Movie Sentiment Vectorisation and ML experimentation.ipynb`  
   - Preprocess the data, generate vector representations (BoW and TF-IDF), and experiment with traditional machine learning models.

3. `Movie Sentiment Hyperparameter Tuning.ipynb`  
   - Perform hyperparameter tuning using Randomized Search and Grid Search for the machine learning models.

4. `Movie Sentiment BERT Experimentation.ipynb`  
   - Fine-tune and evaluate the DistilBERT model on the sentiment analysis task.

5. `Movie Sentiment Bayesian Optimization BERT.ipynb`  
   - Apply Bayesian optimization to further fine-tune the DistilBERT model.

6. `Movie Sentiment Vader.ipynb`  
   - Evaluate the performance of the VADER sentiment analysis tool on the movie reviews.

> Make sure to run the notebooks in the order listed above for proper execution of the complete workflow.

## Best Models

| Model                                      | Accuracy |
|-------------------------------------------|----------|
| Best ML Model (Logistic Regression + TF-IDF) | 89.8%    |
| Best DL Model (DistilBERT with fine-tuning) | 93.4%    |

These models demonstrate the strength of both deep learning and traditional machine learning approaches in handling movie sentiment analysis.
