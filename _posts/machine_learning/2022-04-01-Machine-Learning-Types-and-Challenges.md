---
title: "Machine Learning: Types and Challenges"
subtitle: "lecture notes for machine learning course"
layout: post
author: "Yingfan"
header-style: text
tags:
  - lecture notes
  - machine learning
---



> "*Machine Learning is the science and art of programming computers so they can learn from data.*"

# Types of Machine Learning

Machine Learning systems can be classified according to the amount and type of supervision they get during training。

## Supervised/Unsupervised Learning 

### Supervised Learning 

- the training set you feed to the algorithm includes the desired solutions, called **labels**
- including classification and regression
- most important supervised learning algorithms：
  - k-Nearest Neighbors 
  - Linear Regression 
  - Logistic Regression 
  - Support Vector Machines (SVMs) 
  - Decision Trees and Random Forests 
  - Neural networks

### Unsupervised Learning 

- the training data is unlabeled
- important unsupervised learning algorithms
  - clustering
    - K-means
    - DBSCAN
    - Hierarchical Cluster Analysis(HCA)
  - Anomaly detection and novelty detection
    - One-class SVM
    - Isolation Forest
  - Visualization and dimensionality reduction
    - PCA, Kernel PCA
    - Locally Linear Embedding (LLE)
    - t-SNE
  - Association rule learning
    - Apriori
    - Eclat

### Semi-supervised Learning 

- deal with data that’s partially labeled；plenty of unlabeled instances, and few labeled instances
- example
  - photo-hosting services, such as Google Photos
  - unsupervised part: automatically recognizes that the same person A shows up in photos 1, 5, and 11, while another person B shows up in photos 2, 5, and 7.
  - supervised part: Just add one label per person and it is able to name everyone in every photo, which is useful for searching photos.

### Reinforcement Learning

- The learning system, called an agent in this context, can **observe** the environment, **select** and **perform** actions, and get **rewards** in return
- learn by itself what is the best strategy, called a **policy**, to get the most reward over time
- A policy defines what action the agent should choose when it is in a given situation

## Batch vs Online Learning 

### Batch Learning (offline learning)

- the system is incapable of learning incrementally
- must be trained using all the available data
- **Drawbacks**: take a lot of time and computing resources
- typically trained **offline**
- Trained system, then launch into production, runs without learning anymore
- learn about new data: train a new version of the system from scratch on the full dataset (not just the new data, but also the old data), then replace the old system

### Online Learning 

- train the system incrementally by feeding it data instances sequentially (individually or in small groups called mini-batches)
- **Advantages**: great for systems
  -  that receive data as a continuous flow
  - that need to adapt to change rapidly or autonomously
- can also be used to train systems on huge datasets that cannot fit in one machine’s main memory (out-of-core learning)
  - loads part of the data
  - runs a training step on that data
  - repeats the process until it has run on all of the data

## Instance-Based vs Model-Based Learning 

categorize Machine Learning systems is by how they generalize. When making predictions, the system needs to be able to make good predictions for (generalize to) examples it has never seen before given a number of training examples.

### Instance-Based Learning 

Take spam filter as an example. 

- flag all emails that are identical to emails that have already been flagged by users
- also flag emails that are very similar to known spam email

These two methods are both instance-based learning.

### Model-Based Learning

Another way to generalize from a set of examples is to build a model of these examples and then use that model to make predictions

# Main Challenges of Machine Learning

### Challenges

- **Insufficient quantity of training data** 
- **Non representative Training Data**
  - In order to generalize well, it is crucial that your training data be representative of the new cases you want to generalize to
- **Poor-Quality Data**
  - if the training data is full of errors, outliers, and noise (e.g., due to poor quality measurements), it will make it harder for the system to detect the underlying patterns, so your system is less likely to perform well
- **Irrelevant features**  -- feature engineering process
  - **Feature selection** - selecting the most useful features to train on among existing features 
  - **Feature extraction** - combining existing features to produce a more useful one
  -  **Creating** new features by gathering new data
- **Overfitting** 
  - model is too complicated for the data, fit the training data very well, but would not generalize on the new data it was not trained on
  - **solutions**
    - Simplify the model by 
      - selecting one with fewer parameters
      - reducing the number of attributes in the training data
      - constraining the model using regularization
    - Gather more training data.
    - Reduce the noise in the training data
- **Underfitting**
  - your model is too simple to learn the underlying structure of the data
  - **solutions**
    - Select a more powerful model, with more parameters. 
    - Feed better features to the learning algorithm (feature engineering). 
    - Reduce the constraints on the model (e.g., reduce the regularization hyperparameter).

### Hyperparameter Tuning and Model Selection: holdout validation

The error on new cases is called the **generalization error** or **out-of-sample error**

Train-Test split solve the problem of evaluating generalization ability of our model. **Holdout validation** can help us to select between models, tune hyperparameters.

- holdout validation: 
  - **part of the training set** is reserved for validation set also called development set or dev set.
  - train multiple models with various hyperparameters on the reduced training set: full training set - validation set
  - select the model that performs best on the validation set 
  - train the best model on the full training set and evaluate it on the test set

## Extra Notes

### Norms

- MAE corresponds to l1 norm of a vector 
- RMSE corresponds to l2 norm of a vector
- The higher the norm index, the more the metrics focuses on large values, neglecting the small ones.
- So RMSE is more sensitive to outliers than the MAE.
- When outliers are exponentially rare (like in a bell-shaped curve), the RMSE performs very well and is generally preferred.

### Train, test, validation sets

- Don’t do data exploration on the whole data set - data snooping - not to bias yourself, use a training set
- Use `train_test_split()` for randomly splitting data into train and test set 
- Use `StratifiedShuffleSplit()` to split data in such a way that a particular variable is properly represented in each subset 
- If you need to select between different models or different hyperparameters, either reserve part of the train set as a validation set or even better (but more computationally expensive) use K-fold cross-validation

> K-fold cross-validation: 
>
> - Shuffle the dataset randomly.
> - Split the dataset into k groups 
> - For each unique group: 
>   - Take the group as a hold out or test data set 
>   - Take the remaining groups as a training data set 
>   - Fit a model on the training set and evaluate it on the test set 
>   - Retain the evaluation score and discard the model
> - Summarize the performance of the model using the sample of model evaluation scores
>
> Common values are k = 3, k = 5, and k = 10



