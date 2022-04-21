---
layout:     post
title:      "Machine Learning: Ensemble Learning and Random Forest"
subtitle:   "A thorough introduction to Ensemble learning"
date:       2022-4-20 21:00:00
author:     "Yingfan"
catalog: true
header-style: text
mathjax: true
tags:
  - machine learning
  - lecture notes
  - model
---

A group of predictors is called an **ensemble**; thus, this technique is called **Ensemble Learning**, and an Ensemble Learning algorithm is called an **Ensemble method**.

# Voting classifier

**Hard voting classifier**

Aggregate the predictions of each classifier and **predict the class that gets the most votes.**

**Soft voting classifier**

Aggregate the predictions of each classifier and predict the class with **the highest class probability**, averaged over all the individual classifiers

**Characteristics**

- provided there are a sufficient number of weak learners and they are sufficiently diverse, even if each classifier is a weak learner (meaning it does only slightly better than random guessing), the ensemble can still be a strong learner (achieving high accuracy)
- Ensemble methods work best when the predictors are as independent from one another as possible.
- Soft voting classifier often achieves higher performance than hard voting because it gives more weight to highly confident votes.

**Implementation**

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

log_clf = LogisticRegression(solver="lbfgs", random_state=42)
rnd_clf = RandomForestClassifier(n_estimators=100, random_state=42)
svm_clf = SVC(gamma="scale", random_state=42)

voting_clf = VotingClassifier(
    estimators=[('lr', log_clf), ('rf', rnd_clf), ('svc', svm_clf)],
    voting='hard')


log_clf = LogisticRegression(solver="lbfgs", random_state=42)
rnd_clf = RandomForestClassifier(n_estimators=100, random_state=42)
# need to set probability = True
svm_clf = SVC(gamma="scale", probability=True, random_state=42)  

voting_clf = VotingClassifier(
    estimators=[('lr', log_clf), ('rf', rnd_clf), ('svc', svm_clf)],
    voting='soft')
voting_clf.fit(X_train, y_train)
```

# Bagging

- use the same training algorithm for every predictor and **train them on different random subsets** of the training set.
  - bagging: sampling is performed with replacement
  - pasting: sampling is performed without replacement
- Bagging is an alternative way when you can't have enough data and want to solve overfitting problem
- bagging and pasting can **scale very well** so that they are very popular
- sklearn: `BaggingClassifier` and `BaggingRegressor`

**Out-of-Bag evaluation**

- By default a BaggingClassifier samples m training instances with replacement (bootstrap=True), where m is the size of the training set.
- This means that only about 63% of the training instances are sampled on average for each predictor. 
- The remaining 37% of the training instances that are not sampled are called **out-of-bag (oob) instances**.
  - they are not the same 37% for all predictors.
- So the predictor can be evaluated on these instances, without the need for a separate validation set.
- Then we can evaluate the **ensemble** itself by averaging out the oob evaluations of each predictor.
- set hyperparameter`oob_score=True` can request an automatic oob evaluation after training

**Random Patches method**

- Sampling both training instances and features
- sampling instances: controlled by `max_samples` and `bootstrap`
- sampling features: controlled by max_features and bootstrap_features
- useful when you are **dealing with high-dimensional inputs** (such as images).

**Random subspaces method**

- only sample features but keep all training instances
- bootstrap=False, max_samples=1
- bootstrap_features=True or max_features<1

**Random Forest**

- an ensemble of Decision Trees
- trained via the bagging method (or sometimes pasting), typically with max_samples set to the size of the training set.
- introduces extra randomness when growing trees
  - when splitting a node it searches for the best feature **among a random subset of features**

**Extra Trees**

- At each node, it makes trees even more random by also **using random thresholds for each feature** rather than searching for the best possible thresholds
- much faster to train than regular Random Forests
- `ExtraTreesClassifier` and `ExtraTreesRegressor`

**Feature Importance in RF**

- measure the **relative** importance of each feature
- Scikit-Learn measures a feature’s importance by looking at how much the tree nodes that use that feature **reduce impurity on average** (across all trees in the forest)
- access the result using the `feature_importances_` variable of the classifier/regressor.

# Boosting

- **The general idea** of most boosting methods is to train predictors sequentially, each trying to correct its predecessor
- The **drawback** of boosting methods: they are sequential and cannot be parallelized

**AdaBoost**

- For a new predictor, pay a bit more attention to the training instances that the predecessor underfitted.
- This results in new predictors focusing more and more on the hard cases
- increases the relative weight of misclassified training instances in each new predictor
- To **make predictions**, AdaBoost simply computes the predictions of all the predictors and weighs them using the predictor weights αj

**Gradient Boosting**

- fit the new predictor to the residual errors made by the previous predictor instead of tweaking the instance weights at every iteration
- The `learning_rate` hyperparameter in `GradientBoostingRegressor` scales the contribution of each tree
  - low learning_rate: more trees in the ensemble to fit, predictions will generalize better

# Stacking

- train a model to perform the aggregation of all predictors' prediction
