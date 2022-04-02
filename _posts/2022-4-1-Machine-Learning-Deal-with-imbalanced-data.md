---
layout:     post
title:      "Machine Learning: Deal with imbalanced data"
subtitle:   "A guide to handling imbalanced data in python"
date:       2022-4-1 16:00:00
author:     "Yingfan"
catalog: true
header-style: text
tags:
  - machine learning
  - data processing
  - study notes
---

Imbalanced dataset is a common problem in classification problem where there is a disproportionate ratio of samples in each class of ``y`` column. This could happen in cases like spam filtering, fraud detection. 

In this blog, we will cover five possible ways to handle this problem in python specifically and discuss their use cases as well.

## Change the performance metric

When we have imbalanced dataset, **accuracy** is not the best metric to use when evaluating imbalanced datasets as it can be very misleading.  So Metrics that can provide better insight include:

- **Confusion matrix**
  - a table showing correct predictions and types of incorrect predictions.
  - ![](/img/in-post/post-confusion-matrix.png)
- **Precision**
  - the number of true positives(TP) divided by all positive predictions
  - a measure of a classifier’s **exactness**
  - Low precision indicates a high number of false positives.
- **Recall**
  - the number of true positives(TP) divided by the number of actually positive values
  - also called **Sensitivity** or the **True Positive Rate** (TPR)
  - a measure of a classifier’s **completeness**
- **F1 score**
  - the weighted average of precision and recall.
  - $$f1=\frac{precision\cdot recall}{precision+recall}$$
- **PR AUC**
  - The area under PR curve
  - PR (Precision-Recall) curve is a curve that combines precision (PPV) and Recall (TPR) in a single visualization. For every threshold, you calculate PPV and TPR and plot it. The higher on y-axis your curve is the better your model performance.



## Change the algorithm



## Resampling Techniques



## Generate synthetic samples



## Utilize sklearn parameters







•Random Under Sampling (RUS): throw away data, computationally efficient

•Random Over Sampling (ROS): straightforward and simple, but training your model on many duplicates

•Synthetic Minority Oversampling Technique (SMOTE): more sophisticated and realistic dataset, but you are training on "fake" data (not on test set!)

