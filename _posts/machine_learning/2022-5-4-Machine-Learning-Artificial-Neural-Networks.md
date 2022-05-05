---
layout:     post
title:      "Machine Learning: Artificial Neural Networks"
subtitle:   "Artificial Neural Network with Keras"
date:       2022-5-4 19:00:00
author:     "Yingfan"
catalog: true
header-style: text
mathjax: true
tags:
  - machine learning
  - lecture notes
  - model
---

Artificial Neural Networks (ANN) is a Machine Learning model inspired by the networks of biological neurons found in our brains

# The Perceptron

- one of the simplest ANN architectures 
- **Threshold logic unit (TLU)**: also called a linear threshold unit
  - TLU computes a weighted sum of its inputs, then applies a **step function**
  - A single TLU can be used for simple linear binary classification

![](/img/in-post/post-tlu.png)

- **Perceptron** 
  - composed of a single layer of  TLUs
  - each TLU connects to all the inputs
  - All the input neurons form the input layer.
  - an extra bias feature is generally added (x0 = 1): it is typically represented using a special type of neuron called **a bias neuron**, which outputs 1 all the time
- **structure**
  - ![](/img/in-post/post-perceptron.png)
- **formula for outputs**
  - $$h_{W,b}(X)=\phi(XW+b)$$
  - $X$: input features, one row per instance, one col per feature
  - $W$: connection weights except for the one from the bias neuron (yellow circle) , one row per neuron, one col per unit in the layer
  - $b$: all the connection weights between the bias neuron and the neurons in the output layer
  - $\phi$: the **activation function**: when the artificial neurons are TLUs, it is a **step function**
- **learning rule**
  - The Perceptron is fed one training instance at a time
  - for each instance it makes its predictions and adjusts the weights
  - **Linear decision boundary**, only solve linearly separable problem
- **Multilayer Perceptron (MLP)**
  - 
- **Notes**
  - Perceptrons make predictions based on a hard threshold, 
  - 