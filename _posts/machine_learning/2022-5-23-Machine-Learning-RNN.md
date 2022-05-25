---
layout:     post
title:      "Machine Learning: RNN"
subtitle:   "RNN Introduction"
date:       2022-5-23 23:00:00
author:     "Yingfan"
catalog: true
header-style: text
mathjax: true
tags:
  - machine learning
  - lecture notes
  - model
---

In this lecture we shall discuss how one can predict the future using recurrent neural networks (RNN).

# Introduction

RNNs are capable of handing sequential data. **Sequential Data** refers to any data that contain elements that are ordered into sequences.

### Recurrent Neurons and Layers

A recurrent neural network looks very much like a feedforward neural network, except it also has connections pointing backward.

> Feedforward NN: the activations flow only in one direction, from the input layer to the output layer

The simplest possible RNN is composed of one neuron receiving inputs, producing an output, and sending that output back to itself

**Recurrent Neurons**

![](/img/in-post/post-recurrent-neuron.png)

At each time step $t$ (also called a **frame**), **this recurrent neuron receives the inputs $x_(t)$ as well as its own output from the previous time step, $y_(t–1)$**

**Recurrent Layers**

