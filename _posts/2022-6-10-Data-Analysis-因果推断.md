---
layout:     post
title:      "Data Analysis: 因果推断"
subtitle:   "因果推断方法介绍"
date:       2022-6-10 9:00:00
author:     "Yingfan"
catalog: false
header-style: text
mathjax: true
tags:
  - data analysis
  - interview
---

# Correlation Does Not Imply Causation

相关性通常是对称的，因果性通常是不对称的（单向箭头），相关性不一定说明了因果性，但因果性一般都会在统计层面导致相关性。这篇[论文](http://web.cs.ucla.edu/~kaoru/3-layer-causal-hierarchy.pdf)中提到了 **“The Three Layer Causal Hierarchy”**的概念。

| level                                          | typical activity | example                                                      |
| ---------------------------------------------- | ---------------- | ------------------------------------------------------------ |
| 关联（Association）$P(y|x)$                    | seeing           | How would seeing X change my belief in Y ?                   |
| 干预（Intervention）$P(y|do(x),z)$             | doing            | what if I do X?                                              |
| 反事实推断（Counterfactuals）$P(y_x|x^1, y^1)$ | imagining        | What if I had acted differently? <br />Was it X that caused Y? |

