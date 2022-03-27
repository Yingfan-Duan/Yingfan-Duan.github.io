---
layout:     post
title:      "Data Analysis Methods(1)"
subtitle:   "Study notes of Liuge's blogs"
date:       2022-3-26 18:00:00
author:     "Yingfan"
catalog: true
header-style: text
tags:
  - data analysis
  - study notes
  - interview
---

Study notes based on [Liuge's articles](https://www.nowcoder.com/profile/163768403/myDiscussPost?page=1).

在数据分析中常会用到的方法论有：

- 方差分析
- 描述性分析
- 相关性分析
- 参数估计
- 幸存者偏差
- 辛普森悖论
- RFM分析模型
- AARRR模型
- SWOT矩阵
- MECE分析模型
- 漏斗分析模型
- 下钻分析（维度拆分）

本文将逐一梳理这些方法论的概念及应用情景。

## 描述性分析

描述性分析是用来测量和描述一个分布的各种统计量，可以分为集中趋势、离散程度、分布形状

| category             | statistics                         | Notes                                                        |
| -------------------- | ---------------------------------- | ------------------------------------------------------------ |
| 集中趋势             | mode, median, quantile, mean       | NULL                                                         |
| 离散程度(dispersion) | variation ratio(异众比率)          | 非众数组的频数占总频数的比例；越大说明众数代表性越差；       |
|                      | Interquartile range(四分位距)      | 四分位差不受极值的影响；一定程度上说明了中位数对一组数据的代表程度 |
|                      | 方差，标准差                       | NULL                                                         |
|                      | coefficient of variation(离散系数) | $$cv=\frac{\sigma}{\mu}$$; allow relative comparison of two measurement(单位不同) |
| 分布形状             | skewness(偏度)                     | ![](/img/in-post/post-stats/post-skewness.png)               |
|                      | Kurtosis(峰度)                     | 如果峰度大于三，峰的形状比较尖，比正态分布峰要陡峭           |

**Example**: 某电商平台用户的平均成交金额为20，成交金额的中位数为0，标准差为80，用户成交的分布呈现左偏长尾，请问从以上数据可以得出哪些结论？

**Ans**:

- 标准差是平均成交金额的4倍，说明用户成交金额较为分散
- 中位数为0元，说明平台一半以上的用户无成交记录
- 左偏长尾，说明存在大额成交用户，但是此种用户数量较少

## 相关性分析

| statistics | formula/value                                                | use cases                                                    |
| ---------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| Pearson    | $$\rho_{X,Y}=\frac{cov(X,Y)}{\sigma_X\sigma_Y}$$             | 两个变量之间是**线性关系**，且是**连续数据**；总体都是正**态分布**，或接近正态的单峰分布；两个变量的观测值是成对的，每对观测值之间**相互独立**。 |
| Kendall    | 当τ为1时，表示两个随机变量拥有一致的等级相关性； 当τ为-1时，表示两个随机变量拥有完全相反的等级相关性； 当τ为0时，表示两个随机变量是相互独立的 | 有序分类的两个分类变量                                       |
| Spearman   |                                                              | 同上                                                         |

[**Spearman correlation vs Kendall correlation**](https://datascience.stackexchange.com/a/64261)

- In the normal case, *Kendall correlation* is more **robust** and **efficient** than *Spearman correlation*. <u>It means that *Kendall correlation* is preferred when there are small samples or some outliers.</u>
- *Kendall correlation* has a O(n^2) computation complexity comparing with O(n logn) of *Spearman correlation*, where n is the sample size.
- *Spearman’s rho* usually is larger than *Kendall’s tau*.
- The **interpretation** of *Kendall’s tau* in terms of the probabilities of observing the agreeable (concordant) and non-agreeable (discordant) pairs is very direct.

## AAARRR模型

### Overview

AAARRR stands for Awareness, Acquisition、Activation、Retention、Revenue、Referral.

![](/img/in-post/post-stats/post-AARRR.jpg)

而在以上几个环节中，每个环节都有我们需要关注的指标，下面一一介绍。



## RFM模型



## 波士顿矩阵
