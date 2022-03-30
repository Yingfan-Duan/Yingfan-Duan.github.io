---
layout:     post
title:      "Data Analysis Methods(1)"
subtitle:   "Study notes of Liuge's blogs"
date:       2022-3-26 18:00:00
author:     "Yingfan"
catalog: true
header-style: text
mathjax: true
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

## 幸存者偏差

## 辛普森悖论



## AAARRR模型

### Overview

AAARRR stands for Awareness, Acquisition、Activation、Retention、Revenue、Referral.

![](/img/in-post/post-stats/post-AARRR.jpg)

而在以上几个环节中，每个环节都有我们需要关注的指标，下面一一介绍。

### Acquisition(获客)

- Meaning: 用户如何找到我们的产品、服务或品牌
- key factors: 
  - 载体：海报，电视剧，
  - 渠道
    - 口碑渠道：适合病毒式传播
    - 自然流量：搜索引擎，内容营销等
    - 付费传播：电视剧广告，赞助商等
- KPIs
  - 渠道
    - 自然流量：曝光量，转化率
    - 付费流量：曝光量，转化率，成本
  - 平台数据
    - 新增用户数，APP下载量，平均获客成本

### Activation(激活用户)

比如某电商平台，作为新用户如果需要完成首次购买行为，必须经历以下步骤：

1. 下载APP，或微信小程序/网页
2. 注册账号
3. 找到想要购买的商品
4. 加入购物车
5. 输入收货地址
6. 点击支付购买

在这一系列的动作中，作为BI，可以检测用户到底是哪一步漏斗转换较低导致最后转换失败：

- 是注册账号太麻烦？
- 还是用户找不到想要买的商品？
- 还是收货地址输入太费劲？
- 或者是支付环节不友好？
- 或者是价格太高等等？

只有优化整个产品体验，加大用户操作的便捷性，才能让用户更好的转换。

- **激活用户的措施**
  - 优化产品
  - 小游戏
  - 优惠券，降低用户购买的成本
  - 新客优惠，设置新客优惠价，新客专享

- **KPIs**
  - 在激活用户环节，最关键的就是有多少用户真正的使用了产品。我们将使用了产品的用户定义为活跃用户，比如电商平台的活跃用户即为购买用户。
  - 活跃用户：日活，周活，月活
  - 活跃用户占比（在总用户中）
  - 访问深度
  - 停留时长

### Retention(用户留存)

对已经激活的用户，是否还愿意继续使用产品。

- **留存用户的措施**

  - 签到（获取积分）
  - 小游戏，如蚂蚁森林
  - 会员服务，如京东plus会员
  - 这些服务的目标即希望用户养成打开和使用产品的习惯，类似京东的签到活动，即便是用户当下无购物需求，但是为了京豆还是愿意打开APP，和产品交互的过程，也、是激发用户购物的重要环节。

- **KPIs**

  - 次留
  - 3日留存
  - 7日留存
  - N日留存

  > 对于不同的业务和产品，到底看几日留存较为合适呢？这个要根据用户使用的频率来定义，比如对电商平台，次留，7日留存是非常关键的。对于即时通讯产品（如微信）次留是非常重要的。

### Revenue(产品收入)

不同的产品实现盈利的方式不一样。

比如，电商平台通过商家抽佣、排序广告等方式实现盈利。

短视频平台作为流量分发机器，通过广告等实现盈利。据彭博社消息，字节跳动2020年广告收入1831亿元，从收入构成来看，广告依然是现金牛，占其2020年实际收入的77%。

- **KPIs**

  > 下面的核心指标对部分产品可能不适用。比如客单价适用电商平台，不适用微信平台。

  - 用户：普通用户，付费用户，付费用户占比
  - 广告收入：曝光，转换
  - 用户价值：
    - LTV(Lift time value): 用户首次注册账户到最后一次购买整个期间对平台的收入贡献。
    - ARPU(Average revenue per user)

### Referral(产品推荐)

推荐环节也叫病毒式营销，这一环节说明有多少用户愿意为产品进行宣传。

比如电影，通过朋友圈晒图，可以获取好评电影并激发用户购票观看的需求，所以对于电影行业来说，推荐环节至关重要。

- **KPIs**
  - 分享次数
  - 裂变层数
  - 推荐曝光量
  - 转化率

## RFM模型

RFM模型是衡量客户价值和客户创利能力的重要工具和手段。

在众多的客户关系管理(CRM)的分析模式中，RFM模型是被广泛提到的。该模型涉及到**以下三个变量**：

- R-最近一次消费距当前的时长(Recency)
- F-消费频率(Frequency)
- M-消费金额(Monetary)

这篇[文章](https://www.jianshu.com/p/4b60880f24e2)中有详细举例和分析。

**Example:** 在电商领域，对重要挽留客户有哪些措施可以进行干预？

- 流失原因：首先可以通过电话调研确认用户流失的原因
- 营销手段：采取营销手段进行干预挽留，比如发放优惠券、低价商品引流
- 产品优化：如果是产品或者功能上的问题，可以通过优化产品提高用户体验减少流失

## 波士顿矩阵

波士顿矩阵通过销售增长率和市场占有率两个指标对公司和产品进行四象限分类。

![](/img/in-post/post-stats/post-boston-analysis.jpg)

具体分析方法
