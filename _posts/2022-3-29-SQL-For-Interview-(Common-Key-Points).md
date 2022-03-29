---
layout:     post
title:      "SQL For Interview (Common Key Points)"
subtitle:   "Study notes of Liuge's blogs"
date:       2022-3-29 13:00:00
author:     "Yingfan"
catalog: true
header-style: text
tags:
  - sql
  - study notes
  - interview
---

## Join

> Hive support common SQL join statement, but only support equijoin.

- **INNER JOIN**
- **LEFT JOIN**: 右边列中没有左边匹配的记录时会是NULL
- RIGHT JOIN
- FULL JOIN: 两个表中所有符合WHERE条件的记录

## Group BY

常用的聚合函数有

- count
  - count(*): 总行数, 包括NULL值
  - count(expr), count(distinct expr): 不包括null
- sum
  - sum(col): 组内查询列的和
  - sum(distinct col): 查询列不同值得和
- avg
  - avg(col), avg(distinct col)
- min, max, variance
- corr(col1, col2)
- percentile_approx(col, array(p1,p2,...))

## 列转行

列转行作为SQL高级应用函数，是各大厂高频考点，根本就是考察Lateral View的用法。

- **Lateral view  grammar**

  ```sql
  # lateral view:
  LATERAL VIEW udtf (expression) tableAlias AS coluumAlias
  
  # use with select, from clause
  select 
  from base table LATERAL VIEW udtf (expression) tableAlias AS coluumAlias
  ```

- Notes

  - lateral view一般和udtf (user defined table-generating functions)一起使用，如explode, split
  - lateral view 首先将utdf函数应用到每一行上，这时每一行经utdf处理后得到多行输出，这些输出将会组建成一张虚拟表
  - 这张虚拟表会跟当前表进行inner join操作，join完成之后会得出一张结果虚拟表

- **Outer关键字**

  - ```sql
    select 
    from base table OUTER LATERAL VIEW ...
    ```

  - 避免当udtf 没有得到任何结果时最终虚拟结果表里丢失原数据行的问题

- **Examples**

  Table pageAds:

  | column name | column type |
  | ----------- | ----------- |
  | page_id     | string      |
  | adid_list   | array<int>  |

  | page_id      | adid_list |
  | ------------ | --------- |
  | front_page   | [1,2,3]   |
  | contact_page | [3,4,5]   |
  | end_page     | []        |

  Task:

  在所有页面中统计每条广告出现的次数，输出广告id和该广告出现的总次数。

  Solution:

  ```sql
  select ad_id, count(*)
  from pageAds OUTER LATERAL VIEW explode(adid_list) adTable AS ad_id
  group by ad_id
  ```

  