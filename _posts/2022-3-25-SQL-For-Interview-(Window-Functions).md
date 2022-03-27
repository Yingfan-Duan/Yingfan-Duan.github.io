---
layout:     post
title:      "SQL For Interview (Window Functions)"
subtitle:   "Study notes of Liuge's blogs"
date:       2022-3-25 18:00:00
author:     "Yingfan"
catalog: false
header-style: text
tags:
  - SQL
  - study notes
  - interview
---

SQL study notes based on [this blog](https://mp.weixin.qq.com/s/TJzvYLB52MjjnJRx6XQIrA).

## Common Window Functions

1. sum()over(): 常用来多维度分组求和、求累加值等
2. count()over(): 常用来多维度分组计数、计算汇总计数等
3. min, max, avg()over(): 常用来计算指定分组列对应某指标的最大、最小、平均值
4. sorting by group related functions:
   - lag()over()
   - lead()over()
   - row_number()over()

## Function Usage

```sql
window_func() over (partition by [<col1>,<col2>,…]
[order by <col1>[asc/desc], <col2>[asc/desc]…] <窗口选取语句: windowing_clause>))
```

| part             | details                                                      |
| ---------------- | ------------------------------------------------------------ |
| window_func      | 常用：sum, count, avg, max, row_number, rank, dense_rank, first_value, last_value, lag, lead |
| partition by     | 指定分组的列名， 可以不指定，查询分区子句                    |
| order by         | 可选，默认acs                                                |
| windowing_clause | 可选，比partition by更细粒度的划分分组范围。 **rows between x preceding / following and y preceding / following**：表示窗口范围是从前或后x行到前或后y行，区间表示法即：[x,y]。 **rows x preceding / following**：窗口范围是从前或后第x行到当前行。区间表示法即[-x,0]、[0,x]。 |

**Notice**:

- **window function**不能和同级别聚合函数一起使用，不能和聚合函数嵌套使用

- **partition by:**

  - <u>与聚合函数group by不同的地方</u>：不会减少表中记录的行数，而group by是对原始数据进行聚合统计，一般只有一条反映统计值的结果（每组返回一条）。
  - over之前的函数在每一个分组之内进行，若超出分组，函数会重新计算。

- **order by**

  - order by默认情况下聚合从起始行到当前行的数据
  - 该子句对于<u>排序类函数</u>是必须的，因为如果数据无序，这些函数的结果没有任何意义。

- **windowing_clause**

  - 从句缺失时：
- - order by 指定，窗口从句缺失，则窗口的默认值为range between unbounded preceding and current row，也就是从第一行到当前行；
    - order by 和窗口从句如果都缺失，则窗口的默认值为range between unbounded preceding and unbounded following，即从第一行到最后一行。
  - <u>序列函数</u>(row_number, rank, etc)不支持窗口选取子句。

  > row_number, rank, dense_rank这三个函数的区别主要在分数一致的情况下，row_number()不重复排序，rank()重复且跳数字排序，dense_rank()重复且不跳数字排序

- execution order

  - from -> where -> group by -> having -> select -> **window func** -> order by -> limit
  - 可以理解为窗口函数是将select中的结果数据集当做 输入 再次加工处理。

## Examples

data table:

![](/img/in-post/post-sql/post-hive-ex1.png)

1. 统计淘宝2021年4月各行业购买订单数及总购买订单数

   ```sql
   select industry, order_num, sum(order_num) over() as order_total
   from
   (
       select industry, count(user_id) as order_num
   	from emp_table
   	where substr(order_date,1,7)="2021-04"
   	group by industry
   )tt
   
   ```

   ![](/img/in-post/post-sql/post-hive-ex2.png)

2. 统计顾客的购买明细及月购买总额

   ```sql
   select user_id, usr_name, industry, order_date, money, sum(money)over(partition by month(order_date)) as month_money
   from emp_table
   ```

   ![](/img/in-post/post-sql/post-hive-ex3.png)

3. 统计各行业月订单金额，并求按月的累计金额

   ```sql
   select industry, month_date, month_money, sum(month_money)over(partition by industry order by month_date) as agg_month_money
   from
   (
       select industry, substr(order_date,1,7) as month_date, sum(money) as month_money
       from emp_table
       group by industry, substr(order_date,1,7)
   )tt
   
   ```

   ![](/img/in-post/post-sql/post-hive-ex4.png)

4. 各种聚合作用说明

   | statement                                                    | meaning                      |
   | ------------------------------------------------------------ | ---------------------------- |
   | sum(x) over(partition by industry order by month_date rows between unbounded preceding and current row) | 由组第一行到当前行的聚合     |
   | sum(x) over(partition by industry order by month_date rows between 1 perceding and current row) | 组内当前行和前面一行的聚合   |
   | sum(x) over(partition by industry order by month_date rows 1 preceding) | dito                         |
   | sum(x) over(partition by industry order by month_date rows between 1 preceding and 1 following) | 组内当前行和前后一行的聚合   |
   | sum(x) over(partition by industry order by month_date rows between current row and unbounded following | 组内当前行及后面所有行的聚合 |

   
