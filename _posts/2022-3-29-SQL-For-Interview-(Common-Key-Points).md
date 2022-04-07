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


## Create & Insert 

- Create table (common empty table)

  ```sql
  create table if not exists actor(
      actor_id smallint(5) not null comment '主键id',
      first_name varchar(45) not null comment '名字',
      last_name varchar(45) not null comment '姓氏',
      last_update date not null comment '日期',
      primary key (actor_id))
  ```

- create duplicate table 

  ```sql
  create table table name like table name2
  ```

  

- create table 1 from part of table 2

  ```sql
  create table if not exixts actor_name
  (first_name varchar(45) not null,
   last_name  varchar(45) not null)
  select first_name,last_name from actor;
  ```

## create index

- **use ALTER**

  - 添加**主键**

    `````SQL
    ALTER TABLE tbl_name ADD PRIMARY KEY (col_list);``// 该语句添加一个主键，这意味着索引值必须是唯一的，且不能为NULL。
    `````

  - 添加**唯一索引**

    ```sql
    ALTER TABLE tbl_name ADD UNIQUE index_name (col_list);``// 这条语句创建索引的值必须是唯一的。
    ```

  - 添加**普通索引**

    ```sql
    ALTER TABLE tbl_name ADD INDEX index_name (col_list);``// 添加普通索引，索引值可出现多次。
    ```

  - 添加**全文索引**

    ```sql
    ALTER TABLE tbl_name ADD FULLTEXT index_name (col_list);``// 该语句指定了索引为 FULLTEXT ，用于全文索引。
    ```

  - 添加外键约束

    ```sql
    创建外键语句结构：
    ALTER TABLE <表名>
    ADD CONSTRAINT FOREIGN KEY (<列名>)
    REFERENCES <关联表>（关联列）
    ```

  - 删除索引

    ```sql
    DROP INDEX index_name ON tbl_name;``// 或者``ALTER TABLE tbl_name DROP INDEX index_name；``ALTER TABLE tbl_name DROP PRIMARY KEY;
    ```

- **Use create**

  - 添加**普通索引**

    ```sql
    create index 索引名 on 表名(col1, col2, ..., )
    ```

  - 添加**唯一索引**

    ```{sql}
    create unique index 索引名 on 表名(col1, col2, ..., )
    ```

- **区别**

  - **Alter**可以省略索引名。如果省略索引名，数据库会默认**根据第一个索引列**赋予一个名称；**Create**必须指定索引名称。
  - **Create**不能用于创建**Primary key**索引；
  - **Alter**允许一条语句同时**创建多个索引**；**Create**一次只能**创建一个索引**

- **强制索引**

  - 使用强制索引查询

    ```sql
    SELECT * 
    FROM table_name 
    FORCE INDEX (index_list)
    WHERE condition; 
    ```

    在此语法中，将`FORCE INDEX`子句放在FROM子句之后，后跟查询优化器必须使用的命名索引列表。

  - 强制索引的优点

## Add column

```sql
ALTER TABLE <表名> ADD COLUMN <新字段名> <数据类型> [约束条件] [FIRST|AFTER 已存在的字段名];

# example
alter table actor 
add column create_date datetime not null default '2020-10-01 00:00:00' after last_update;
```

## 触发器

SQL41

````sql

drop trigger trigger_name
````



- 

## update

```sql
update table name 
set col1=val1, col2=val2,...
where condition;

update table name 
set col1 = replace(col1, val1, val2)
where id = 5(some condition)
```



## View

- defination

  在 SQL 中，视图是基于 SQL 语句的结果集的可视化的表。

  视图包含行和列，就像一个真实的表。视图中的字段就是来自一个或多个数据库中的真实的表中的字段。

  您可以向视图添加 SQL 函数、WHERE 以及 JOIN 语句，也可以呈现数据，就像这些数据来自于某个单一的表一样。

- usage

  ```{sql}
  # create view
  CREATE VIEW view_name AS
  SELECT column_name(s)
  FROM table_name
  WHERE condition;
  
  # update view
  CREATE OR REPLACE VIEW view_name AS
  SELECT column_name(s)
  FROM table_name
  WHERE condition
  
  # delete view
  DROP VIEW view_name
  ```

  > 视图总是显示最新的数据！每当用户查询视图时，数据库引擎通过使用视图的 SQL 语句重建数据。

## 数据倾斜

- **定义**：数据的分布不平衡，有些Reducer节点被分配的key值特别多，有些Reducer节点被分配的特别少，导致在处理数据的时候，有些很快就处理完了，而有些又迟迟未能处理完，所以其他的 Reducer 都执行完了，都在等这个量大的 key 值，导致整体任务最终迟迟无法完成。

- **表现**：常见表现就是map任务已执行完毕，个别reduce task，长时间卡在99%

- **原因**

  - 业务数据本身的特性。比如，某些业务数据作为key的字段本就很集中，含有大量重复值。
  - 建表时考虑不周。比如，日志表某些字段信息丢失，导致含大量空值，如又作为join关联条件，就会碰到数据倾斜的问题。
  - 某些SQL语句本身就有数据倾斜。比如计数时总喜欢用DISTINCT，COUNT(DISTINCT)，如果某些特殊值过多，则该值会被分入同一个reduce，则会非常耗时。

- **解决方法**

  - SQL 语句调整

    - 少用count distinct， 尽量采用sum group by的方式来替换 count(distinct) 完成计算
    - 保证关联条件数据类型一致
    - 大小表join，可以使用map join让小维度表先进内存，即在map端完成join，不经过reduce。
    - 内连接时，可以内层通过where条件筛选掉空值；或给空值分配随机的key值：可以对识别出的空值key变成一个字符串+随机数，这样就可以将原来集中的key分散开来，避免了数据倾斜的风险，且由于是无效数据无法关联上，因此不会出现在结果表中。
    - 左连接需要保留主表的无效数据，只需要将上述方法得到的结果再与驱动表的这些无数据取并集就可以
    - 有时，group by时维度过小的话，数据过于集中，数据自身倾斜，在SQL逻辑优化效果的不大情况下，有时候是可以将倾斜的key单独拿出来处理，即将大key和其他key分开处理，最后union回去。

  - 设置参数

    ````sql
    # 开启map端combiner聚合功能
    set hive.map.aggr=true
    
    # 开启负载均衡（万能膏药）
    hive.groupby.skewindata=true
    
    # 增加reducer任务数量（拉去数量分流）
    set mapred.reduce.tasks=20
    
    # 合并小文件
    set hive.merge.mapfiles=true
    ````

    

    