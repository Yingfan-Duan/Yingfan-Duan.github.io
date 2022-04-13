---
layout:     post
title:      "SQL For Interview (Strings)"
subtitle:   "A collection of string operations"
date:       2022-4-12 15:00:00
author:     "Yingfan"
catalog: true
header-style: text
tags:
  - sql
  - study notes
  - interview
---

This blog will cover all common string operation functions in SQL.

# Functions

- **LOWER(), UPPER()**

  - 首字母大写：

    ```sql
    update table set field = concat(upper(left(field, 1)), substring(field, 2, (length(field)-1)))
    ```

- **TRIM**

  - **LTRIM()**

  - **RTRIM()**

  - **TRIM()**

    ```sql
    SELECT TRIM('  bar   ');
    # 'bar'
    
    SELECT TRIM(LEADING 'x' FROM 'xxxbarxxx');
    # 'barxxx'
    
    SELECT TRIM(BOTH 'x' FROM 'xxxbarxxx');
    SELECT TRIM('x' FROM 'xxxbarxxx');
    # 'bar'
    
    SELECT TRIM(TRAILING 'xyz' FROM 'barxxyz');
    # 'barx'
    ```

- **substring operation**
  - **left(str, length)**
  - **right(str, length)**
  - **substring(str, index, length)**
  - **substr(str, start, length)**
  - **mid(str, start, length)**

- **concat string**

  - **concat(str1, str2, ...)**

    - 如果任何参数是NULL，返回NULL
    - 一个数字参数被变换为等价的字符串形式

  - **concat_ws(separator, str1, str2, ...)**

    - 如果separator是NULL，返回NULL

  - **group_concat()**

    ```sql
    # syntax
    SELECT col1, col2, ..., colN
    GROUP_CONCAT ( [DISTINCT] col_name1 
    [ORDER BY clause]  [SEPARATOR str_val] ) 
    FROM table_name GROUP BY col_name2;
    ```

    - **Distinct:** It eliminates the repetition of values from result.
    - **Order By:** It sort the values of group in specific order and then concatenate them.
    - **Separator:** By default, the values of group are separated by (**,** ) operator. In order to change this separator value, Separator clause is used followed by a string literal. It is given as **Separator ‘str_value’**.

- **find and locate**

  - ELT() & FIELD()

    - **ELT(index, val1, val2, ...)**
    - 返回字符串列表中 第几个字符
    - 如果index超过范围或是0，返回NULL
    - **FIELD(value, val2, val2, ...)**
      - 查找字符串是否出现，出现在第几个位置 索引从1开始
      - 如果不存在查找的字符串，返回0
      - 如果查找的是NULL，返回0

  - **find_in_set(str, strlist)**

    - 如果字符串str在由N子串组成的表strlist之中，返回一个1到N的值
    - 如果str不是在strlist里面或如果strlist是空字符串，返回0
    - 如果任何一个参数是NULL，返回NULL
    - 如果第一个参数包含一个“,”，该函数将工作不正常

    ```sql
    SELECT id,name 
    FROM string_test 
    WHERE find_in_set('fly',hobby);
    ```

  - **position(substr in str)**

    - 返回子串substr在字符串str第一个出现的位置，
    - 如果substr不是在str里面，返回0.

  - **locate(substr, str, pos)**

    - 返回子串substr在字符串str第一个出现的位置，从位置pos开始。
    - 如果substr不是在str里面，返回0

  - **instr(str, substr)**

    - 返回子串substr在字符串str中的第一个出现的位置

- **modify string**

  - **repeat(str, count)**
    - 返回由重复countTimes次的字符串str组成的一个字符串
    - 如果count <= 0，返回一个空字符串
    - 如果str或count是NULL，返回NULL
  - **reverse(str)**
  - **insert(str, pos, len, newstr)**
    - 在位置pos起始的、长度为len的子串由字符串newstr代替
  - **space(num)**: 插入多个空格
  - **replace(str, find_string, replace_with)**

# Regular Expression

- **like**

  - The percent sign (%) represents zero, one, or multiple characters
  - The underscore sign (_) represents one, single character

- **regexp, rlike**

  - expr REGEXP pat
  - 如果符合，返回1，否则返回0
  - 如果expr或pat是NULL，返回NULL

- **regular expression syntax**

  | characters       | meaning                                  |
  | ---------------- | ---------------------------------------- |
  | ^a               | start with a                             |
  | $a               | end with a                               |
  | .                | any character                            |
  | a*               | a 0或多次                                |
  | a+               | a 1或多次                                |
  | a?               | a 0或1次                                 |
  | a\|b             | a或b                                     |
  | (abc)*           | abc 0或多次                              |
  | a{n}             | a n次                                    |
  | [a-dX], [\^a-dX] | a,b,c,d,X中的任意一个，^代表不是任意一个 |

  