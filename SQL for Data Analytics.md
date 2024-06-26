### 1. How do you calculate the moving average in SQL?
**Answer:**  
A moving average is used to smooth out short-term fluctuations and highlight longer-term trends or cycles in data. It can be calculated using the `AVG` function and the `OVER` clause with a `ROWS` or `RANGE` specification.
```sql
-- Example of calculating a 3-day moving average
SELECT date, value,
       AVG(value) OVER (ORDER BY date ROWS BETWEEN 2 PRECEDING AND CURRENT ROW) AS moving_avg
FROM sales;
```
---

### 2. How do you rank data within groups in SQL?
**Answer:**  
You can rank data within groups using window functions such as `RANK()`, `DENSE_RANK()`, and `ROW_NUMBER()`. These functions assign a rank to each row within the partition of a result set.
```sql
-- Example of ranking employees by salary within each department
SELECT name, department, salary,
       RANK() OVER (PARTITION BY department ORDER BY salary DESC) AS rank
FROM employees;
```
---

### 3. How do you handle duplicate rows in SQL?
**Answer:**  
To handle duplicate rows, you can use the `DISTINCT` keyword to select only unique rows. Alternatively, you can use `ROW_NUMBER()` with a common table expression (CTE) to identify and remove duplicates.
```sql
-- Example of removing duplicates using ROW_NUMBER()
WITH CTE AS (
    SELECT name, department, salary,
           ROW_NUMBER() OVER (PARTITION BY name, department ORDER BY salary DESC) AS row_num
    FROM employees
)
DELETE FROM CTE WHERE row_num > 1;
```
---

### 4. How do you pivot data in SQL?
**Answer:**  
Pivoting data means converting rows into columns. This can be achieved using the `PIVOT` function in databases that support it or using conditional aggregation with `CASE` statements.
```sql
-- Example of pivoting data using conditional aggregation
SELECT department,
       SUM(CASE WHEN gender = 'Male' THEN 1 ELSE 0 END) AS male_count,
       SUM(CASE WHEN gender = 'Female' THEN 1 ELSE 0 END) AS female_count
FROM employees
GROUP BY department;
```
---

### 5. How do you unpivot data in SQL?
**Answer:**  
Unpivoting data means converting columns into rows. This can be done using the `UNPIVOT` function in databases that support it or using a `UNION ALL` statement.
```sql
-- Example of unpivoting data using UNION ALL
SELECT department, 'Male' AS gender, male_count AS count
FROM departments
UNION ALL
SELECT department, 'Female' AS gender, female_count AS count
FROM departments;
```
---

### 6. How do you use the LEAD and LAG functions in SQL?
**Answer:**  
The `LEAD` and `LAG` functions are used to access data from subsequent or preceding rows in the result set, respectively. These functions are useful for calculating differences or comparisons between rows.
```sql
-- Example of using LEAD and LAG functions
SELECT date, value,
       LAG(value, 1) OVER (ORDER BY date) AS previous_value,
       LEAD(value, 1) OVER (ORDER BY date) AS next_value
FROM sales;
```
---

### 7. How do you handle time series data in SQL?
**Answer:**  
Handling time series data often involves calculating rolling averages, cumulative sums, and differences between time periods. Window functions like `SUM()`, `AVG()`, `ROW_NUMBER()`, and `LAG()` are commonly used.
```sql
-- Example of calculating a cumulative sum
SELECT date, value,
       SUM(value) OVER (ORDER BY date) AS cumulative_sum
FROM sales;
```
---

### 8. How do you perform a recursive query in SQL?
**Answer:**  
A recursive query is used to retrieve hierarchical data, such as organizational structures or bill-of-materials data. It is implemented using a common table expression (CTE) with the `WITH` keyword.
```sql
-- Example of a recursive query to find all subordinates of an employee
WITH RECURSIVE EmployeeHierarchy AS (
    SELECT id, name, manager_id
    FROM employees
    WHERE manager_id IS NULL
    UNION ALL
    SELECT e.id, e.name, e.manager_id
    FROM employees e
    INNER JOIN EmployeeHierarchy eh ON e.manager_id = eh.id
)
SELECT * FROM EmployeeHierarchy;
```
---

### 9. How do you optimize a SQL query for performance?
**Answer:**  
Query optimization involves several techniques, such as indexing, query rewriting, using appropriate joins, and avoiding unnecessary columns in SELECT statements. Analyzing query execution plans can also help identify bottlenecks.
```sql
-- Example of using an index to optimize a query
CREATE INDEX idx_employee_department ON employees(department);
SELECT name, department
FROM employees
WHERE department = 'HR';
```
---

### 10. How do you handle missing data in SQL?
**Answer:**  
Missing data can be handled using functions like `COALESCE` and `NVL` to replace NULL values with default values. Additionally, filtering out NULL values in queries can be done using the `IS NOT NULL` condition.
```sql
-- Example of handling missing data using COALESCE
SELECT name, COALESCE(salary, 0) AS salary
FROM employees;
```
---

### 11. How do you calculate the median in SQL?
**Answer:**  
Calculating the median involves finding the middle value in a sorted list of numbers. This can be done using window functions and the `PERCENTILE_CONT` function in databases that support it.
```sql
-- Example of calculating the median using PERCENTILE_CONT
SELECT PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY salary) AS median_salary
FROM employees;
```
---

### 12. How do you perform a self-join in SQL?
**Answer:**  
A self-join is a join in which a table is joined with itself. It is useful for querying hierarchical data or comparing rows within the same table.
```sql
-- Example of a self-join
SELECT e1.name AS employee_name, e2.name AS manager_name
FROM employees e1
JOIN employees e2 ON e1.manager_id = e2.id;
```
---

### 13. How do you create a stored procedure in SQL?
**Answer:**  
A stored procedure is a precompiled collection of one or more SQL statements stored in the database. It can be created using the `CREATE PROCEDURE` statement.
```sql
-- Example of creating a stored procedure
CREATE PROCEDURE GetEmployeeDetails
AS
BEGIN
    SELECT * FROM employees;
END;
```
---

### 14. How do you create a trigger in SQL?
**Answer:**  
A trigger is a stored procedure that automatically executes in response to certain events on a particular table or view. It can be created using the `CREATE TRIGGER` statement.
```sql
-- Example of creating a trigger
CREATE TRIGGER trgAfterInsert ON employees
FOR INSERT
AS
BEGIN
    PRINT 'New employee record inserted';
END;
```
---

### 15. How do you create a materialized view in SQL?
**Answer:**  
A materialized view is a database object that stores the result of a query physically. It can be created using the `CREATE MATERIALIZED VIEW` statement and is used to improve query performance by precomputing and storing complex query results.
```sql
-- Example of creating a materialized view
CREATE MATERIALIZED VIEW emp_dept_view AS
SELECT e.name, d.name AS department_name
FROM employees e
JOIN departments d ON e.department_id = d.id;
```
---

### 16. How do you refresh a materialized view in SQL?
**Answer:**  
A materialized view can be refreshed to reflect changes in the underlying base tables. This can be done using the `REFRESH MATERIALIZED VIEW` statement in databases that support it.
```sql
-- Example of refreshing a materialized view (PostgreSQL syntax)
REFRESH MATERIALIZED VIEW emp_dept_view;
```
---

### 17. How do you use the OVER() clause in SQL?
**Answer:**  
The `OVER()` clause is used with window functions to define the partitioning and ordering of rows. It allows you to perform calculations across a set of table rows related to the current row.
```sql
-- Example of using the OVER() clause with the ROW_NUMBER() function
SELECT name, department,
       ROW_NUMBER() OVER (PARTITION BY department ORDER BY salary DESC) AS row_num
FROM employees;
```
---

### 18. How do you calculate a cumulative sum in SQL?
**Answer:**  
A cumulative sum is calculated by adding each value in a column to the sum of all previous values in that column. This can be done using the `SUM()` function with the `OVER()` clause.
```sql
-- Example of calculating a cumulative sum
SELECT date, value,
       SUM(value) OVER (ORDER BY date) AS cumulative_sum
FROM sales;
```
---

### 19. How do you calculate a running total in SQL?
**Answer:**  
A running total is similar to a cumulative sum but is typically reset based on a partitioning column. This can be done using the `SUM()` function with the `OVER()` clause and `PARTITION BY`.
```sql
-- Example of calculating a running total
SELECT date, value,
       SUM(value) OVER (PARTITION BY category ORDER BY date) AS running_total
FROM sales;
```
---

### 20. How do you perform conditional aggregation in SQL?
**Answer:**  
Conditional aggregation involves using aggregate functions with conditional logic to calculate different results based on specified conditions. This can be done using `CASE` statements within aggregate functions.
```sql
-- Example of conditional aggregation
SELECT department,
       SUM(CASE WHEN gender = 'Male' THEN salary ELSE 0 END) AS male_salary,
       SUM(CASE WHEN gender = 'Female' THEN salary ELSE 0 END) AS female_salary
FROM employees
GROUP BY department;
```
---

### 21. How do you perform a full outer join in SQL?
**Answer:**  
A full outer join returns all rows when there is a match in either left or right table records. It returns NULL for non-matching rows from both sides.
```sql
-- Example of full outer join
SELECT e.name, d.name AS department_name
FROM employees e
FULL OUTER JOIN departments d ON e.department_id = d.id;
```
---

### 22. How do you handle NULL values in SQL?
**Answer:**  
NULL values can be handled using functions like `COALESCE` to replace NULLs with a default value, or by using conditional expressions like `CASE` to manage NULLs.
```sql
-- Example of handling NULL values using COALESCE
SELECT name, COALESCE(salary, 0) AS salary
FROM employees;
```
---

### 23. How do you remove duplicates in SQL?
**Answer:**  
To remove duplicates, you can use the `DISTINCT` keyword to select only unique rows. Additionally, using window functions like `ROW_NUMBER()` can help identify and remove duplicate rows.
```sql
-- Example of removing duplicates using DISTINCT
SELECT DISTINCT name, department
FROM employees;
```
---

### 24. How do you create a temporary table in SQL?
**Answer:**  
A temporary table is a table that exists temporarily and is usually used for storing intermediate results. It can be created using the `CREATE TEMPORARY TABLE` statement.
```sql
-- Example of creating a temporary table
CREATE TEMPORARY TABLE temp_employees AS
SELECT * FROM employees WHERE department = 'Sales';
```
---

### 25. How do you use the GROUP BY clause in SQL?
**Answer:**  
The `GROUP BY` clause groups rows that have the same values in specified columns into summary rows. It is often used with aggregate functions to perform calculations on each group.
```sql
-- Example of using GROUP BY
SELECT department, AVG(salary) AS avg_salary
FROM employees
GROUP BY department;
```
---

### 26. How do you use the HAVING clause in SQL?
**Answer:**  
The `HAVING` clause is used to filter groups of rows created by the `GROUP BY` clause based on a specified condition. Unlike the `WHERE` clause, which filters individual rows, `HAVING` filters groups.
```sql
-- Example of using HAVING
SELECT department, AVG(salary) AS avg_salary
FROM employees
GROUP BY department
HAVING AVG(salary) > 50000;
```
---

### 27. How do you use window functions in SQL?
**Answer:**  
Window functions perform calculations across a set of table rows that are related to the current row. They are used with the `OVER()` clause to define the partitioning and ordering of rows.
```sql
-- Example of using window functions
SELECT name, department, salary,
       AVG(salary) OVER (PARTITION BY department) AS avg_department_salary
FROM employees;
```
---

### 28. How do you use the CASE statement in SQL?
**Answer:**  
The `CASE` statement is used to create conditional logic in SQL queries. It allows you to return different values based on specified conditions, similar to an if-else statement in programming languages.
```sql
-- Example of using CASE statement
SELECT name, salary,
       CASE
           WHEN salary > 50000 THEN 'High'
           WHEN salary BETWEEN 30000 AND 50000 THEN 'Medium'
           ELSE 'Low'
       END AS salary_level
FROM employees;
```
---

### 29. How do you calculate the difference between two dates in SQL?
**Answer:**  
The difference between two dates can be calculated using the `DATEDIFF` function or by subtracting one date from another, depending on the SQL dialect.
```sql
-- Example of calculating date difference (SQL Server syntax)
SELECT DATEDIFF(day, '2023-01-01', '2023-12-31') AS days_diff;

-- Example of calculating date difference (PostgreSQL syntax)
SELECT '2023-12-31'::date - '2023-01-01'::date AS days_diff;
```
---

### 30. How do you use the UNION operator in SQL?
**Answer:**  
The `UNION` operator is used to combine the result sets of two or more SELECT queries. Each SELECT statement within the UNION must have the same number of columns in the result sets with similar data types. It removes duplicate rows between the various SELECT statements.
```sql
-- Example of UNION
SELECT name FROM employees
UNION
SELECT name FROM managers;
```
---

### 31. How do you use the UNION ALL operator in SQL?
**Answer:**  
The `UNION ALL` operator is used to combine the result sets of two or more SELECT queries, including duplicate rows. It returns all rows from the combined SELECT statements.
```sql
-- Example of UNION ALL
SELECT name FROM employees
UNION ALL
SELECT name FROM managers;
```
---

### 32. How do you use the INTERSECT operator in SQL?
**Answer:**  
The `INTERSECT` operator is used to return the common rows from two or more SELECT queries. It returns only the rows that appear in both result sets.
```sql
-- Example of INTERSECT
SELECT name FROM employees
INTERSECT
SELECT name FROM managers;
```
---

### 33. How do you use the EXCEPT operator in SQL?
**Answer:**  
The `EXCEPT` operator is used to return the rows from the first SELECT query that are not present in the second SELECT query. It effectively performs a set difference operation.
```sql
-- Example of EXCEPT
SELECT name FROM employees
EXCEPT
SELECT name FROM managers;
```
---

### 34. How do you use the EXISTS operator in SQL?
**Answer:**  
The `EXISTS` operator is used to test for the existence of any rows in a subquery. It returns TRUE if the subquery returns one or more rows, otherwise FALSE.
```sql
-- Example of EXISTS
SELECT name
FROM employees
WHERE EXISTS (
    SELECT 1
    FROM managers
    WHERE managers.name = employees.name
);
```
---

### 35. How do you use the IN operator in SQL?
**Answer:**  
The `IN` operator is used to filter rows based on a specified list of values. It returns TRUE if the column value matches any value in the list.
```sql
-- Example of IN
SELECT name, department
FROM employees
WHERE department IN ('HR', 'Sales', 'IT');
```
---

### 36. How do you use the BETWEEN operator in SQL?
**Answer:**  
The `BETWEEN` operator is used to filter rows based on a range of values. It returns TRUE if the column value is between the specified start and end values.
```sql
-- Example of BETWEEN
SELECT name, salary
FROM employees
WHERE salary BETWEEN 30000 AND 50000;
```
---

### 37. How do you use the LIKE operator in SQL?
**Answer:**  
The `LIKE` operator is used to search for a specified pattern in a column. It is often used with wildcards:
- `%` represents zero, one, or multiple characters.
- `_` represents a single character.
```sql
-- Example of LIKE operator
SELECT name
FROM employees
WHERE name LIKE 'J%';
```
---

### 38. How do you use the CONCAT function in SQL?
**Answer:**  
The `CONCAT` function is used to concatenate two or more strings into a single string. It is commonly used to combine values from different columns or to add static text to query results.
```sql
-- Example of CONCAT function
SELECT CONCAT(first_name, ' ', last_name) AS full_name
FROM employees;
```
---

### 39. How do you create an index in SQL?
**Answer:**  
An index is created on a table column to improve the performance of queries. It can be created using the `CREATE INDEX` statement.
```sql
-- Example of creating an index
CREATE INDEX idx_employee_name ON employees(name);
```
---

### 40. How do you create a unique index in SQL?
**Answer:**  
A unique index ensures that all values in the indexed column are unique. It can be created using the `CREATE UNIQUE INDEX` statement.
```sql
-- Example of creating a unique index
CREATE UNIQUE INDEX idx_employee_email ON employees(email);
```
---

### 41. How do you create a composite index in SQL?
**Answer:**  
A composite index is an index on two or more columns of a table. It is used to improve the performance of queries that filter or sort data based on multiple columns.
```sql
-- Example of creating a composite index
CREATE INDEX idx_name_department ON employees(name, department);
```
---

### 42. How do you create a full-text index in SQL?
**Answer:**  
A full-text index is used to improve the performance of full-text searches on large text fields. It can be created using the `CREATE FULLTEXT INDEX` statement in databases that support it.
```sql
-- Example of creating a full-text index (MySQL syntax)
CREATE FULLTEXT INDEX idx_employee_resume ON employees(resume);
```
---

### 43. How do you drop an index in SQL?
**Answer:**  
An index can be dropped using the `DROP INDEX` statement. This removes the index from the table and can improve performance if the index is no longer needed.
```sql
-- Example of dropping an index
DROP INDEX idx_employee_name;
```
---

### 44. How do you use the TRANSLATE function in SQL?
**Answer:**  
The `TRANSLATE` function replaces each occurrence of a character in a string with another specified character. It is commonly used to perform character replacements and data cleaning.
```sql
-- Example of TRANSLATE function
SELECT TRANSLATE(name, 'aeiou', '12345') AS translated_name
FROM employees;
```
---

### 45. How do you use the SUBSTRING function in SQL?
**Answer:**  
The `SUBSTRING` function is used to extract a part of a string based on specified starting position and length. It is commonly used to manipulate and format string data.
```sql
-- Example of SUBSTRING function
SELECT SUBSTRING(name, 1, 3) AS name_prefix
FROM employees;
```
---

### 46. How do you use the REPLACE function in SQL?
**Answer:**  
The `REPLACE` function is used to replace occurrences of a specified substring within a string with another substring. It is commonly used to modify and clean string data.
```sql
-- Example of REPLACE function
SELECT REPLACE(name, 'John', 'Jonathan') AS updated_name
FROM employees;
```
---

### 47. How do you create a view in SQL?
**Answer:**  
A view is a virtual table based on the result-set of an SQL query. It contains rows and columns just like a real table and can be used to simplify complex queries, enhance security by restricting access to specific data, and present data in a specific format.
```sql
-- Example of creating a view
CREATE VIEW employee_departments AS
SELECT employees.name, departments.name AS department_name
FROM employees
INNER JOIN departments ON employees.department_id = departments.id;
```
---

### 48. How do you update data in a view in SQL?
**Answer:**  
To update data in a view, the view must be updatable, meaning it should be based on a single table or a simple join of tables, and should not contain aggregate functions or GROUP BY clauses. Updates can be performed using the `UPDATE` statement.
```sql
-- Example of updating data in a view
UPDATE employee_departments
SET department_name = 'Marketing'
WHERE name = 'John Doe';
```
---

### 49. How do you delete data from a view in SQL?
**Answer:**  
To delete data from a view, the view must be updatable. Deletions can be performed using the `DELETE` statement.
```sql
-- Example of deleting data from a view
DELETE FROM employee_departments
WHERE name = 'John Doe';
```
---

### 50. How do you create a sequence in SQL?
**Answer:**  
A sequence is a database object that generates a sequence of unique numeric values according to a specified increment. Sequences are commonly used to generate unique primary key values.
```sql
-- Example of creating a sequence (Oracle syntax)
CREATE SEQUENCE emp_sequence
START WITH 1
INCREMENT BY 1;

-- Using the sequence to insert data
INSERT INTO employees (id, name, department)
VALUES (emp_sequence.NEXTVAL, 'John Doe', 'HR');
```
---
