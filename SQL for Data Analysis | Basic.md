# 100 Essential SQL Interview Questions and Answers | Easy Level


### 1. What is SQL?
**Answer:**  
SQL (Structured Query Language) is a standard programming language used for managing and manipulating relational databases. It allows users to perform tasks such as querying data, updating records, deleting data, and creating and modifying database structures.

---

### 2. What is a primary key?
**Answer:**  
A primary key is a unique identifier for a record in a database table. It ensures that each record can be uniquely identified and helps to establish relationships between tables. A primary key cannot contain NULL values.

---

### 3. What is a foreign key?
**Answer:**  
A foreign key is a column or a set of columns in one table that uniquely identifies a row of another table. It creates a relationship between the two tables and enforces referential integrity.

---

### 4. What is a JOIN in SQL?
**Answer:**  
A JOIN clause is used to combine rows from two or more tables based on a related column between them. Common types of JOINs include:
- **INNER JOIN**
- **LEFT JOIN**
- **RIGHT JOIN**
- **FULL OUTER JOIN**

---

### 5. What is the difference between INNER JOIN and OUTER JOIN?
**Answer:**  
- **INNER JOIN:** Returns only the rows where there is a match in both tables involved in the join. If there are no matching rows, the result is null.
- **OUTER JOIN:** Includes all rows from one table and the matching rows from the other table. If there is no match, it returns null for the columns from the table without a match. This can be further divided into:
  - **LEFT JOIN:** Includes all rows from the left table.
  - **RIGHT JOIN:** Includes all rows from the right table.

---

### 6. What is a subquery?
**Answer:**  
A subquery is a query nested inside another query. It is used to perform intermediate steps in complex queries, often for filtering results or performing calculations that are then used by the main query.

---

### 7. How do you handle NULL values in SQL?
**Answer:**  
- **Using `COALESCE`:** Replace NULL values with a specified value.
- **Filtering out NULLs:** Use a condition in the WHERE clause to exclude rows with NULL values, e.g., `WHERE column IS NOT NULL`.

---

### 8. What is the difference between DELETE and TRUNCATE?
**Answer:**  
- **DELETE:** Removes specified rows from a table based on a condition and can be rolled back if used within a transaction. It also fires any associated triggers.
- **TRUNCATE:** Removes all rows from a table without logging individual row deletions and cannot be rolled back in most databases. TRUNCATE is faster than DELETE and resets any identity columns.

---

### 9. What is a stored procedure?
**Answer:**  
A stored procedure is a precompiled collection of one or more SQL statements stored in the database. It is used to encapsulate repetitive tasks, improve performance by reducing the need to compile SQL code multiple times and enhance security by controlling access to data.

---

### 10. What is indexing in SQL?
**Answer:**  
Indexing is a technique used to improve the performance of SQL queries by reducing the amount of data that needs to be scanned. An index is created on a table column, and it helps the database to find rows more quickly and efficiently.

---

### 11. What is normalization in SQL?
**Answer:**  
Normalization is the process of organizing data in a database to reduce redundancy and improve data integrity. It involves dividing large tables into smaller ones and defining relationships between them. Common normalization forms include:
- **1NF (First Normal Form)**
- **2NF (Second Normal Form)**
- **3NF (Third Normal Form)**
- **BCNF (Boyce-Codd Normal Form)**

---

### 12. What is denormalization?
**Answer:**  
Denormalization is the process of combining normalized tables to improve read performance. It involves adding redundant data to one or more tables to avoid complex joins and improve query execution time.

---

### 13. What is a view in SQL?
**Answer:**  
A view is a virtual table based on the result set of an SQL query. It contains rows and columns just like a real table and can be used to simplify complex queries, enhance security by restricting access to specific data, and present data in a specific format.

---

### 14. How do you create a view in SQL?
**Answer:**  
To create a view, use the `CREATE VIEW` statement followed by the view name and the `AS` keyword with the SELECT query. Example:
```sql
CREATE VIEW view_name AS
SELECT column1, column2
FROM table_name
WHERE condition;
```

---

### 15. What is a trigger in SQL?
**Answer:**  
A trigger is a stored procedure that automatically executes in response to certain events on a particular table or view. Triggers can be used for enforcing business rules, validating input data, and maintaining audit trails.

---

### 16. What is a transaction in SQL?
**Answer:**  
A transaction is a sequence of one or more SQL operations treated as a single unit of work. Transactions ensure data integrity by following the ACID properties (Atomicity, Consistency, Isolation, Durability). A transaction can be started with the `BEGIN TRANSACTION` statement and completed with `COMMIT` or `ROLLBACK`.

---

### 17. What is the difference between COMMIT and ROLLBACK?
**Answer:**  
- **COMMIT:** Saves all changes made in the transaction to the database permanently.
- **ROLLBACK:** Reverts all changes made in the transaction, restoring the database to its previous state before the transaction began.

---

### 18. What are aggregate functions in SQL?
**Answer:**  
Aggregate functions perform calculations on multiple rows of a table and return a single value. Common aggregate functions include:
- **SUM()**: Calculates the total sum of a numeric column.
- **AVG()**: Calculates the average value of a numeric column.
- **COUNT()**: Counts the number of rows.
- **MIN()**: Finds the minimum value in a column.
- **MAX()**: Finds the maximum value in a column.

---

### 19. What is the GROUP BY clause in SQL?
**Answer:**  
The `GROUP BY` clause groups rows that have the same values in specified columns into summary rows. It is often used with aggregate functions to perform calculations on each group. Example:
```sql
SELECT column1, COUNT(*)
FROM table_name
GROUP BY column1;
```

---

### 20. What is the HAVING clause in SQL?
**Answer:**  
The `HAVING` clause is used to filter groups of rows created by the `GROUP BY` clause based on a specified condition. Unlike the `WHERE` clause, which filters individual rows, `HAVING` filters groups. Example:
```sql
SELECT column1, COUNT(*)
FROM table_name
GROUP BY column1
HAVING COUNT(*) > 1;
```

--- 

### 21. What is the difference between WHERE and HAVING clause?
**Answer:**  
- **WHERE Clause:** Filters individual rows before any groupings are made. It cannot be used with aggregate functions.
- **HAVING Clause:** Filters groups of rows after the `GROUP BY` clause. It can be used with aggregate functions.

---

### 22. What is a UNION operator in SQL?
**Answer:**  
The `UNION` operator is used to combine the result sets of two or more SELECT queries. Each SELECT statement within the UNION must have the same number of columns in the result sets with similar data types. It removes duplicate rows between the various SELECT statements.

---

### 23. What is the difference between UNION and UNION ALL?
**Answer:**  
- **UNION:** Combines the result sets of two or more SELECT queries and removes duplicate rows.
- **UNION ALL:** Combines the result sets of two or more SELECT queries and includes duplicate rows.

---

### 24. What is the purpose of the DISTINCT keyword in SQL?
**Answer:**  
The `DISTINCT` keyword is used to remove duplicate rows from the result set of a SELECT query. It ensures that the returned results contain only unique values.

---

### 25. What are the different types of SQL commands?
**Answer:**  
SQL commands are divided into five categories:
- **DDL (Data Definition Language):** Includes commands like CREATE, ALTER, DROP.
- **DML (Data Manipulation Language):** Includes commands like SELECT, INSERT, UPDATE, DELETE.
- **DCL (Data Control Language):** Includes commands like GRANT, REVOKE.
- **TCL (Transaction Control Language):** Includes commands like COMMIT, ROLLBACK, SAVEPOINT.
- **DQL (Data Query Language):** Primarily includes the SELECT command.

---

### 26. What is a constraint in SQL?
**Answer:**  
A constraint is a rule applied to a column or a table to enforce data integrity and consistency. Common types of constraints include:
- **NOT NULL**
- **UNIQUE**
- **PRIMARY KEY**
- **FOREIGN KEY**
- **CHECK**
- **DEFAULT**

---

### 27. What is a schema in SQL?
**Answer:**  
A schema is a logical container for database objects such as tables, views, indexes, and procedures. It helps organize and manage database objects and can also provide a level of security by allowing different users to access specific schemas.

---

### 28. What is the difference between CHAR and VARCHAR data types?
**Answer:**  
- **CHAR:** A fixed-length character data type. It always uses the specified number of bytes, regardless of the length of the stored string.
- **VARCHAR:** A variable-length character data type. It uses only as many bytes as necessary to store the string, up to the specified maximum length.

---

### 29. What is a composite key?
**Answer:**  
A composite key is a combination of two or more columns in a table that together serve as a unique identifier for a record. Each column in the composite key can contain duplicate values, but the combination of all columns must be unique.

---

### 30. What is referential integrity?
**Answer:**  
Referential integrity is a property that ensures that relationships between tables remain consistent. When one table has a foreign key that points to another table, referential integrity ensures that the foreign key value matches a primary key value in the referenced table, or is NULL.

---

### 31. What is an alias in SQL?
**Answer:**  
An alias is a temporary name given to a table or column for the duration of a SQL query. It is used to make column names more readable and to simplify complex queries. Aliases are defined using the `AS` keyword.

---

### 32. What is a default constraint in SQL?
**Answer:**  
A default constraint is used to set a default value for a column when no value is specified during an insert operation. It ensures that the column always contains a value.

---

### 33. What is the difference between a clustered and a non-clustered index?
**Answer:**  
- **Clustered Index:** Sorts and stores the data rows in the table based on the index key. There can be only one clustered index per table.
- **Non-Clustered Index:** Creates a separate structure from the data rows. The index contains pointers to the data rows. There can be multiple non-clustered indexes per table.

---

### 34. What is a candidate key in SQL?
**Answer:**  
A candidate key is a column, or a set of columns, that can uniquely identify a row in a table. A table can have multiple candidate keys, but one of them will be chosen as the primary key.

---

### 35. What is a self-join?
**Answer:**  
A self-join is a join in which a table is joined with itself. It is useful for querying hierarchical data or comparing rows within the same table.

---

### 36. What is a sequence in SQL?
**Answer:**  
A sequence is a database object that generates a sequence of unique numeric values according to a specified increment. Sequences are commonly used to generate unique primary key values.

---

### 37. What is the purpose of the LIKE operator in SQL?
**Answer:**  
The `LIKE` operator is used in a `WHERE` clause to search for a specified pattern in a column. It is often used with wildcards:
- `%` represents zero, one, or multiple characters.
- `_` represents a single character.

---

### 38. What is a synonym in SQL?
**Answer:**  
A synonym is an alias or alternative name for a database object such as a table, view, sequence, or stored procedure. Synonyms make it easier to reference objects and can help with database schema management.

---

### 39. What is data integrity?
**Answer:**  
Data integrity refers to the accuracy, consistency, and reliability of data stored in a database. It is maintained through the use of constraints, triggers, and transactions to ensure that the data remains valid and consistent.

---

### 40. What is the ACID property in SQL?
**Answer:**  
The ACID properties ensure reliable processing of database transactions:
- **Atomicity:** Ensures that all operations within a transaction are completed successfully; otherwise, the transaction is aborted.
- **Consistency:** Ensures that a transaction brings the database from one valid state to another.
- **Isolation:** Ensures that the operations of one transaction are isolated from those of other transactions.
- **Durability:** Ensures that the results of a committed transaction are permanently saved in the database.

---

### 41. What is the difference between SQL and MySQL?
**Answer:**  
- **SQL:** A standard language used to manage and manipulate relational databases.
- **MySQL:** A relational database management system (RDBMS) that uses SQL to manage databases.

---

### 42. What is a cursor in SQL?
**Answer:**  
A cursor is a database object used to retrieve, manipulate, and navigate through a result set row by row. Cursors are useful for processing individual rows returned by a query.

---

### 43. What is an index in SQL?
**Answer:**  
An index is a database object that improves the speed of data retrieval operations on a table. It is created on one or more columns of a table and helps to locate rows more quickly and efficiently.

---

### 44. What is the purpose of the EXISTS clause in SQL?
**Answer:**  
The `EXISTS` clause is used to test for the existence of any record in a subquery. It returns TRUE if the subquery returns one or more records, otherwise FALSE.

---

### 45. What is the difference between DROP and DELETE commands in SQL?
**Answer:**  
- **DROP:** Removes a database object, such as a table or view, from the database entirely.
- **DELETE:** Removes rows from a table based on a specified condition but does not delete the table itself.

---

### 46. What is the difference between CHAR and VARCHAR data types in SQL?
**Answer:**  
- **CHAR:** A fixed-length character data type that always uses the specified number of bytes, regardless of the stored string's length.
- **VARCHAR:** A variable-length character data type that uses only as many bytes as necessary to store the string, up to the specified maximum length.

---

### 47. What is a constraint in SQL?
**Answer:**  
A constraint is a rule applied to a column or a table to enforce data integrity and consistency. Common types of constraints include NOT NULL, UNIQUE, PRIMARY KEY, FOREIGN KEY, CHECK, and DEFAULT.

---

### 48. What is the purpose of the BETWEEN operator in SQL?
**Answer:**  
The `BETWEEN` operator is used to filter the result set within a specified range of values. It is often used in the WHERE clause to filter rows based on a range of values for a particular column.

---

### 49. What is a stored procedure in SQL?
**Answer:**  
A stored procedure is a precompiled collection of one or more SQL statements stored in the database. It is used to encapsulate repetitive tasks, improve performance by reducing the need to compile SQL code multiple times, and enhance security by controlling access to data.

---

### 50. What is the purpose of the IN operator in SQL?
**Answer:**  
The `IN` operator is used to filter the result set to include only rows where a specified column's value matches any value in a list of specified values. It is often used in the WHERE clause.

---

### 51. What is the purpose of the ORDER BY clause in SQL?
**Answer:**  
The `ORDER BY` clause is used to sort the result set of a query by one or more columns. By default, it sorts in ascending order; however, it can also sort in descending order using the `DESC` keyword.

---

### 52. What is a unique key in SQL?
**Answer:**  
A unique key is a constraint that ensures all values in a column or a set of columns are unique across the table. Unlike the primary key, a table can have multiple unique keys, and unique keys can contain NULL values.

---

### 53. What is the purpose of the CASE statement in SQL?
**Answer:**  
The `CASE` statement is used to create conditional logic in SQL queries. It allows you to return different values based on specified conditions, similar to an if-else statement in programming languages.

---

### 54. What is the purpose of the LIMIT clause in SQL?
**Answer:**  
The `LIMIT` clause is used to restrict the number of rows returned by a query. It is often used in conjunction with the `ORDER BY` clause to return a subset of the result set.

---

### 55. What is a cross join in SQL?
**Answer:**  
A cross join, also known as a Cartesian join, returns the Cartesian product of the two tables involved in the join. This means it returns all possible combinations of rows from the two tables.

---

### 56. What is the difference between an alias and a synonym in SQL?
**Answer:**  
- **Alias:** A temporary name given to a table or column in a SQL query for the duration of that query.
- **Synonym:** A permanent alias for a database object, which remains until explicitly dropped.

---

### 57. What is the difference between SQL and PL/SQL?
**Answer:**  
- **SQL:** A standard language for managing and manipulating relational databases.
- **PL/SQL:** A procedural language extension for SQL, used in Oracle databases to write complex scripts and functions.

---

### 58. What is a foreign key constraint in SQL?
**Answer:**  
A foreign key constraint is a rule that maintains referential integrity between two tables. It ensures that the value in a foreign key column matches a value in the primary key column of the referenced table.

---

### 59. What is the purpose of the NVL function in SQL?
**Answer:**  
The `NVL` function is used to replace NULL values with a specified value. It is commonly used to ensure that calculations and operations are performed correctly even when some data is missing.

---

### 60. What is a subquery in SQL?
**Answer:**  
A subquery is a query nested inside another query. It is used to perform intermediate steps in complex queries, often for filtering results or performing calculations that are then used by the main query.

---

### 61. What is a correlated subquery in SQL?
**Answer:**  
A correlated subquery is a subquery that references columns from the outer query. It is executed once for each row processed by the outer query and is used to perform row-by-row processing.

---

### 62. What is the purpose of the COALESCE function in SQL?
**Answer:**  
The `COALESCE` function returns the first non-NULL value in a list of expressions. It is used to handle NULL values by providing a default value when NULLs are encountered.

---

### 63. What is a bitmap index in SQL?
**Answer:**  
A bitmap index is a type of index that uses bitmaps (binary arrays) to represent the existence of values in a table. Bitmap indexes are efficient for columns with a low cardinality (few distinct values) and are used to improve query performance.

---

### 64. What is the purpose of the ROW_NUMBER() function in SQL?
**Answer:**  
The `ROW_NUMBER()` function assigns a unique sequential integer to rows within a result set. It is often used for pagination or to provide a unique identifier for rows in a query result.

---

### 65. What is the difference between RANK() and DENSE_RANK() in SQL?
**Answer:**  
- **RANK():** Assigns a rank to each row within a partition, with gaps in the ranking sequence for ties.
- **DENSE_RANK():** Assigns a rank to each row within a partition without gaps in the ranking sequence for ties.

---

### 66. What is the purpose of the CAST() function in SQL?
**Answer:**  
The `CAST()` function is used to convert an expression from one data type to another. It is commonly used to ensure that data types are compatible in expressions and comparisons.

---

### 67. What is a materialized view in SQL?
**Answer:**  
A materialized view is a database object that stores the result of a query physically. It is used to improve query performance by precomputing and storing complex query results, reducing the need to recompute them each time.

---

### 68. What is the difference between INNER JOIN and CROSS JOIN?
**Answer:**  
- **INNER JOIN:** Returns only the rows where there is a match in both tables based on the specified condition.
- **CROSS JOIN:** Returns the Cartesian product of two tables, including all possible combinations of rows from both tables.

---

### 69. What is a recursive query in SQL?
**Answer:**  
A recursive query is a query that refers to itself, typically using a common table expression (CTE). It is used to perform hierarchical or tree-structured data retrieval.

---

### 70. What is the purpose of the GROUP_CONCAT() function in SQL?
**Answer:**  
The `GROUP_CONCAT()` function concatenates values from multiple rows into a single string, separated by a specified delimiter. It is used to aggregate and display multiple values from a group as a single string.

---

### 71. What is the purpose of the IFNULL function in SQL?
**Answer:**  
The `IFNULL` function returns a specified value if the given expression is NULL. If the expression is not NULL, it returns the expression itself. It is used to handle NULL values by providing a default value.

---

### 72. What is the difference between a primary key and a unique key?
**Answer:**  
- **Primary Key:** Uniquely identifies each row in a table and does not allow NULL values. A table can have only one primary key.
- **Unique Key:** Ensures that all values in a column or a set of columns are unique. It allows NULL values (only one per column) and a table can have multiple unique keys.

---

### 73. What is a lateral view in SQL?
**Answer:**  
A lateral view is used in conjunction with a table-generating function to enable the output of the function to be joined with the original table. It is often used in data transformation and analysis tasks.

---

### 74. What is the purpose of the CURRENT_DATE function in SQL?
**Answer:**  
The `CURRENT_DATE` function returns the current date of the system. It is used to retrieve the current date for use in queries and calculations.

---

### 75. What is a composite index in SQL?
**Answer:**  
A composite index is an index on two or more columns of a table. It is used to improve the performance of queries that filter or sort data based on multiple columns.

---

### 76. What is the purpose of the AUTO_INCREMENT attribute in SQL?
**Answer:**  
The `AUTO_INCREMENT` attribute is used to generate a unique identifier for new rows in a table automatically. It is often applied to primary key columns to ensure that each row has a unique identifier.

---

### 77. What is a table alias in SQL?
**Answer:**  
A table alias is a temporary name given to a table within a query. It is used to make the query more readable and to simplify references to the table, especially in complex queries involving joins.

---

### 78. What is the difference between COUNT(*) and COUNT(column_name)?
**Answer:**  
- **COUNT(*):** Counts all rows in the table, including rows with NULL values.
- **COUNT(column_name):** Counts non-NULL values in the specified column.

---

### 79. What is the purpose of the TO_DATE function in SQL?
**Answer:**  
The `TO_DATE` function converts a string to a date data type. It is used to transform string representations of dates into actual date values for use in queries and calculations.

---

### 80. What is the purpose of the CASE WHEN statement in SQL?
**Answer:**  
The `CASE WHEN` statement is used to implement conditional logic in SQL queries. It allows different values to be returned based on specified conditions, similar to an if-else statement in programming languages.

---

### 81. What is the purpose of the GETDATE() function in SQL?
**Answer:**  
The `GETDATE()` function returns the current date and time of the system. It is used to retrieve the current timestamp for use in queries and calculations.

---

### 82. What is a partitioned table in SQL?
**Answer:**  
A partitioned table is a large table that is divided into smaller, more manageable pieces called partitions. Each partition can be accessed and maintained separately, which can improve query performance and manageability.

---

### 83. What is the difference between TRUNCATE and DROP?
**Answer:**  
- **TRUNCATE:** Removes all rows from a table but retains the table structure for future use. It is faster than DELETE and cannot be rolled back in most databases.
- **DROP:** Deletes the table and its structure from the database entirely.

---

### 84. What is a cursor in SQL?
**Answer:**  
A cursor is a database object used to retrieve, manipulate, and navigate through a result set row by row. Cursors are useful for processing individual rows returned by a query.

---

### 85. What is an inline view in SQL?
**Answer:**  
An inline view is a subquery in the FROM clause of a SQL statement. It acts as a temporary table for the duration of the query and can simplify complex queries by breaking them down into smaller, more manageable parts.

---

### 86. What is the difference between IS NULL and IS NOT NULL?
**Answer:**  
- **IS NULL:** Checks whether a column value is NULL.
- **IS NOT NULL:** Checks whether a column value is not NULL.

---

### 87. What is the purpose of the CONCAT function in SQL?
**Answer:**  
The `CONCAT` function is used to concatenate two or more strings into a single string. It is commonly used to combine values from different columns or to add static text to query results.

---

### 88. What is a data warehouse in SQL?
**Answer:**  
A data warehouse is a centralized repository that stores large volumes of data from multiple sources. It is used for reporting, analysis, and decision-making processes. Data in a data warehouse is typically structured and optimized for read-heavy queries.

---

### 89. What is a surrogate key in SQL?
**Answer:**  
A surrogate key is a unique identifier for a record in a table that is not derived from application data. It is typically an auto-incremented number and is used as a primary key.

---

### 90. What is the purpose of the LEFT function in SQL?
**Answer:**  
The `LEFT` function is used to extract a specified number of characters from the left side of a string. It is commonly used to manipulate and format string data.

---

### 91. What is the purpose of the RIGHT function in SQL?
**Answer:**  
The `RIGHT` function is used to extract a specified number of characters from the right side of a string. It is commonly used to manipulate and format string data.

---

### 92. What is a star schema in SQL?
**Answer:**  
A star schema is a type of database schema that is commonly used in data warehousing and business intelligence. It consists of a central fact table surrounded by dimension tables. The fact table contains quantitative data, while dimension tables contain descriptive attributes.

---

### 93. What is a snowflake schema in SQL?
**Answer:**  
A snowflake schema is a type of database schema that is a more normalized form of the star schema. In a snowflake schema, dimension tables are normalized into multiple related tables, reducing redundancy and improving data integrity.

---

### 94. What is the difference between the EQUI JOIN and NON-EQUI JOIN?
**Answer:**  
- **EQUI JOIN:** A type of join that uses the equality operator (`=`) to match rows between tables based on a common column.
- **NON-EQUI JOIN:** A type of join that uses operators other than the equality operator (such as `<`, `>`, `<=`, `>=`, `<>`) to match rows between tables.

---

### 95. What is the purpose of the LENGTH function in SQL?
**Answer:**  
The `LENGTH` function returns the number of characters in a string. It is used to determine the length of string data.

---

### 96. What is a fact table in SQL?
**Answer:**  
A fact table is a central table in a star or snowflake schema in a data warehouse. It contains quantitative data (facts) and foreign keys to dimension tables. Fact tables are used to store metrics and measures for analysis and reporting.

---

### 97. What is a dimension table in SQL?
**Answer:**  
A dimension table is a table in a star or snowflake schema in a data warehouse. It contains descriptive attributes (dimensions) related to the facts in the fact table. Dimension tables are used to provide context and meaning to the facts for analysis and reporting.

---

### 98. What is the purpose of the REPLACE function in SQL?
**Answer:**  
The `REPLACE` function is used to replace occurrences of a specified substring within a string with another substring. It is commonly used to modify and clean string data.

---

### 99. What is the difference between the DATE and TIMESTAMP data types in SQL?
**Answer:**  
- **DATE:** Stores date values (year, month, day) without time.
- **TIMESTAMP:** Stores date and time values (year, month, day, hour, minute, second, fractional seconds).

---

### 100. What is a common table expression (CTE) in SQL?
**Answer:**  
A common table expression (CTE) is a temporary result set that can be referenced within a SELECT, INSERT, UPDATE, or DELETE statement. It is defined using the `WITH` keyword and provides a way to simplify complex queries and improve readability.
