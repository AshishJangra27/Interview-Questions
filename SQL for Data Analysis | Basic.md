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
