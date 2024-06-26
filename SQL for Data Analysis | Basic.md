# 100 Essential SQL Interview Questions and Answers

## Easy Level Questions

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

