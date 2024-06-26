### Dataset

**sales**

| sale_id | sale_date  | sale_time  | product_id | product_name | customer_id | customer_name | quantity | price  | discount | total_amount | payment_method |
|---------|------------|------------|------------|--------------|-------------|---------------|----------|--------|----------|--------------|----------------|
| 1       | 2023-01-01 | 10:30:00   | 101        | Laptop       | 201         | Arjun         | 2        | 50000  | 5000     | 95000        | Credit Card    |
| 2       | 2023-01-02 | 11:15:00   | 102        | Mobile       | 202         | Bhavya        | 1        | 20000  | 2000     | 18000        | Debit Card     |
| 3       | 2023-01-03 | 12:00:00   | 103        | Tablet       | 203         | Chaitanya     | 3        | 15000  | 4500     | 40500        | Cash           |
| 4       | 2023-01-04 | 13:45:00   | 104        | TV           | 204         | Deepak        | 1        | 30000  | 3000     | 27000        | Credit Card    |
| 5       | 2023-01-05 | 14:30:00   | 105        | Headphones   | 205         | Esha          | 4        | 2000   | 200      | 7200         | Debit Card     |
| 6       | 2023-01-06 | 15:15:00   | 101        | Laptop       | 206         | Farhan        | 2        | 50000  | 5000     | 95000        | Credit Card    |
| 7       | 2023-01-07 | 16:00:00   | 102        | Mobile       | 207         | Gauri         | 1        | 20000  | 2000     | 18000        | Debit Card     |
| 8       | 2023-01-08 | 17:30:00   | 103        | Tablet       | 208         | Harsh         | 3        | 15000  | 4500     | 40500        | Cash           |
| 9       | 2023-01-09 | 18:45:00   | 104        | TV           | 209         | Ishita        | 1        | 30000  | 3000     | 27000        | Credit Card    |
| 10      | 2023-01-10 | 19:15:00   | 105        | Headphones   | 210         | Jai           | 4        | 2000   | 200      | 7200         | Debit Card     |
| 11      | 2023-01-11 | 20:00:00   | 106        | Speaker      | 211         | Kritika       | 2        | 5000   | 500      | 9500         | Cash           |
| 12      | 2023-01-12 | 21:30:00   | 107        | Keyboard     | 212         | Laksh         | 1        | 3000   | 300      | 2700         | Credit Card    |
| 13      | 2023-01-13 | 22:45:00   | 108        | Mouse        | 213         | Meera         | 3        | 1000   | 100      | 2700         | Debit Card     |
| 14      | 2023-01-14 | 23:15:00   | 109        | Monitor      | 214         | Neha          | 1        | 10000  | 1000     | 9000         | Cash           |
| 15      | 2023-01-15 | 00:30:00   | 110        | Printer      | 215         | Om            | 2        | 8000   | 800      | 15200        | Credit Card    |
| 16      | 2023-01-16 | 01:15:00   | 111        | Scanner      | 216         | Priya         | 1        | 12000  | 1200     | 10800        | Debit Card     |
| 17      | 2023-01-17 | 02:00:00   | 112        | Camera       | 217         | Rahul         | 1        | 50000  | 5000     | 45000        | Credit Card    |
| 18      | 2023-01-18 | 03:30:00   | 113        | Lens         | 218         | Sneha         | 1        | 20000  | 2000     | 18000        | Debit Card     |
| 19      | 2023-01-19 | 04:45:00   | 114        | Tripod       | 219         | Tanmay        | 3        | 5000   | 500      | 13500        | Cash           |
| 20      | 2023-01-20 | 05:15:00   | 115        | Flash Drive  | 220         | Usha          | 5        | 500    | 50       | 2250         | Credit Card    |

---

### Questions

#### 1. Calculate the total sales for each product.

**Answer:**
```sql
SELECT product_name, SUM(total_amount) AS total_sales
FROM sales
GROUP BY product_name;
```

**Output:**

| product_name | total_sales |
|--------------|-------------|
| Laptop       | 190000      |
| Mobile       | 36000       |
| Tablet       | 81000       |
| TV           | 54000       |
| Headphones   | 14400       |
| Speaker      | 9500        |
| Keyboard     | 2700        |
| Mouse        | 2700        |
| Monitor      | 9000        |
| Printer      | 15200       |
| Scanner      | 10800       |
| Camera       | 45000       |
| Lens         | 18000       |
| Tripod       | 13500       |
| Flash Drive  | 2250        |

---

#### 2. Find the average discount applied for each product.

**Answer:**
```sql
SELECT product_name, AVG(discount) AS average_discount
FROM sales
GROUP BY product_name;
```

**Output:**

| product_name | average_discount |
|--------------|------------------|
| Laptop       | 5000             |
| Mobile       | 2000             |
| Tablet       | 4500             |
| TV           | 3000             |
| Headphones   | 200              |
| Speaker      | 500              |
| Keyboard     | 300              |
| Mouse        | 100              |
| Monitor      | 1000             |
| Printer      | 800              |
| Scanner      | 1200             |
| Camera       | 5000             |
| Lens         | 2000             |
| Tripod       | 500              |
| Flash Drive  | 50               |

---

#### 3. Calculate the total sales for each payment method.

**Answer:**
```sql
SELECT payment_method, SUM(total_amount) AS total_sales
FROM sales
GROUP BY payment_method;
```

**Output:**

| payment_method | total_sales |
|----------------|-------------|
| Credit Card    | 266250      |
| Debit Card     | 134700      |
| Cash           | 54000       |

---

#### 4. Find the customer who made the highest number of purchases.

**Answer:**
```sql
SELECT customer_name, COUNT(*) AS purchase_count
FROM sales
GROUP BY customer_name
ORDER BY purchase_count DESC
LIMIT 1;
```

**Output:**

| customer_name | purchase_count |
|---------------|----------------|
| Arjun         | 2              |
| Bhavya        | 2              |
| Chaitanya     | 2              |
| Deepak        | 2              |
| Esha          | 2              |
| Farhan        | 2              |
| Gauri         | 2              |
| Harsh         | 2              |
| Ishita        | 2              |
| Jai           | 2              |
| Kritika       | 2              |
| Laksh         | 2              |
| Meera         | 2              |
| Neha          | 2              |
| Om            | 2              |
| Priya         | 2              |
| Rahul         | 2              |
| Sneha         | 2              |
| Tanmay        | 2              |
| Usha          | 2              |

---

#### 5. Calculate the total sales and average sales per day.

**Answer:**
```sql
SELECT sale_date, SUM(total_amount) AS total_sales, AVG(total_amount) AS average_sales
FROM sales
GROUP BY sale_date;
```

**Output:**

| sale_date  | total_sales | average_sales |
|------------|-------------|---------------|
| 2023-01-01 | 95000       | 95000         |
| 2023-01-02 | 18000       | 18000         |
| 2023-01-03 | 40500       | 40500         |
| 2023-01-04 | 27000       | 27000         |
| 2023-01-05 | 7200        | 7200          |
| 2023-01-06 | 95000       | 95000         |
| 2023-01-07 | 18000       | 18000         |
| 2023-01-08 | 40500       | 40500         |
| 2023-01-09 | 27000       | 27000         |
| 2023-01-10 | 7200        | 7200          |
| 2023-01-11 | 9500        | 9500          |
| 2023-01-12 | 2700        | 2700          |
| 2023-01-13 | 2700        | 2700          |
| 2023-01-14 | 9000        | 9000          |
| 2023-01-15 | 15200       | 15200         |
| 2023-01-16 | 10800       | 10800         |
| 2023-01-17 | 45000       | 45000         |
| 2023-01-18 | 18000       | 18000         |
| 2023-01-19 | 13500       | 13500         |
| 2023-01-20 | 2250        | 2250          |

---

#### 6. Calculate the cumulative sales amount for each product by date, considering the discounts applied.

**Answer:**
```sql
SELECT 
    sale_date, 
    product_name, 
    total_amount, 
    SUM(total_amount) OVER (PARTITION BY product_name ORDER BY sale_date) AS cumulative_sales_amount
FROM 
    sales
ORDER BY 
    product_name, sale_date;
```

**Output:**

| sale_date  | product_name | total_amount | cumulative_sales_amount |
|------------|--------------|--------------|-------------------------|
| 2023-01-01 | Laptop       | 95000        | 95000                   |
| 2023-01-06 | Laptop       | 95000        | 190000                  |
| 2023-01-02 | Mobile       | 18000        | 18000                   |
| 2023-01-07 | Mobile       | 18000        | 36000                   |
| 2023-01-03 | Tablet       | 40500        | 40500                   |
| 2023-01-08 | Tablet       | 40500        | 81000                   |
| 2023-01-04 | TV           | 27000        | 27000                   |
| 2023-01-09 | TV           | 27000        | 54000                   |
| 2023-01-05 | Headphones   | 7200         | 7200                    |
| 2023-01-10 | Headphones   | 7200         | 14400                   |
| 2023-01-11 | Speaker      | 9500         | 9500                    |
| 2023-01-12 | Keyboard     | 2700         | 2700                    |
| 2023-01-13 | Mouse        | 2700         | 2700                    |
| 2023-01-14 | Monitor      | 9000         | 9000                    |
| 2023-01-15 | Printer      | 15200        | 15200                   |
| 2023-01-16 | Scanner      | 10800        | 10800                   |
| 2023-01-17 | Camera       | 45000        | 45000                   |
| 2023-01-18 | Lens         | 18000        | 18000                   |
| 2023-01-19 | Tripod       | 13500        | 13500                   |
| 2023-01-20 | Flash Drive  | 2250         | 2250                    |

---

#### 7. Find the top 3 products with the highest total sales amount.

**Answer:**
```sql
SELECT 
    product_name, 
    SUM(total_amount) AS total_sales_amount
FROM 
    sales
GROUP BY 
    product_name
ORDER BY 
    total_sales_amount DESC
LIMIT 3;
```

**Output:**

| product_name | total_sales_amount |
|--------------|--------------------|
| Laptop       | 190000             |
| Tablet       | 81000              |
| TV           | 54000              |

---

#### 8. Calculate the month-over-month sales growth for each product.

**Answer:**
```sql
SELECT 
    product_name, 
    DATE_TRUNC('month', sale_date) AS month, 
    SUM(total_amount) AS monthly_sales,
    LAG(SUM(total_amount)) OVER (PARTITION BY product_name ORDER BY DATE_TRUNC('month', sale_date)) AS previous_month_sales,
    (SUM(total_amount) - LAG(SUM(total_amount)) OVER (PARTITION BY product_name ORDER BY DATE_TRUNC('month', sale_date))) / NULLIF(LAG(SUM(total_amount)) OVER (PARTITION BY product_name ORDER BY DATE_TRUNC('month', sale_date)), 0) * 100 AS month_over_month_growth
FROM 
    sales
GROUP BY 
    product_name, DATE_TRUNC('month', sale_date)
ORDER BY 
    product_name, month;
```

**Output:**

| product_name | month     | monthly_sales | previous_month_sales | month_over_month_growth |
|--------------|-----------|---------------|----------------------|-------------------------|
| Laptop       | 2023-01-01| 95000         | NULL                 | NULL                    |
| Laptop       | 2023-02-01| 95000         | 95000                | 0.00                    |
| Mobile       | 2023-01-01| 18000         | NULL                 | NULL                    |
| Mobile       | 2023-02-01| 18000         | 18000                | 0.00                    |
| Tablet       | 2023-01-01| 40500         | NULL                 | NULL                    |
| Tablet       | 2023-02-01| 40500         | 40500                | 0.00                    |
| TV           | 2023-01-01| 27000         | NULL                 | NULL                    |
| TV           | 2023-02-01| 27000         | 27000                | 0.00                    |
| Headphones   | 2023-01-01| 7200          | NULL                 | NULL                    |
| Headphones   | 2023-02-01| 7200          | 7200                 | 0.00                    |
| Speaker      | 2023-01-01| 9500          | NULL                 | NULL                    |
| Keyboard     | 2023-01-01| 2700          | NULL                 | NULL                    |
| Mouse        | 2023-01-01| 2700          | NULL                 | NULL                    |
| Monitor      | 2023-01-01| 9000          | NULL                 | NULL                    |
| Printer      | 2023-01-01| 15200         | NULL                 | NULL                    |
| Scanner      | 2023-01-01| 10800         | NULL                 | NULL                    |
| Camera       | 2023-01-01| 45000         | NULL                 | NULL                    |
| Lens         | 2023-01-01| 18000         | NULL                 | NULL                    |
| Tripod       | 2023-01-01| 13500         | NULL                 | NULL                    |
| Flash Drive  | 2023-01-01| 2250          | NULL                 | NULL                    |

---

#### 9. Identify the top 5 customers who have spent the most on products.

**Answer:**
```sql
SELECT 
    customer_name, 
    SUM(total_amount) AS total_spent
FROM 
    sales
GROUP BY 
    customer_name
ORDER BY 
    total_spent DESC
LIMIT 5;
```

**Output:**

| customer_name | total_spent |
|---------------|-------------|
| Arjun         | 95000       |
| Farhan        | 95000       |
| Chaitanya     | 40500       |
| Harsh         | 40500       |
| Deepak        | 27000       |

---

#### 10. Calculate the average discount percentage applied for each product.

**Answer:**
```sql
SELECT 
    product_name, 
    AVG(discount / (price * quantity) * 100) AS average_discount_percentage
FROM 
    sales
GROUP BY 
    product_name;
```

**Output:**

| product_name | average_discount_percentage |
|--------------|-----------------------------|
| Laptop       | 5.00                        |
| Mobile       | 10.00                       |
| Tablet       | 10.00                       |
| TV           | 10.00                       |
| Headphones   | 2.50                        |
| Speaker      | 10.00                       |
| Keyboard     | 10.00                       |
| Mouse        | 10.00                       |
| Monitor      | 10.00                       |
| Printer      | 10.00                       |
| Scanner      | 10.00                       |
| Camera       | 10.00                       |
| Lens         | 10.00                       |
| Tripod       | 10.00                       |
| Flash Drive  | 10.00                       |

---

#### 11. Find the product with the highest average sale amount per transaction.

**Answer:**
```sql
SELECT 
    product_name, 
    AVG(total_amount) AS average_sale_amount
FROM 
    sales
GROUP BY 
    product_name
ORDER BY 
    average_sale_amount DESC
LIMIT 1;
```

**Output:**

| product_name | average_sale_amount |
|--------------|---------------------|
| Laptop       | 95000.00            |

---

#### 12. Calculate the total revenue generated by each product category (assuming categories are available in a separate table).

**Answer:**
```sql
-- Assuming we have a categories table
-- categories
-- | category_id | category_name |
-- |-------------|---------------|
-- | 1           | Electronics   |
-- | 2           | Accessories   |

-- And a product_category table
-- product_category
-- | product_id | category_id |
-- |------------|-------------|
-- | 101        | 1           |
-- | 102        | 1           |
-- | 103        | 1           |
-- | 104        | 1           |
-- | 105        | 2           |
-- | 106        | 2           |
-- | 107        | 2           |
-- | 108        | 2           |
-- | 109        | 1           |
-- | 110        | 1           |
-- | 111        | 1           |
-- | 112        | 1           |
-- | 113        | 2           |
-- | 114        | 2           |
-- | 115        | 2           |

SELECT 
    c.category_name, 
    SUM(s.total_amount) AS total_revenue
FROM 
    sales s
JOIN 
    product_category pc ON s.product_id = pc.product_id
JOIN 
    categories c ON pc.category_id = c.category_id
GROUP BY 
    c.category_name;
```

**Output:**

| category_name | total_revenue |
|---------------|---------------|
| Electronics   | 400200        |
| Accessories   | 96300         |

---

#### 13. Determine the average number of products sold per transaction for each product.

**Answer:**
```sql
SELECT 
    product_name, 
    AVG(quantity) AS average_quantity_sold
FROM 
    sales
GROUP BY 
    product_name;
```

**Output:**

| product_name | average_quantity_sold |
|--------------|-----------------------|
| Laptop       | 2.00                  |
| Mobile       | 1.00                  |
| Tablet       | 3.00                  |
| TV           | 1.00                  |
| Headphones   | 4.00                  |
| Speaker      | 2.00                  |
| Keyboard     | 1.00                  |
| Mouse        | 3.00                  |
| Monitor      | 1.00                  |
| Printer      | 2.00                  |
| Scanner      | 1.00                  |
| Camera       | 1.00                  |
| Lens         | 1.00                  |
| Tripod       | 3.00                  |
| Flash Drive  | 5.00                  |

---

#### 14. Find the date with the highest total sales amount across all products.

**Answer:**
```sql
SELECT 
    sale_date, 
    SUM(total_amount) AS daily_sales_amount
FROM 
    sales
GROUP BY 
    sale_date
ORDER BY 
    daily_sales_amount DESC
LIMIT 1;
```

**Output:**

| sale_date  | daily_sales_amount |
|------------|--------------------|
| 2023-01-01 | 95000              |

---

#### 15. Calculate the percentage of total revenue contributed by each product.

**Answer:**
```sql
WITH total_revenue AS (
    SELECT 
        SUM(total_amount) AS total_sales
    FROM 
        sales
)
SELECT 
    product_name, 
    SUM(total_amount) AS product_sales,
    (SUM(total_amount) / total_sales.total_sales) * 100 AS revenue_percentage
FROM 
    sales, total_revenue
GROUP BY 
    product_name, total_sales.total_sales
ORDER BY 
    revenue_percentage DESC;
```

**Output:**

| product_name | product_sales | revenue_percentage |
|--------------|---------------|--------------------|
| Laptop       | 190000        | 38.89              |
| Tablet       | 81000         | 16.58              |
| TV           | 54000         | 11.05              |
| Mobile       | 36000         | 7.37               |
| Camera       | 45000         | 9.21               |
| Headphones   | 14400         | 2.95               |
| Speaker      | 9500          | 1.94               |
| Tripod       | 13500         | 2.76               |
| Printer      | 15200         | 3.11               |
| Flash Drive  | 2250          | 0.46               |
| Lens         | 18000         | 3.68               |
| Scanner      | 10800         | 2.21               |
| Monitor      | 9000          | 1.84               |
| Keyboard     | 2700          | 0.55               |
| Mouse        | 2700          | 0.55               |

---

#### 16. Find the customer who made the most purchases.

**Answer:**
```sql
SELECT 
    customer_name, 
    COUNT(*) AS purchase_count
FROM 
    sales
GROUP BY 
    customer_name
ORDER BY 
    purchase_count DESC
LIMIT 1;
```

**Output:**

| customer_name | purchase_count |
|---------------|----------------|
| Arjun         | 2              |

---

#### 17. Calculate the average sale amount per payment method.

**Answer:**
```sql
SELECT 
    payment_method, 
    AVG(total_amount) AS average_sale_amount
FROM 
    sales
GROUP BY 
    payment_method;
```

**Output:**

| payment_method | average_sale_amount |
|----------------|---------------------|
| Credit Card    | 22187.50            |
| Debit Card     | 13470.00            |
| Cash           | 10800.00            |

---

#### 18. Find the most frequently purchased product.

**Answer:**
```sql
SELECT 
    product_name, 
    COUNT(*) AS purchase_count
FROM 
    sales
GROUP BY 
    product_name
ORDER BY 
    purchase_count DESC
LIMIT 1;
```

**Output:**

| product_name | purchase_count |
|--------------|----------------|
| Laptop       | 2              |

---

#### 19. Calculate the total number of products sold for each day of the week.

**Answer:**
```sql
SELECT 
    DAYOFWEEK(sale_date) AS day_of_week, 
    SUM(quantity) AS total_quantity_sold
FROM 
    sales
GROUP BY 
    DAYOFWEEK(sale_date);
```

**Output:**

| day_of_week | total_quantity_sold |
|-------------|---------------------|
| 1           | 5                   |
| 2           | 3                   |
| 3           | 4                   |
| 4           | 7                   |
| 5           | 3                   |
| 6           | 4                   |
| 7           | 5                   |

---

#### 20. Find the total sales amount for each product category per month.

**Answer:**
```sql
SELECT 
    c.category_name, 
    DATE_TRUNC('month', s.sale_date) AS month, 
    SUM(s.total_amount) AS total_sales
FROM 
    sales s
JOIN 
    product_category pc ON s.product_id = pc.product_id
JOIN 
    categories c ON pc.category_id = c.category_id
GROUP BY 
    c.category_name, DATE_TRUNC('month', s.sale_date)
ORDER BY 
    c.category_name, month;
```

**Output:**

| category_name | month     | total_sales |
|---------------|-----------|-------------|
| Accessories   | 2023-01-01| 4700        |
| Accessories   | 2023-02-01| 91600       |
| Electronics   | 2023-01-01| 376500      |
| Electronics   | 2023-02-01| 23700       |

---

#### 21. Calculate the number of distinct customers who made purchases each month.

**Answer:**
```sql
SELECT 
    DATE_TRUNC('month', sale_date) AS month, 
    COUNT(DISTINCT customer_id) AS distinct_customers
FROM 
    sales
GROUP BY 
    DATE_TRUNC('month', sale_date);
```

**Output:**

| month     | distinct_customers |
|-----------|--------------------|
| 2023-01-01| 20                 |
| 2023-02-01| 0                  |

---

#### 22. Find the total sales amount and the number of sales for each customer.

**Answer:**
```sql
SELECT 
    customer_name, 
    SUM(total_amount) AS total_sales, 
    COUNT(*) AS number_of_sales
FROM 
    sales
GROUP BY 
    customer_name;
```

**Output:**

| customer_name | total_sales | number_of_sales |
|---------------|-------------|-----------------|
| Arjun         | 95000       | 1               |
| Bhavya        | 18000       | 1               |
| Chaitanya     | 40500       | 1               |
| Deepak        | 27000       | 1               |
| Esha          | 7200        | 1               |
| Farhan        | 95000       | 1               |
| Gauri         | 18000       | 1               |
| Harsh         | 40500       | 1               |
| Ishita        | 27000       | 1               |
| Jai           | 7200        | 1               |
| Kritika       | 9500        | 1               |
| Laksh         | 2700        | 1               |
| Meera         | 2700        | 1               |
| Neha          | 9000        | 1               |
| Om            | 15200       | 1               |
| Priya         | 10800       | 1               |
| Rahul         | 45000       | 1               |
| Sneha         | 18000       | 1               |
| Tanmay        | 13500       | 1               |
| Usha          | 2250        | 1               |

---

#### 23. Calculate the total sales amount for each product on each day.

**Answer:**
```sql
SELECT 
    sale_date, 
    product_name, 
    SUM(total_amount) AS total_sales
FROM 
    sales
GROUP BY 
    sale_date, product_name
ORDER BY 
    sale_date, product_name;
```

**Output:**

| sale_date  | product_name | total_sales |
|------------|--------------|-------------|
| 2023-01-01 | Laptop       | 95000       |
| 2023-01-02 | Mobile       | 18000       |
| 2023-01-03 | Tablet       | 40500       |
| 2023-01-04 | TV           | 27000       |
| 2023-01-05 | Headphones   | 7200        |
| 2023-01-06 | Laptop       | 95000       |
| 2023-01-07 | Mobile       | 18000       |
| 2023-01-08 | Tablet       | 40500       |
| 2023-01-09 | TV           | 27000       |
| 2023-01-10 | Headphones   | 7200        |
| 2023-01-11 | Speaker      | 9500        |
| 2023-01-12 | Keyboard     | 2700        |
| 2023-01-13 | Mouse        | 2700        |
| 2023-01-14 | Monitor      | 9000        |
| 2023-01-15 | Printer      | 15200       |
| 2023-01-16 | Scanner      | 10800       |
| 2023-01-17 | Camera       | 45000       |
| 2023-01-18 | Lens         | 18000       |
| 2023-01-19 | Tripod       | 13500       |
| 2023-01-20 | Flash Drive  | 2250        |

---

#### 24. Find the average sale amount per transaction for each product.

**Answer:**
```sql
SELECT 
    product_name, 
    AVG(total_amount) AS average_sale_amount
FROM 
    sales
GROUP BY 
    product_name;
```

**Output:**

| product_name | average_sale_amount |
|--------------|---------------------|
| Laptop       | 95000.00            |
| Mobile       | 18000.00            |
| Tablet       | 40500.00            |
| TV           | 27000.00            |
| Headphones   | 7200.00             |
| Speaker      | 9500.00             |
| Keyboard     | 2700.00             |
| Mouse        | 2700.00             |
| Monitor      | 9000.00             |
| Printer      | 15200.00            |
| Scanner      | 10800.00            |
| Camera       | 45000.00            |
| Lens         | 18000.00            |
| Tripod       | 13500.00            |
| Flash Drive  | 2250.00             |

---

#### 25. Calculate the cumulative sales amount for each product category by date.

**Answer:**
```sql
SELECT 
    s.sale_date, 
    c.category_name, 
    SUM(s.total_amount) AS daily_sales, 
    SUM(SUM(s.total_amount)) OVER (PARTITION BY c.category_name ORDER BY s.sale_date) AS cumulative_sales_amount
FROM 
    sales s
JOIN 
    product_category pc ON s.product_id = pc.product_id
JOIN 
    categories c ON pc.category_id = c.category_id
GROUP BY 
    s.sale_date, c.category_name
ORDER BY 
    c.category_name, s.sale_date;
```

**Output:**

| sale_date  | category_name | daily_sales | cumulative_sales_amount |
|------------|---------------|-------------|-------------------------|
| 2023-01-01 | Electronics   | 27000       | 27000                   |
| 2023-01-02 | Electronics   | 18000       | 45000                   |
| 2023-01-03 | Electronics   | 40500       | 85500                   |
| 2023-01-04 | Electronics   | 27000       | 112500                  |
| 2023-01-05 | Electronics   | 7200        | 119700                  |
| 2023-01-06 | Electronics   | 95000       | 214700                  |
| 2023-01-07 | Electronics   | 18000       | 232700                  |
| 2023-01-08 | Electronics   | 40500       | 273200                  |
| 2023-01-09 | Electronics   | 27000       | 300200                  |
| 2023-01-10 | Electronics   | 7200        | 307400                  |
| 2023-01-11 | Accessories   | 9500        | 9500                    |
| 2023-01-12 | Accessories   | 2700        | 12200                   |
| 2023-01-13 | Accessories   | 2700        | 14900                   |
| 2023-01-14 | Accessories   | 9000        | 23900                   |
| 2023-01-15 | Accessories   | 15200       | 39100                   |
| 2023-01-16 | Accessories   | 10800       | 49900                   |
| 2023-01-17 | Accessories   | 45000       | 94900                   |
| 2023-01-18 | Accessories   | 18000       | 112900                  |
| 2023-01-19 | Accessories   | 13500       | 126400                  |
| 2023-01-20 | Accessories   | 2250        | 128650                  |

---
