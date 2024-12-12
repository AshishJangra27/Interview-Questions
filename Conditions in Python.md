# Practical If-Else and Nested If-Else Exercises

Below are 50 practical Python practice questions focusing on **if-else**, **nested if-else**,

---

```python
# Q1: Nested if: Check nationality.
# If 'India', print 'Namaste'.
# If 'USA', print 'Hello'.
# If 'France', print 'Bonjour'.
# Else print 'Hi'.
nationality = 'India'
if nationality == 'India':
    print('Namaste')
elif nationality == 'USA':
    print('Hello')
elif nationality == 'France':
    print('Bonjour')
else:
    print('Hi')
```

```python
# Q2: Nested if: If income > 1000000, then if >2000000: 'Very High', else 'High'.
# Else if income > 500000: 'Medium', else 'Low'.
income = 1500000
if income > 1000000:
    if income > 2000000:
        print('Very High')
    else:
        print('High')
elif income > 500000:
    print('Medium')
else:
    print('Low')
```

```python
# Q3: Multi-elif: Check car_color.
# 'red'->'Stop sign color', 'green'->'Go color', 'yellow'->'Caution', else 'Unknown'.
car_color = 'yellow'
if car_color == 'red':
    print('Stop sign color')
elif car_color == 'green':
    print('Go color')
elif car_color == 'yellow':
    print('Caution')
else:
    print('Unknown')
```

```python
# Q4: Nested if: If user_age >=18:
#   if age<21 print 'Adult but no alcohol', else 'Adult with alcohol'
# else 'Minor'.
user_age = 19
if user_age >= 18:
    if user_age < 21:
        print('Adult but no alcohol')
    else:
        print('Adult with alcohol')
else:
    print('Minor')
```

```python
# Q5: Multi-elif: Temperature ranges:
# <0='Freezing', <15='Cold', <25='Warm', else 'Hot'.
temp = 10
if temp < 0:
    print('Freezing')
elif temp < 15:
    print('Cold')
elif temp < 25:
    print('Warm')
else:
    print('Hot')
```

```python
# Q6: Nested if: If exam_score >=90: 'A'
# Else if >=80: if >=85:'B+' else:'B'
# Else 'Below B'.
exam_score = 83
if exam_score >= 90:
    print('A')
elif exam_score >= 80:
    if exam_score >= 85:
        print('B+')
    else:
        print('B')
else:
    print('Below B')
```

```python
# Q7: Multi-elif: Day number (1=Mon,...7=Sun)
# Mon-Fri='Workday', Sat='Half-day', Sun='Holiday', else 'Invalid'.
day_num = 6
if day_num == 1 or day_num == 2 or day_num == 3 or day_num == 4 or day_num == 5:
    print('Workday')
elif day_num == 6:
    print('Half-day')
elif day_num == 7:
    print('Holiday')
else:
    print('Invalid')
```

```python
# Q8: Nested if: If user_role='admin':
#   if active=True: 'Full Access', else 'Admin not active'
# else 'Limited Access'.
user_role = 'admin'
active = True
if user_role == 'admin':
    if active == True:
        print('Full Access')
    else:
        print('Admin not active')
else:
    print('Limited Access')
```

```python
# Q9: Multi-elif: Speed zones:
# <30='Slow zone', <60='Normal', <90='Fast', else 'Very Fast'.
speed = 75
if speed < 30:
    print('Slow zone')
elif speed < 60:
    print('Normal')
elif speed < 90:
    print('Fast')
else:
    print('Very Fast')
```

```python
# Q10: Nested if: If budget>1000:
#   if >2000='Luxury', else='Moderate'
# Else if >500='Basic', else='Low'.
budget = 1500
if budget > 1000:
    if budget > 2000:
        print('Luxury')
    else:
        print('Moderate')
elif budget > 500:
    print('Basic')
else:
    print('Low')
```

```python
# Q11: Multi-elif: Letter grade from marks:
# >=90='A', >=80='B', >=70='C', else='D'.
marks = 72
if marks >= 90:
    print('A')
elif marks >= 80:
    print('B')
elif marks >= 70:
    print('C')
else:
    print('D')
```

```python
# Q12: Nested if: If candidate_age>=18:
#   if >=65:'Senior Voter', else:'Adult Voter'
# else:'Underage'.
candidate_age = 70
if candidate_age >= 18:
    if candidate_age >= 65:
        print('Senior Voter')
    else:
        print('Adult Voter')
else:
    print('Underage')
```

```python
# Q13: Multi-elif: BMI categories:
# <18.5='Underweight', <25='Normal', <30='Overweight', else='Obese'.
bmi = 27
if bmi < 18.5:
    print('Underweight')
elif bmi < 25:
    print('Normal')
elif bmi < 30:
    print('Overweight')
else:
    print('Obese')
```

```python
# Q14: Nested if: If rainfall>50:
#   if >100='Severe Flood', else='Flood Warning'
# else 'No Flood'.
rainfall = 110
if rainfall > 50:
    if rainfall > 100:
        print('Severe Flood')
    else:
        print('Flood Warning')
else:
    print('No Flood')
```

```python
# Q15: Multi-elif: Language codes:
# 'en'='English', 'fr'='French', 'hi'='Hindi', else='Unknown'.
lang_code = 'fr'
if lang_code == 'en':
    print('English')
elif lang_code == 'fr':
    print('French')
elif lang_code == 'hi':
    print('Hindi')
else:
    print('Unknown')
```

```python
# Q16: Nested if: If product_price>500:
#   if >1000='High-end', else='Mid-range'
# else 'Budget'.
product_price = 300
if product_price > 500:
    if product_price > 1000:
        print('High-end')
    else:
        print('Mid-range')
else:
    print('Budget')
```

```python
# Q17: Multi-elif: Age group:
# <13='Child', <20='Teen', <60='Adult', else='Senior'.
age = 45
if age < 13:
    print('Child')
elif age < 20:
    print('Teen')
elif age < 60:
    print('Adult')
else:
    print('Senior')
```

```python
# Q18: Nested if: If electricity_units>500:
#   if >1000='High Bill', else='Medium Bill'
# else 'Low Bill'.
electricity_units = 1200
if electricity_units > 500:
    if electricity_units > 1000:
        print('High Bill')
    else:
        print('Medium Bill')
else:
    print('Low Bill')
```

```python
# Q19: Multi-elif: Meal time by hour:
# <10='breakfast', <15='lunch', <21='dinner', else='late snack'.
hour = 16
if hour < 10:
    print('breakfast')
elif hour < 15:
    print('lunch')
elif hour < 21:
    print('dinner')
else:
    print('late snack')
```

```python
# Q20: Nested if: If password_len>=8:
#   if has_digit='Strong', else='Moderate'
# else 'Weak'.
password = 'abc1234'
has_digit = ('0' in password or '1' in password or '2' in password or 
             '3' in password or '4' in password or '5' in password or
             '6' in password or '7' in password or '8' in password or '9' in password)

if len(password) >= 8:
    if has_digit:
        print('Strong')
    else:
        print('Moderate')
else:
    print('Weak')
```

```python
# Q21: Multi-elif: Height check:
# <150='Short', <170='Below average', <185='Average', else='Tall'.
height = 160
if height < 150:
    print('Short')
elif height < 170:
    print('Below average')
elif height < 185:
    print('Average')
else:
    print('Tall')
```

```python
# Q22: Nested if: If tickets_sold>1000:
#   if >5000='Blockbuster', else='Hit'
# else if >500='Average', else='Flop'.
tickets_sold = 600
if tickets_sold > 1000:
    if tickets_sold > 5000:
        print('Blockbuster')
    else:
        print('Hit')
elif tickets_sold > 500:
    print('Average')
else:
    print('Flop')
```

```python
# Q23: Multi-elif: Over speed categories:
# over_speed=0:'Safe', <=10:'Warning', <=20:'Penalty', else:'Severe Penalty'.
over_speed = 15
if over_speed == 0:
    print('Safe')
elif over_speed <= 10:
    print('Warning')
elif over_speed <= 20:
    print('Penalty')
else:
    print('Severe Penalty')
```

```python
# Q24: Nested if: If exam_passed=True:
#   if interview_passed=True:'Job Offer', else:'No Offer'
# else:'No Offer'.
exam_passed = True
interview_passed = False
if exam_passed == True:
    if interview_passed == True:
        print('Job Offer')
    else:
        print('No Offer')
else:
    print('No Offer')
```

```python
# Q25: Multi-elif: Month number:
# 1='Jan',2='Feb',3='Mar', else='Other month'.
month_num = 3
if month_num == 1:
    print('Jan')
elif month_num == 2:
    print('Feb')
elif month_num == 3:
    print('Mar')
else:
    print('Other month')
```

```python
# Q26: Nested if: If score>80:
#   if >90='Excellent', else='Good'
# else if >60='Average', else 'Poor'.
score = 75
if score > 80:
    if score > 90:
        print('Excellent')
    else:
        print('Good')
elif score > 60:
    print('Average')
else:
    print('Poor')
```

```python
# Q27: Multi-elif: Credit score rating:
# >=800='Excellent', >=700='Good', >=600='Fair', else='Poor'.
credit_score = 650
if credit_score >= 800:
    print('Excellent')
elif credit_score >= 700:
    print('Good')
elif credit_score >= 600:
    print('Fair')
else:
    print('Poor')
```

```python
# Q28: Nested if: If temperature>30:
#   if >40='Heatwave', else='Hot'
# else if >20='Warm', else 'Cool'.
temperature = 25
if temperature > 30:
    if temperature > 40:
        print('Heatwave')
    else:
        print('Hot')
elif temperature > 20:
    print('Warm')
else:
    print('Cool')
```

```python
# Q29: Multi-elif: Salary range:
# >=100000='Top Earner', >=50000='Middle', >=20000='Low Middle', else='Low'.
salary = 30000
if salary >= 100000:
    print('Top Earner')
elif salary >= 50000:
    print('Middle')
elif salary >= 20000:
    print('Low Middle')
else:
    print('Low')
```

```python
# Q30: Nested if: If fuel>10:
#   if >50='Full Tank', else='Partial'
# else if >0='Low fuel', else 'Empty'.
fuel = 5
if fuel > 10:
    if fuel > 50:
        print('Full Tank')
    else:
        print('Partial')
elif fuel > 0:
    print('Low fuel')
else:
    print('Empty')
```

```python
# Q31: Multi-elif: Internet speed:
# >=100='High Speed', >=50='Medium Speed', >=10='Low Speed', else='Very Low'.
internet_speed = 8
if internet_speed >= 100:
    print('High Speed')
elif internet_speed >= 50:
    print('Medium Speed')
elif internet_speed >= 10:
    print('Low Speed')
else:
    print('Very Low')
```

```python
# Q32: Nested if: If distance>1000:
#   if >2000='Very Far', else='Far'
# else if >500='Moderate', else 'Close'.
distance = 1800
if distance > 1000:
    if distance > 2000:
        print('Very Far')
    else:
        print('Far')
elif distance > 500:
    print('Moderate')
else:
    print('Close')
```

```python
# Q33: Multi-elif: Customer rating:
# 5='Excellent',4='Good',3='Average',2='Below Average',1='Poor', else='Invalid'.
rating = 4
if rating == 5:
    print('Excellent')
elif rating == 4:
    print('Good')
elif rating == 3:
    print('Average')
elif rating == 2:
    print('Below Average')
elif rating == 1:
    print('Poor')
else:
    print('Invalid')
```

```python
# Q34: Nested if: If marks_obtained>80:
#   if >90='Outstanding', else='Great'
# else if >50='Okay', else 'Fail'.
marks_obtained = 55
if marks_obtained > 80:
    if marks_obtained > 90:
        print('Outstanding')
    else:
        print('Great')
elif marks_obtained > 50:
    print('Okay')
else:
    print('Fail')
```

```python
# Q35: Multi-elif: Weather code:
# 'S'='Sunny','R'='Rainy','C'='Cloudy','W'='Windy', else='Unknown'.
weather_code = 'C'
if weather_code == 'S':
    print('Sunny')
elif weather_code == 'R':
    print('Rainy')
elif weather_code == 'C':
    print('Cloudy')
elif weather_code == 'W':
    print('Windy')
else:
    print('Unknown')
```

```python
# Q36: Nested if: If num>0:
#   if even='Positive Even', else='Positive Odd'
# else if num<0='Negative', else 'Zero'.
num = 14
if num > 0:
    if num % 2 == 0:
        print('Positive Even')
    else:
        print('Positive Odd')
elif num < 0:
    print('Negative')
else:
    print('Zero')
```

```python
# Q37: Multi-elif: Year category:
# <1900='Historic', <2000='20th Century', <2020='Modern', else='Recent'.
year = 1985
if year < 1900:
    print('Historic')
elif year < 2000:
    print('20th Century')
elif year < 2020:
    print('Modern')
else:
    print('Recent')
```

```python
# Q38: Nested if: If stock_level>100:
#   if >500='Surplus', else='Sufficient'
# else if >0='Limited', else 'Out of stock'.
stock_level = 50
if stock_level > 100:
    if stock_level > 500:
        print('Surplus')
    else:
        print('Sufficient')
elif stock_level > 0:
    print('Limited')
else:
    print('Out of stock')
```

```python
# Q39: Multi-elif: Priority level:
# 1='High',2='Medium',3='Low', else='No priority'.
priority = 1
if priority == 1:
    print('High')
elif priority == 2:
    print('Medium')
elif priority == 3:
    print('Low')
else:
    print('No priority')
```

```python
# Q40: Nested if: If attempts>3:
#   if >5='Locked Out', else='One last try'
# else if >0='Try again', else 'No attempt left'.
attempts = 6
if attempts > 3:
    if attempts > 5:
        print('Locked Out')
    else:
        print('One last try')
elif attempts > 0:
    print('Try again')
else:
    print('No attempt left')
```

```python
# Q41: Multi-elif: Temperature scale:
# 'C'='Celsius', 'F'='Fahrenheit', 'K'='Kelvin', else='Unknown'.
scale = 'F'
if scale == 'C':
    print('Celsius')
elif scale == 'F':
    print('Fahrenheit')
elif scale == 'K':
    print('Kelvin')
else:
    print('Unknown')
```

```python
# Q42: Nested if: If storage>100GB:
#   if >500GB='Large Storage', else='Moderate'
# else if >50GB='Small', else 'Minimal'.
storage = 60
if storage > 100:
    if storage > 500:
        print('Large Storage')
    else:
        print('Moderate')
elif storage > 50:
    print('Small')
else:
    print('Minimal')
```

```python
# Q43: Multi-elif: Time of day by hour:
# <6='Early Morning', <12='Morning', <18='Afternoon', <24='Night'.
hour_of_day = 5
if hour_of_day < 6:
    print('Early Morning')
elif hour_of_day < 12:
    print('Morning')
elif hour_of_day < 18:
    print('Afternoon')
elif hour_of_day < 24:
    print('Night')
else:
    print('Invalid hour')
```

```python
# Q44: Nested if: If cpu_usage>70%:
#   if >90%='Overloaded', else='High Load'
# else if >30%='Normal', else 'Low Load'.
cpu_usage = 95
if cpu_usage > 70:
    if cpu_usage > 90:
        print('Overloaded')
    else:
        print('High Load')
elif cpu_usage > 30:
    print('Normal')
else:
    print('Low Load')
```

```python
# Q45: Multi-elif: Check feedback char:
# 'A'='Excellent', 'B'='Good', 'C'='Fair', 'D'='Poor', else='Unknown'.
feedback = 'D'
if feedback == 'A':
    print('Excellent')
elif feedback == 'B':
    print('Good')
elif feedback == 'C':
    print('Fair')
elif feedback == 'D':
    print('Poor')
else:
    print('Unknown')
```

```python
# Q46: Nested if: If hours_worked>40:
#   if >60='Excessive OT', else='Overtime'
# else if >20='Part-time', else 'Minimal'.
hours_worked = 62
if hours_worked > 40:
    if hours_worked > 60:
        print('Excessive OT')
    else:
        print('Overtime')
elif hours_worked > 20:
    print('Part-time')
else:
    print('Minimal')
```

```python
# Q47: Multi-elif: Product rating:
# >=4.5='Excellent', >=4='Good', >=3='Average', else='Poor'.
product_rating = 3.5
if product_rating >= 4.5:
    print('Excellent')
elif product_rating >= 4:
    print('Good')
elif product_rating >= 3:
    print('Average')
else:
    print('Poor')
```

```python
# Q48: Nested if: If grade_level>10:
#   if >12='College', else='High School'
# else if >5='Primary', else 'Kindergarten'.
grade_level = 11
if grade_level > 10:
    if grade_level > 12:
        print('College')
    else:
        print('High School')
elif grade_level > 5:
    print('Primary')
else:
    print('Kindergarten')
```

```python
# Q49: Multi-elif: Performance score:
# >=90='Star', >=75='Good', >=50='Average', else='Needs Improvement'.
performance_score = 95
if performance_score >= 90:
    print('Star')
elif performance_score >= 75:
    print('Good')
elif performance_score >= 50:
    print('Average')
else:
    print('Needs Improvement')
```

```python
# Q50: Nested if: If balance>10000:
#   if >50000='Wealthy', else='Comfortable'
# else if >0='Stable', else 'Debt'.
balance = 60000
if balance > 10000:
    if balance > 50000:
        print('Wealthy')
    else:
        print('Comfortable')
elif balance > 0:
    print('Stable')
else:
    print('Debt')
```
