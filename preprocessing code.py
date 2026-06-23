import pandas as pd
from sqlalchemy import create_engine
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# ----------------------------------------------------------
# DATABASE CONNECTION
# ----------------------------------------------------------

engine = create_engine(
    "mysql+pymysql://root:vaish%402605@localhost:3306/ecommerce"
)

# Load complete table
df = pd.read_sql(
    "SELECT * FROM ecommerce_sales",
    engine
)

# ----------------------------------------------------------
# CLEAN DATA
# ----------------------------------------------------------

df.columns = df.columns.str.strip()

df['Sales'] = pd.to_numeric(df['Sales'], errors='coerce')
df['Profit'] = pd.to_numeric(df['Profit'], errors='coerce')
df['Quantity'] = pd.to_numeric(df['Quantity'], errors='coerce')

df['Order_Date'] = pd.to_datetime(
    df['Order_Date'],
    errors='coerce'
)

df.dropna(inplace=True)

print("Shape:", df.shape)
print("\nData Types:")
print(df.dtypes)

print("\nLAST 5 RECORDS")
print(df.tail())

# ----------------------------------------------------------
# 5. SHAPE
# ----------------------------------------------------------

print("\nDATASET SHAPE")
print(df.shape)

print(f"Rows : {df.shape[0]}")
print(f"Columns : {df.shape[1]}")

# ----------------------------------------------------------
# 6. COLUMN NAMES
# ----------------------------------------------------------

print("\nCOLUMN NAMES")
print(df.columns.tolist())

# ----------------------------------------------------------
# 7. DATA TYPES
# ----------------------------------------------------------

print("\nDATA TYPES")
print(df.dtypes)

# ----------------------------------------------------------
# 8. DATASET INFO
# ----------------------------------------------------------

print("\nDATASET INFO")
df.info()

# ----------------------------------------------------------
# 9. DESCRIPTIVE STATISTICS
# ----------------------------------------------------------

print("\nSTATISTICAL SUMMARY")
print(df.describe())

# ----------------------------------------------------------
# 10. MISSING VALUES
# ----------------------------------------------------------

print("\nMISSING VALUES")
print(df.isnull().sum())

# ----------------------------------------------------------
# 11. DUPLICATE RECORDS
# ----------------------------------------------------------

duplicates = df.duplicated().sum()

print("\nDUPLICATE RECORDS")
print(duplicates)

# ----------------------------------------------------------
# 12. DATE CONVERSION
# ----------------------------------------------------------

df['Order_Date'] = pd.to_datetime(
    df['Order_Date'],
    errors='coerce'
)

# ----------------------------------------------------------
# 13. FEATURE ENGINEERING
# ----------------------------------------------------------

df['Year'] = df['Order_Date'].dt.year
df['Month'] = df['Order_Date'].dt.month
df['Month_Name'] = df['Order_Date'].dt.month_name()
df['Day'] = df['Order_Date'].dt.day

df['Profit_Margin'] = (
    (df['Profit'] / df['Sales']) * 100
).round(2)

print("\nFEATURE ENGINEERING COMPLETED")

# ----------------------------------------------------------
# 14. UNIQUE VALUES
# ----------------------------------------------------------

print("\nUNIQUE CATEGORIES")
print(df['Category'].unique())

print("\nUNIQUE REGIONS")
print(df['Region'].unique())

# ----------------------------------------------------------
# 15. SALES DISTRIBUTION
# ----------------------------------------------------------

plt.figure(figsize=(8,5))
plt.hist(df['Sales'], bins=20)
plt.title("Sales Distribution")
plt.xlabel("Sales")
plt.ylabel("Frequency")
plt.tight_layout()
plt.show()

# ----------------------------------------------------------
# 16. PROFIT DISTRIBUTION
# ----------------------------------------------------------

plt.figure(figsize=(8,5))
plt.hist(df['Profit'], bins=20)
plt.title("Profit Distribution")
plt.xlabel("Profit")
plt.ylabel("Frequency")
plt.tight_layout()
plt.show()

# ----------------------------------------------------------
# 17. CATEGORY-WISE SALES
# ----------------------------------------------------------

category_sales = (
    df.groupby('Category')['Sales']
      .sum()
      .sort_values(ascending=False)
)

print("\nCATEGORY SALES")
print(category_sales)

plt.figure(figsize=(8,5))
category_sales.plot(kind='bar')
plt.title("Category Wise Sales")
plt.xlabel("Category")
plt.ylabel("Total Sales")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# ----------------------------------------------------------
# 18. REGION-WISE PROFIT
# ----------------------------------------------------------

region_profit = (
    df.groupby('Region')['Profit']
      .sum()
      .sort_values(ascending=False)
)

print("\nREGION PROFIT")
print(region_profit)

plt.figure(figsize=(8,5))
region_profit.plot(kind='bar')
plt.title("Region Wise Profit")
plt.xlabel("Region")
plt.ylabel("Total Profit")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# ----------------------------------------------------------
# 19. TOP 10 PRODUCTS
# ----------------------------------------------------------

top_products = (
    df.groupby('Product_Name')['Sales']
      .sum()
      .sort_values(ascending=False)
      .head(10)
)

print("\nTOP 10 PRODUCTS")
print(top_products)

plt.figure(figsize=(10,5))
top_products.plot(kind='bar')
plt.title("Top 10 Products")
plt.xlabel("Product Name")
plt.ylabel("Total Sales")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# ----------------------------------------------------------
# 20. MONTHLY SALES TREND
# ----------------------------------------------------------

monthly_sales = (
    df.groupby(['Year', 'Month_Name'])['Sales']
      .sum()
      .reset_index()
)

print("\nMONTHLY SALES TREND")
print(monthly_sales)

# ----------------------------------------------------------
# 21. CORRELATION MATRIX
# ----------------------------------------------------------

numeric_df = df.select_dtypes(include=np.number)

corr_matrix = numeric_df.corr()

print("\nCORRELATION MATRIX")
print(corr_matrix)

plt.figure(figsize=(10,6))
sns.heatmap(
    corr_matrix,
    annot=True
)
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.show()

# ----------------------------------------------------------
# 22. OUTLIER DETECTION
# ----------------------------------------------------------

plt.figure(figsize=(8,5))
sns.boxplot(x=df['Sales'])
plt.title("Sales Outliers")
plt.tight_layout()
plt.show()

plt.figure(figsize=(8,5))
sns.boxplot(x=df['Profit'])
plt.title("Profit Outliers")
plt.tight_layout()
plt.show()

# ----------------------------------------------------------
# 23. BUSINESS INSIGHTS
# ----------------------------------------------------------

print("\nBUSINESS INSIGHTS")

print("Highest Sales Category :", category_sales.idxmax())
print("Most Profitable Region :", region_profit.idxmax())
print("Best Selling Product :", top_products.index[0])

# ----------------------------------------------------------
# SAVE CLEANED DATA
# ----------------------------------------------------------

df.to_csv(
    "ecommerce_sales_cleaned.csv",
    index=False
)

df.to_sql(
    name='ecommerce_sales_cleaned',
    con=engine,
    if_exists='replace',
    index=False
)

print("\n✅ Cleaned dataset saved successfully")
print("✅ Cleaned dataset stored in MySQL")
