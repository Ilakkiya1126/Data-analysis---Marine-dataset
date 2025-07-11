import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ------------------ Load Dataset ------------------
df = pd.read_csv("marine-economy.csv")

# ------------------ Preprocessing ------------------

# Rename columns to lowercase and underscore format
df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')

# Convert year and data_value to numeric
df['year'] = pd.to_numeric(df['year'], errors='coerce')
df['data_value'] = pd.to_numeric(df['data_value'], errors='coerce')

# Drop rows with missing year or data_value
df = df.dropna(subset=['year', 'data_value'])

# Remove duplicate rows
df = df.drop_duplicates()

# Normalize text fields (strip and lowercase)
for col in ['category', 'variable', 'units', 'magnitude', 'source', 'flag']:
    df[col] = df[col].astype(str).str.strip().str.lower()

# Filter only actual magnitude
df = df[df['magnitude'] == 'actual'].reset_index(drop=True)

# ------------------ Basic Info ------------------
print("First 5 Records:")
print(df.head())

print("\nData Types and Non-Null Counts:")
print(df.info())

print("\nDescriptive Statistics:")
print(df.describe())

# ------------------ Visualization ------------------

# Set seaborn style
sns.set(style="whitegrid")
plt.rcParams["figure.figsize"] = (12, 6)

# 1. Line Plot: data_value over years for each category
plt.figure()
sns.lineplot(data=df, x="year", y="data_value", hue="category", marker="o")
plt.title("Trend of Data Value Over Years by Category")
plt.ylabel("Data Value")
plt.xlabel("Year")
plt.legend(title="Category", bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()

# 2. Box Plot: distribution by category
plt.figure()
sns.boxplot(data=df, x="category", y="data_value")
plt.title("Distribution of Data Value by Category")
plt.ylabel("Data Value")
plt.xlabel("Category")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# 3. Heatmap: average data_value by category and year
pivot_table = df.pivot_table(values='data_value', index='category', columns='year', aggfunc='mean')
plt.figure()
sns.heatmap(pivot_table, annot=True, fmt=".1f", cmap="YlGnBu", linewidths=0.5)
plt.title("Average Data Value Heatmap (Category vs Year)")
plt.ylabel("Category")
plt.xlabel("Year")
plt.tight_layout()
plt.show()
