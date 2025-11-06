# -----------------------------------------------------
# 1. Import Libraries
# -----------------------------------------------------
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder

# -----------------------------------------------------
#  2. Load Dataset
# -----------------------------------------------------
df = pd.read_csv(r"C:\Users\LENOVO\Desktop\Predective_Analysis\ecommerce_customers_unit1.csv")

print(" DATA LOADED SUCCESSFULLY\n")

# -----------------------------------------------------
# 3. Basic Exploration
# -----------------------------------------------------
print("FIRST 5 ROWS (HEAD):\n", df.head(), "\n")
print("LAST 5 ROWS (TAIL):\n", df.tail(), "\n")
print("DATA INFO:\n")
df.info()
print("\nDESCRIPTIVE STATISTICS (NUMERICAL):\n", df.describe(), "\n")
print("DESCRIPTIVE STATISTICS (CATEGORICAL):\n", df.describe(include='object'), "\n")
print("MISSING VALUES PER COLUMN:\n", df.isnull().sum(), "\n")
print("SHAPE OF DATASET:", df.shape, "\n")

# -----------------------------------------------------
#  4. Handling Missing Values (FFILL + BFILL)
# -----------------------------------------------------
df.fillna(method='ffill', inplace=True)
df.fillna(method='bfill', inplace=True)

print("🧹 Missing values handled successfully using Forward Fill & Backward Fill!\n")

# -----------------------------------------------------
#  5. Statistical Analysis
# -----------------------------------------------------
print("MEAN of Numerical Columns:\n", df.mean(numeric_only=True), "\n")
print("MEDIAN of Numerical Columns:\n", df.median(numeric_only=True), "\n")
print("MODE of Each Column:\n", df.mode().iloc[0], "\n")
print("VARIANCE:\n", df.var(numeric_only=True), "\n")
print("STANDARD DEVIATION:\n", df.std(numeric_only=True), "\n")

# Correlation Analysis
print(" CORRELATION MATRIX:\n", df.corr(numeric_only=True), "\n")

# Optional Heatmap
plt.figure(figsize=(10,6))
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.show()

# -----------------------------------------------------
#  6. Data Visualization Summary (Optional)
# -----------------------------------------------------
numeric_cols = df.select_dtypes(include=np.number).columns

# Histograms
df[numeric_cols].hist(figsize=(12,8), bins=20)
plt.suptitle("Distribution of Numerical Columns", fontsize=14)
plt.show()

# Boxplots for outlier check
plt.figure(figsize=(12,8))
sns.boxplot(data=df[numeric_cols])
plt.title("Boxplot for Outlier Detection")
plt.show()

# -----------------------------------------------------
# 7. Encoding Categorical Variables
# -----------------------------------------------------
label_encoder = LabelEncoder()

# Label Encode 'gender' if present
if 'gender' in df.columns:
    df['gender'] = label_encoder.fit_transform(df['gender'])

# One-Hot Encode multi-class categorical variables
onehot_cols = [col for col in df.select_dtypes(include='object').columns if col not in ['signup_date', 'last_purchase_date']]
df = pd.get_dummies(df, columns=onehot_cols, drop_first=True)

print("🏷️ Label Encoding & One-Hot Encoding applied successfully!\n")

# -----------------------------------------------------
# 🌟 8. Feature Scaling (Standardization & Normalization)
# -----------------------------------------------------
scaler_standard = StandardScaler()
scaler_minmax = MinMaxScaler()

num_cols = df.select_dtypes(include=np.number).columns

# Create scaled versions
df_standardized = df.copy()
df_standardized[num_cols] = scaler_standard.fit_transform(df_standardized[num_cols])

df_normalized = df.copy()
df_normalized[num_cols] = scaler_minmax.fit_transform(df_normalized[num_cols])

print("⚖️ Feature Scaling (Standardization & Normalization) done!\n")

# -----------------------------------------------------
# 9. Post-Processing Analysis
# -----------------------------------------------------
print("INFO AFTER PREPROCESSING:")
df.info()

print("\n🔍 PREVIEW (STANDARDIZED DATA):")
print(df_standardized.head(), "\n")

print("🔍 PREVIEW (NORMALIZED DATA):")
print(df_normalized.head(), "\n")

