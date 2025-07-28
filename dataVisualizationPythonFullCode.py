import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Load data
df = pd.read_csv('googleplaystore.csv')

# 2. Check nulls
print("\nMissing values per column:\n", df.isnull().sum())

# 3. Drop records with nulls
df = df.dropna()

# 4. Data Cleaning and Formatting

# -- Size column: Convert 'Varies with device', Mb, Kb to float (size in KB)
def size_to_kb(x):
    if x == "Varies with device":
        return np.nan
    if x[-1] == "M":
        return float(x[:-1]) * 1000
    if x[-1] == "k":
        return float(x[:-1])
    return np.nan

df['Size'] = df['Size'].map(size_to_kb)

# Fill still-empty sizes with the median of the respective Category
df['Size'] = df.groupby('Category')['Size'].transform(lambda x: x.fillna(x.median()))

# -- Reviews to int
df['Reviews'] = df['Reviews'].astype(int)

# -- Installs: remove + and , then convert to int
df['Installs'] = df['Installs'].str.replace('+','', regex=False).str.replace(',','', regex=False).astype(int)

# -- Price: remove $ and convert to float
df['Price'] = df['Price'].str.replace('$','', regex=False).astype(float)

# 5. Sanity checks

# -- Ratings should be between 1 and 5
df = df[(df['Rating'] >= 1) & (df['Rating'] <= 5)]

# -- Reviews can't exceed installs
df = df[df['Reviews'] <= df['Installs']]

# -- For Free apps, price must be 0
df = df[~((df['Type'] == "Free") & (df['Price'] > 0))]

# -- Remove any additional missing values that may have resulted
df = df.dropna()

# ==================
# 5b. EDA Plots (Distribution & Outliers)
# ==================
plt.figure(figsize=(14,8))
plt.subplot(2,2,1)
sns.boxplot(x=df['Price'])
plt.title('Price Boxplot')

plt.subplot(2,2,2)
sns.boxplot(x=df['Reviews'])
plt.title('Reviews Boxplot')

plt.subplot(2,2,3)
sns.histplot(df['Rating'], kde=True, bins=20)
plt.title('Rating Histogram')

plt.subplot(2,2,4)
sns.histplot(df['Size'], kde=True, bins=30)
plt.title('Size Histogram')
plt.tight_layout()
plt.show()

# ==================
# 6. Outlier Treatment
# ==================
# Price: drop apps with price > $200
print("Prices > $200:", df[df['Price'] > 200][['App','Price']])
df = df[df['Price'] <= 200]

# Reviews: drop records with Reviews > 2,000,000
print("Reviews > 2M:", df[df['Reviews'] > 2_000_000][['App','Reviews']])
df = df[df['Reviews'] <= 2_000_000]

# Installs outliers
percentiles = df['Installs'].quantile([0.10,0.25,0.5,0.7,0.9,0.95,0.99])
print("\nInstalls percentiles:\n", percentiles)
# We'll use 99th percentile as cutoff (for ex: ~100,000,000)
INSTALLS_CUTOFF = percentiles.loc[0.99]
df = df[df['Installs'] <= INSTALLS_CUTOFF]

# ==================
# 7. BIVARIATE PLOTS
# ==================
# Rating vs Price
plt.figure(figsize=(12,5))
sns.jointplot(x='Price', y='Rating', data=df, kind='scatter', height=6)
plt.title('Rating vs Price')
plt.show()

# Rating vs Size
sns.jointplot(x='Size', y='Rating', data=df, kind='scatter', height=6)
plt.title('Rating vs Size')
plt.show()

# Rating vs Reviews (log reviews for clarity)
sns.jointplot(x=np.log1p(df['Reviews']), y='Rating', data=df, kind='scatter', height=6)
plt.title('Rating vs log(Reviews)')
plt.show()

# Boxplot of Rating vs Content Rating
plt.figure(figsize=(12,6))
sns.boxplot(x='Content Rating', y='Rating', data=df)
plt.xticks(rotation=30)
plt.title('Rating vs Content Rating')
plt.show()

# Boxplot of Rating vs Category (Top N)
top_cats = df['Category'].value_counts().index[:10]
plt.figure(figsize=(16,6))
sns.boxplot(x='Category', y='Rating', data=df[df['Category'].isin(top_cats)])
plt.title('Rating vs Top Categories')
plt.show()

# Comment on the distribution and outliers in your report (not as code).

# ==================
# 8. Data Preprocessing
# ==================
inp1 = df.copy()
# Log transformation (to reduce skew)
inp1['Reviews'] = np.log1p(inp1['Reviews'])
inp1['Installs'] = np.log1p(inp1['Installs'])

# Drop columns not useful for modeling
cols_to_drop = ['App', 'Last Updated', 'Current Ver', 'Android Ver']
inp1 = inp1.drop(cols_to_drop, axis=1)

# Get dummies for Category, Genres, Content Rating, Type
inp2 = pd.get_dummies(inp1, columns=['Category','Genres','Content Rating','Type'], drop_first=True)

# ==================
# 9. Train-test split
# ==================
from sklearn.model_selection import train_test_split

df_train, df_test = train_test_split(inp2, test_size=0.3, random_state=0)

# Separate X/y
X_train = df_train.drop('Rating', axis=1)
y_train = df_train['Rating']
X_test = df_test.drop('Rating', axis=1)
y_test = df_test['Rating']

# ==================
# 11. Model Building - Linear Regression
# ==================
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

lm = LinearRegression()
lm.fit(X_train, y_train)

# R2 on train set
r2_train = lm.score(X_train, y_train)
print(f"\nTrain R2: {r2_train:.3f}")

# ==================
# 12. Predict on test set, report R2
# ==================
y_pred = lm.predict(X_test)
r2_test = r2_score(y_test, y_pred)
print(f"Test R2: {r2_test:.3f}")

# Optionally: show coefficients for top features
coef = pd.Series(lm.coef_, index=X_train.columns)
print("\nTop 10 features by weight:\n", coef.abs().sort_values(ascending=False).head(10))