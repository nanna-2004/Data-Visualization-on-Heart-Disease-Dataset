import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv('heart.csv')
print(df.head(10))

# Basic statistics
print(df.describe())
print(df.info())

# Distribution plots
sns.displot(df['Age'], kde=True, color='blue')
plt.title("Age Distribution")
plt.show()

sns.displot(df['RestingBP'], kde=True, color='blue')
plt.title("Resting Blood Pressure Distribution")
plt.show()

sns.displot(df['Cholesterol'], kde=True, color='blue')
plt.title("Cholesterol Distribution")
plt.show()

sns.displot(df['MaxHR'], kde=True, color='blue')
plt.title("Max Heart Rate Distribution")
plt.show()

# Pie charts for categorical features
for feature in ['Sex', 'ChestPainType', 'RestingECG', 'ST_Slope', 'HeartDisease']:
    df.groupby(feature).size().plot(kind='pie', autopct='%.1f%%', figsize=(6,6))
    plt.title(f"Distribution of {feature}")
    plt.ylabel('')
    plt.show()

# Violin plots for visualizing distributions with HeartDisease
sns.violinplot(x=df['Age'])
plt.title("Age Violin Plot")
plt.show()

sns.violinplot(x='HeartDisease', y='Sex', data=df)
plt.title("Sex vs Heart Disease")
plt.show()

sns.violinplot(x='HeartDisease', y='Age', data=df)
plt.title("Age vs Heart Disease")
plt.show()

sns.violinplot(x='HeartDisease', y='Cholesterol', data=df)
plt.title("Cholesterol vs Heart Disease")
plt.show()

# Correlation heatmap using only numeric columns
numeric_df = df.select_dtypes(include=[np.number])
plt.figure(figsize=(10, 8))
sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.show()

# Pairplot
sns.pairplot(df)
plt.show()

# Jointplots for relationship exploration
sns.jointplot(x='Age', y='MaxHR', data=df, kind='hex')
plt.show()

sns.jointplot(x='Age', y='MaxHR', data=df, kind='reg')
plt.show()

sns.jointplot(x='Cholesterol', y='MaxHR', data=df, kind='reg')
plt.show()

sns.jointplot(x='HeartDisease', y='MaxHR', data=df, kind='reg')
plt.show()

