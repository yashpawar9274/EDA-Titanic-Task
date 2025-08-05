import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

# Create folders
os.makedirs('dataset', exist_ok=True)
os.makedirs('images', exist_ok=True)

# Load dataset (adjust path as needed)
df = pd.read_csv('dataset/titanic.csv')

# Basic Info
print(df.info())
print(df.describe())

# Handle missing values safely
df.fillna({
    'Age': df['Age'].median(),
    'Embarked': df['Embarked'].mode()[0]
}, inplace=True)

# 🎯 Boxplot: Age vs Pclass
plt.figure(figsize=(8, 6))
sns.boxplot(data=df, x='Pclass', y='Age')
plt.title('Boxplot: Age vs Pclass')
plt.savefig('images/boxplot.png')
plt.show()

# 🔥 Heatmap: Correlation Matrix
plt.figure(figsize=(10, 6))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Matrix Heatmap')
plt.savefig('images/heatmap.png')
plt.show()
