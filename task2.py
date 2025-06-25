import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px

# Load dataset
df = pd.read_csv("titanic.csv")

# Show basic info
print("Dataset Shape:", df.shape)
print("\nData Types:\n", df.dtypes)
print("\nMissing Values:\n", df.isnull().sum())
print("\nSummary Statistics:\n", df.describe(include='all'))

# Handle missing values if columns exist
if 'Age' in df.columns:
    df['Age'] = df['Age'].fillna(df['Age'].median())

if 'Embarked' in df.columns:
    df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])

if 'Cabin' in df.columns:
    df.drop(columns=['Cabin'], inplace=True)

# Encode categorical columns 
if 'Sex' in df.columns:
    df['Sex'] = df['Sex'].map({'male': 1, 'female': 0})

if 'Embarked' in df.columns:
    df = pd.get_dummies(df, columns=['Embarked'], drop_first=True)

# visualization

# 1. Histogram of Age
plt.figure(figsize=(8, 5))
sns.histplot(df['Age'], bins=30, kde=True)
plt.title('Age Distribution')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.show()

# 2. Boxplot: Age by Pclass
plt.figure(figsize=(8, 5))
sns.boxplot(x='Pclass', y='Age', data=df)
plt.title('Age by Passenger Class')
plt.show()

# 3. Countplot: Survival
plt.figure(figsize=(6, 4))
sns.countplot(x='Survived', data=df)
plt.title('Survival Count')
plt.xticks([0, 1], ['Not Survived', 'Survived'])
plt.show()

# 4. Countplot: Sex vs Survived (check column name)
if 'Sex' in df.columns:
    plt.figure(figsize=(6, 4))
    sns.countplot(x='Sex', hue='Survived', data=df)
    plt.title('Survival by Sex')
    plt.xticks([0, 1], ['Female', 'Male'])
    plt.show()

# 5. Pie chart for Embarked (load original to get that column)
df_raw = pd.read_csv("titanic.csv")
if 'Embarked' in df_raw.columns:
    df_raw['Embarked'] = df_raw['Embarked'].fillna(df_raw['Embarked'].mode()[0])
    df_raw['Embarked'].value_counts().plot.pie(autopct='%1.1f%%', title='Port of Embarkation')
    plt.ylabel('')
    plt.show()

# 6. Correlation Heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.show()

# 7. Pairplot (subset)
sns.pairplot(df[['Age', 'Fare', 'Pclass', 'Survived']])
plt.show()

# 8. Interactive Plot with Plotly
fig = px.scatter(df, x='Age', y='Fare', color=df['Survived'].map({0: 'Not Survived', 1: 'Survived'}),
                 title='Age vs Fare Colored by Survival')
fig.show()

# ---------- Observations ----------
print("\n📌 Key Observations:")
print("- Females had higher survival rate.")
print("- 1st class passengers were more likely to survive.")
print("- Younger passengers had better chances.")
print("- Higher fares were associated with survival.")
