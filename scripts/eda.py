import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def run_basic_eda(df):
    print("First 5 rows:")
    print(df.head())
    print("\nSummary statistics:")
    print(df.describe())
    print("\nMissing values per column:")
    print(df.isnull().sum())

def plot_distributions(df, num_cols):
    for col in num_cols:
        plt.figure(figsize=(8, 4))
        sns.histplot(df[col], kde=True)
        plt.title(f"Distribution of {col}")
        plt.show()

def plot_correlation_matrix(df, num_cols):
    plt.figure(figsize=(10, 8))
    corr = df[num_cols].corr()
    sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Correlation Matrix")
    plt.show()
