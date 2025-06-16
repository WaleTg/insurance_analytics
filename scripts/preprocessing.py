# scripts/preprocessing.py

def format_columns(df):
    df['TransactionMonth'] = pd.to_datetime(df['TransactionMonth'], errors='coerce')
    categorical_cols = df.select_dtypes(include=['object']).columns
    df[categorical_cols] = df[categorical_cols].astype('category')
    return df

def check_missing(df):
    return df.isnull().sum().sort_values(ascending=False)
