import pandas as pd

def load_data(file_path):
    df = pd.read_csv(file_path, sep='|')  # since your file is pipe-separated
    return df
