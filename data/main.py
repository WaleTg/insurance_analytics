# main.py

from scripts.data_loader import load_data
from scripts.preprocessing import format_columns, check_missing
from scripts.eda import plot_distribution, plot_loss_ratio_by_province

# Load data from full path
df = load_data()

# Preprocess
df = format_columns(df)

# Missing value summary
missing = check_missing(df)
print("Missing values:\n", missing)

# EDA visualizations
plot_distribution(df, "TotalClaims")
plot_distribution(df, "TotalPremium")
plot_loss_ratio_by_province(df)
