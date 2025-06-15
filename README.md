# insurance_analytics
Predictive analytics and EDA on insurance claims data for AlphaCare Insurance Solutions.
Insurance Analytics - Task 1: Exploratory Data Analysis (EDA)

This repository contains the code and notebooks for the initial exploratory data analysis (EDA) of the car insurance claim dataset provided by AlphaCare Insurance Solutions. The goal of Task 1 is to understand the dataset’s structure, distributions, and correlations to guide further modeling and analysis.

Project Structure

insurance_analytics/
│
├── data/
│ └── MachineLearningRating_v3.txt # Dataset (pipe-separated)
│
├── scripts/
│ ├── data_loader.py # Functions to load and preprocess data
│ └── eda.py # EDA helper functions (summary, plots)
│
└── notebooks/
└── eda.ipynb # Jupyter notebook for interactive analysis

Getting Started

Prerequisites

Python 3.8+

Libraries: pandas, matplotlib, seaborn, jupyter

Install dependencies via pip:

pip install pandas matplotlib seaborn jupyter

How to run

Clone this repository.

Place the dataset MachineLearningRating_v3.txt inside the data/ folder.

Launch the Jupyter notebook:

jupyter notebook notebooks/eda.ipynb

Run cells step-by-step to load data, perform summary statistics, visualize distributions, and examine feature correlations.

Overview of Task 1

Load pipe-separated dataset with custom load_data() function.

Perform basic EDA including:

Data preview and summary statistics

Missing value analysis

Distribution plots of numeric features

Correlation matrix visualization

This helps understand the data quality and relationships before applying machine learning models in subsequent tasks.
