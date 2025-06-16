# scripts/task_3_hypothesis_testing.py

import pandas as pd
from scipy.stats import chi2_contingency, ttest_ind

DATA_PATH = "data/MachineLearningRating_v3.txt"  # pipe‑separated input

def load_data():
    """Load the pipe‑separated insurance dataset."""
    return pd.read_csv(DATA_PATH, sep='|', low_memory=False)

def prepare_kpis(df):
    """Add KPI columns: HasClaim, ClaimSeverity, Margin."""
    # HasClaim: True if TotalClaims > 0
    df['HasClaim'] = df['TotalClaims'] > 0
    
    # ClaimSeverity: only for rows with claims
    df['ClaimSeverity'] = df.loc[df['HasClaim'], 'TotalClaims']
    
    # Margin: TotalPremium minus TotalClaims
    df['Margin'] = df['TotalPremium'] - df['TotalClaims']
    return df

def chi2_test_frequency(df, group_col):
    """
    Chi‑square test on claim frequency (HasClaim) across categories of group_col.
    Returns p‑value.
    """
    ct = pd.crosstab(df[group_col], df['HasClaim'])
    _, p, _, _ = chi2_contingency(ct)
    return p

def ttest_metric(df, group_col, group1, group2, metric):
    """
    Welch’s t‑test comparing `metric` between two groups in `group_col`.
    Returns p‑value.
    """
    a = df.loc[df[group_col] == group1, metric].dropna()
    b = df.loc[df[group_col] == group2, metric].dropna()
    _, p = ttest_ind(a, b, equal_var=False)
    return p

def main():
    df = load_data()
    df = prepare_kpis(df)

    # 1. H₀: No claim‐frequency differences across provinces
    p1 = chi2_test_frequency(df, 'Province')
    print(f"Claim Frequency by Province: p = {p1:.4f}")

    # 2. H₀: No claim‐frequency differences across postal codes
    p2 = chi2_test_frequency(df, 'PostalCode')
    print(f"Claim Frequency by PostalCode: p = {p2:.4f}")

    # 3. H₀: No margin differences across postal codes
    #    Compare two example postal codes (replace with specific codes as needed)
    codes = df['PostalCode'].dropna().unique()
    if len(codes) >= 2:
        pc1, pc2 = codes[0], codes[1]
        p3 = ttest_metric(df, 'PostalCode', pc1, pc2, 'Margin')
        print(f"Margin difference {pc1} vs {pc2}: p = {p3:.4f}")
    else:
        print("Not enough postal codes to test margin difference.")

    # 4. H₀: No claim‐frequency differences between women and men
    p4 = chi2_test_frequency(df, 'Gender')
    print(f"Claim Frequency by Gender: p = {p4:.4f}")

    # Business interpretations
    print("\nInterpretations:")
    for name, p in [
        ("Province freq", p1),
        ("PostalCode freq", p2),
        ("PostalCode margin", p3 if 'p3' in locals() else None),
        ("Gender freq", p4),
    ]:
        if p is not None:
            decision = "Reject H₀" if p < 0.05 else "Fail to reject H₀"
            print(f"  - {name}: {decision} (p={p:.4f})")

if __name__ == "__main__":
    main()
