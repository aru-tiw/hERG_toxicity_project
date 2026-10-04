# save as download_missing.py
import pandas as pd
import numpy as np
from chembl_webresource_client.new_client import new_client
import time

activity_api = new_client.activity
molecule_api = new_client.molecule

def pull_activity(chembl_id, label):
    print(f"\n  Pulling {label} ({chembl_id})...")
    try:
        records = activity_api.filter(
            target_chembl_id=chembl_id,
            standard_type__in=['IC50', 'Ki'],
            standard_relation='=',
            assay_type='B'
        ).only([
            'molecule_chembl_id',
            'canonical_smiles',
            'standard_type',
            'standard_value',
            'standard_units',
            'assay_description',
            'document_year'
        ])
        df = pd.DataFrame(records)
    except Exception as e:
        print(f"  Error: {e}")
        return pd.DataFrame()

    if df.empty:
        print(f"  No data found")
        return df

    df = df[
        (df['standard_units'] == 'nM') &
        (df['standard_value'].notna()) &
        (df['canonical_smiles'].notna())
    ].copy()
    df['standard_value'] = pd.to_numeric(df['standard_value'], errors='coerce')
    df = df[df['standard_value'] > 0].copy()
    df['pIC50']  = -np.log10(df['standard_value'] * 1e-9)
    df['target'] = label
    df = df[(df['pIC50'] > 3) & (df['pIC50'] < 12)]
    return df.reset_index(drop=True)

# ── Download 5-HT2B and CYP3A4 ───────────────────────────
missing = {
    '5-HT2B' : 'CHEMBL1833',
    'CYP3A4'  : 'CHEMBL340',
}

for target_name, chembl_id in missing.items():
    df = pull_activity(chembl_id, target_name)
    if not df.empty:
        fname = f"{target_name.replace('/', '_')}_human.csv"
        df.to_csv(fname, index=False)
        pct_high = (df['pIC50'] > 6).mean() * 100
        print(f"  Saved {len(df):,} records ({df['canonical_smiles'].nunique():,} "
              f"unique compounds) to {fname}")
        print(f"  Mean pIC50: {df['pIC50'].mean():.2f} | "
              f"High concern (>6): {pct_high:.1f}%")
    time.sleep(1)

# ── Fix withdrawn drugs download ──────────────────────────
print("\n\nDownloading withdrawn drugs...")
try:
    records = molecule_api.filter(withdrawn_flag=True)
    withdrawn = pd.DataFrame(records)
    
    # Print available columns so we know what the API returns
    print(f"Available columns: {withdrawn.columns.tolist()}")
    print(f"Total withdrawn drugs: {len(withdrawn)}")
    
    # Save everything - we'll filter columns after seeing what's available
    withdrawn.to_csv('withdrawn_drugs.csv', index=False)
    print("Saved to withdrawn_drugs.csv")

    # Show a sample row to understand the structure
    print("\nSample record:")
    print(withdrawn.iloc[0].to_dict())

except Exception as e:
    print(f"Error: {e}")