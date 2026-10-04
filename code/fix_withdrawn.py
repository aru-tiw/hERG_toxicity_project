import pandas as pd

withdrawn = pd.read_csv('withdrawn_drugs.csv')

# SMILES is nested inside molecule_structures dict
import ast

def extract_smiles(mol_struct):
    try:
        if isinstance(mol_struct, str):
            mol_struct = ast.literal_eval(mol_struct)
        if isinstance(mol_struct, dict):
            return mol_struct.get('canonical_smiles', None)
    except:
        pass
    return None

withdrawn['smiles'] = withdrawn['molecule_structures'].apply(extract_smiles)
withdrawn['drug_name'] = withdrawn['pref_name']

# Keep only what we need
withdrawn_clean = withdrawn[['molecule_chembl_id', 'drug_name', 'smiles']].dropna(subset=['smiles'])
withdrawn_clean['withdrawn'] = 1

print(f"Withdrawn drugs with SMILES: {len(withdrawn_clean)}")
print(withdrawn_clean.head())
withdrawn_clean.to_csv('withdrawn_drugs_clean.csv', index=False)