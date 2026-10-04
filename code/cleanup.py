import pandas as pd
import numpy as np
from rdkit import Chem, RDLogger
RDLogger.DisableLog('rdApp.*')

def canonical(smi):
    try:
        mol = Chem.MolFromSmiles(str(smi))
        return Chem.MolToSmiles(mol) if mol else None
    except:
        return None

# ── Load both datasets ─────────────────────────────────────────────────────
print("Loading datasets...")
chembl = pd.read_csv('herg_clean.csv')
tdc    = pd.read_csv('herg_dataset.csv').dropna(subset=['Drug', 'Y'])

# ── Canonicalize SMILES ────────────────────────────────────────────────────
print("Canonicalizing SMILES...")
chembl['canon'] = chembl['canonical_smiles'].apply(canonical)
tdc['canon']    = tdc['Drug'].apply(canonical)

chembl_valid = chembl.dropna(subset=['canon'])
tdc_valid    = tdc.dropna(subset=['canon'])

chembl_set = set(chembl_valid['canon'])
tdc_set    = set(tdc_valid['canon'])

overlap = chembl_set & tdc_set
print(f"\nChEMBL compounds:   {len(chembl_set)}")
print(f"TDC compounds:      {len(tdc_set)}")
print(f"Overlapping:        {len(overlap)}")
print(f"Overlap as % TDC:   {len(overlap)/len(tdc_set)*100:.1f}%")

if len(overlap) > 0:
    print("\nSample overlapping SMILES:")
    for smi in list(overlap)[:5]:
        print(f"  {smi}")

    # Show label agreement for overlapping compounds
    print("\nChecking label consistency for overlapping compounds...")
    chembl_labels = chembl_valid[chembl_valid['canon'].isin(overlap)][['canon','toxic']]
    tdc_labels    = tdc_valid[tdc_valid['canon'].isin(overlap)][['canon','Y']]
    merged = chembl_labels.merge(tdc_labels, on='canon')
    merged['agree'] = merged['toxic'] == merged['Y']
    print(f"Label agreement: {merged['agree'].sum()}/{len(merged)} overlapping compounds")
    print(merged[['canon','toxic','Y','agree']].to_string())

# ── Save deduplicated TDC (overlaps removed) ───────────────────────────────
tdc_clean = tdc_valid[~tdc_valid['canon'].isin(overlap)].copy()
tdc_clean.to_csv('herg_tdc_deduped.csv', index=False)

print(f"\nOriginal TDC:  {len(tdc_valid)} compounds")
print(f"After removal: {len(tdc_clean)} compounds")
print(f"Saved herg_tdc_deduped.csv")