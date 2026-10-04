import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
from rdkit import RDLogger
import warnings
warnings.filterwarnings('ignore')
RDLogger.DisableLog('rdApp.*')

gen = rdFingerprintGenerator.GetMorganGenerator(
    radius=2, fpSize=2048,
    atomInvariantsGenerator=rdFingerprintGenerator.GetMorganFeatureAtomInvGen()
)

def safe_fp(smi):
    try:
        mol = Chem.MolFromSmiles(str(smi))
        if mol:
            return np.array(gen.GetFingerprint(mol), dtype=np.uint8)
    except:
        pass
    return None

# ── Load ChEMBL hERG ───────────────────────────────────────────────────────
print("Loading ChEMBL hERG data...")
herg = pd.read_csv('hERG_human.csv')
print(f"Raw records: {len(herg)}")

# Deduplicate by SMILES — take mean pIC50 per compound
herg = herg.dropna(subset=['canonical_smiles', 'pIC50'])
herg = herg.groupby('canonical_smiles', as_index=False)['pIC50'].mean()
print(f"After dedup: {len(herg)} unique compounds")

# Binary label: pIC50 >= 6 → hERG blocker (IC50 < 1µM)
THRESHOLD = 6.0
herg['toxic'] = (herg['pIC50'] >= THRESHOLD).astype(int)
print(f"hERG blockers (pIC50 >= {THRESHOLD}): {herg['toxic'].sum()}")
print(f"Non-blockers:                           {(herg['toxic']==0).sum()}")

# ── Compute fingerprints ───────────────────────────────────────────────────
print("\nComputing FCFP4 fingerprints...")
fps = [safe_fp(s) for s in herg['canonical_smiles']]
mask = [fp is not None for fp in fps]
herg_clean = herg[mask].reset_index(drop=True)
X = np.vstack([fp for fp in fps if fp is not None])

print(f"Valid fingerprints: {len(herg_clean)}")

np.save('herg_X.npy', X)
herg_clean.to_csv('herg_clean.csv', index=False)
print("Saved herg_X.npy and herg_clean.csv")