import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
from autogluon.tabular import TabularPredictor
from sklearn.metrics import roc_auc_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ── FCFP4 fingerprint (must match what train_compare.py used) ──────────────
gen = rdFingerprintGenerator.GetMorganGenerator(
    radius=2, fpSize=2048,
    atomInvariantsGenerator=rdFingerprintGenerator.GetMorganFeatureAtomInvGen()
)

def smiles_to_fp(smi):
    try:
        mol = Chem.MolFromSmiles(str(smi))
        if mol:
            return np.array(gen.GetFingerprint(mol), dtype=np.uint8)
    except:
        pass
    return None

# ── Load TDC hERG dataset ──────────────────────────────────────────────────
print("Loading TDC hERG dataset...")
herg = pd.read_csv('herg_dataset.csv')
print(f"Total rows: {len(herg)}")
print(herg['Y'].value_counts())

# Compute fingerprints
fps, labels, ids = [], [], []
for _, row in herg.iterrows():
    fp = smiles_to_fp(row['Drug'])
    if fp is not None and not pd.isna(row['Y']):
        fps.append(fp)
        labels.append(int(row['Y']))
        ids.append(row['Drug_ID'])

X = np.array(fps)
y = np.array(labels)
cols = [f'fp_{i}' for i in range(2048)]
test_df = pd.DataFrame(X, columns=cols)

print(f"\nValid compounds: {len(y)} | hERG blockers (Y=1): {y.sum()} | Non-blockers (Y=0): {(y==0).sum()}")

# ── Evaluate each model ────────────────────────────────────────────────────
models = {
    'A — Baseline (2048 bits)':       'model_A_baseline',
    'B — Feature-selected (300 bits)': 'model_B_feature_selected',
    'C — UMAP-weighted':               'model_C_sample_weighted',
}

results = {}
for name, path in models.items():
    try:
        predictor = TabularPredictor.load(path)
        probs = predictor.predict_proba(test_df)[1].values
        auroc = roc_auc_score(y, probs)
        results[name] = auroc
        print(f"{name}: AUROC = {auroc:.4f}")
    except Exception as e:
        print(f"{name}: FAILED — {e}")

# ── Bar plot ───────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 4))
colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
bars = ax.barh(list(results.keys()), list(results.values()),
               color=colors, edgecolor='black', linewidth=0.5)
ax.axvline(0.5, color='gray', linestyle='--', linewidth=1, label='Random (0.50)')
ax.set_xlabel('AUROC on TDC hERG Test Set', fontsize=12)
ax.set_title('External Validation: ChEMBL-trained models → TDC hERG', fontsize=12)
ax.set_xlim(0.3, 1.0)
for bar, val in zip(bars, results.values()):
    ax.text(val + 0.005, bar.get_y() + bar.get_height()/2,
            f'{val:.3f}', va='center', fontsize=10)
ax.legend()
plt.tight_layout()
plt.savefig('herg_tdc_evaluation.png', dpi=150, bbox_inches='tight')
print("\nSaved herg_tdc_evaluation.png")
print("\n=== SUMMARY ===")
for name, auroc in results.items():
    print(f"  {name}: {auroc:.4f}")