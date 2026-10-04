# save as train_compare.py
import matplotlib
matplotlib.use('Agg')
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
from rdkit import RDLogger
from autogluon.tabular import TabularPredictor
from scipy.stats import pointbiserialr
import matplotlib.pyplot as plt
import pickle
import warnings
warnings.filterwarnings('ignore')
RDLogger.DisableLog('rdApp.*')

# ── Fingerprint generator (FCFP4) ─────────────────────────
gen = rdFingerprintGenerator.GetMorganGenerator(
    radius=2, fpSize=2048,
    atomInvariantsGenerator=rdFingerprintGenerator.GetMorganFeatureAtomInvGen()
)
ALL_FEATURE_COLS = [f'fp_{i}' for i in range(2048)]

def safe_fp(smi):
    try:
        mol = Chem.MolFromSmiles(str(smi))
        if mol:
            return np.array(gen.GetFingerprint(mol), dtype=np.uint8)
    except:
        pass
    return None

# ── Load ChEMBL training data ─────────────────────────────
print("Loading ChEMBL embedding...")
chembl = pd.read_csv('final_embedding.csv')
chembl = chembl[chembl['canonical_smiles'].notna()].reset_index(drop=True)
print(f"ChEMBL: {len(chembl):,} compounds | "
      f"Toxic: {chembl['toxic'].sum():,} | "
      f"Non-toxic: {(chembl['toxic']==0).sum():,}")

print("Computing FCFP4 fingerprints...")
fps = [safe_fp(s) for s in chembl['canonical_smiles']]
mask = [fp is not None for fp in fps]
X_train = np.vstack([fp for fp in fps if fp is not None]).astype(np.float32)
y_train = chembl['toxic'].values[mask]
chembl_clean = chembl.iloc[mask].reset_index(drop=True)
print(f"Valid fingerprints: {X_train.shape[0]:,}")

# ── Load ClinTox test data ────────────────────────────────
print("\nLoading ClinTox test data...")
clintox = pd.read_csv('clintox.csv')
clintox = clintox[clintox['smiles'].notna()].reset_index(drop=True)

fps_test = [safe_fp(s) for s in clintox['smiles']]
mask_test = [fp is not None for fp in fps_test]
X_test = np.vstack([fp for fp in fps_test if fp is not None]).astype(np.float32)
y_test = clintox['CT_TOX'].values[mask_test]
clintox_clean = clintox.iloc[mask_test].reset_index(drop=True)
print(f"ClinTox test: {X_test.shape[0]:,} compounds | "
      f"CT_TOX=1: {y_test.sum():,}")

# ══════════════════════════════════════════════════════════
# MODEL A — Baseline (all 2048 bits, no emphasis)
# ══════════════════════════════════════════════════════════
print("\n" + "="*60)
print("MODEL A: Baseline (all 2048 bits, no emphasis)")
print("="*60)

train_A = pd.DataFrame(X_train, columns=ALL_FEATURE_COLS)
train_A['toxic'] = y_train

test_A = pd.DataFrame(X_test, columns=ALL_FEATURE_COLS)
test_A['toxic'] = y_test

predictor_A = TabularPredictor(
    label='toxic',
    eval_metric='roc_auc',
    path='model_A_baseline'
).fit(
    train_data=train_A,
    time_limit=300,
    presets='best_quality',
    verbosity=1
)

results_A = predictor_A.evaluate(test_A)
print(f"Model A — ClinTox AUROC: {results_A['roc_auc']:.4f}")

# ══════════════════════════════════════════════════════════
# MODEL B — Feature-selected (top bits correlated with toxicity)
# This uses the chemical structure insight from UMAP:
# which fingerprint features are most associated with toxicity
# ══════════════════════════════════════════════════════════
print("\n" + "="*60)
print("MODEL B: Feature-selected (chemistry-informed bits only)")
print("="*60)

# Compute point-biserial correlation of each bit with toxicity label
print("Computing bit-toxicity correlations...")
correlations = []
for i in range(2048):
    bit_vals = X_train[:, i]
    if bit_vals.std() > 0:  # skip constant bits
        corr, _ = pointbiserialr(y_train, bit_vals)
        correlations.append((i, abs(corr)))
    else:
        correlations.append((i, 0.0))

corr_df = pd.DataFrame(correlations, columns=['bit', 'correlation'])
corr_df = corr_df.sort_values('correlation', ascending=False)

# Save correlation profile
corr_df.to_csv('bit_toxicity_correlations.csv', index=False)

# Select top-K most correlated bits
# Try K=200 (highly selective) and K=500 (moderate selection)
TOP_K = 300
top_bits = corr_df.head(TOP_K)['bit'].tolist()
top_cols  = [f'fp_{i}' for i in top_bits]

print(f"Selected top {TOP_K} bits out of 2048")
print(f"Mean correlation of selected bits: "
      f"{corr_df.head(TOP_K)['correlation'].mean():.4f}")
print(f"Mean correlation of all bits:      "
      f"{corr_df['correlation'].mean():.4f}")

train_B = pd.DataFrame(X_train[:, top_bits], columns=top_cols)
train_B['toxic'] = y_train

test_B = pd.DataFrame(X_test[:, top_bits], columns=top_cols)
test_B['toxic'] = y_test

predictor_B = TabularPredictor(
    label='toxic',
    eval_metric='roc_auc',
    path='model_B_feature_selected'
).fit(
    train_data=train_B,
    time_limit=300,
    presets='best_quality',
    verbosity=1
)

results_B = predictor_B.evaluate(test_B)
print(f"Model B — ClinTox AUROC: {results_B['roc_auc']:.4f}")

# ══════════════════════════════════════════════════════════
# MODEL C — Sample-weighted (emphasize toxic UMAP clusters)
# Compounds in high-toxicity UMAP regions get higher weight
# ══════════════════════════════════════════════════════════
print("\n" + "="*60)
print("MODEL C: Sample-weighted (toxic UMAP regions emphasized)")
print("="*60)

# Use tox_score from UMAP analysis as sample weight
# Higher tox_score = compound is in a more toxic chemical region
# Scale weights: non-toxic=1.0, mildly toxic=2.0, highly toxic=5.0
tox_scores = chembl_clean['tox_score'].values[mask]

def score_to_weight(score, label):
    if label == 1:
        # Upweight toxic compounds more if they're in toxic regions
        if score > 5:   return 5.0
        elif score > 3: return 3.0
        else:           return 2.0
    else:
        # Slightly downweight non-toxic compounds in toxic regions
        # (they're informative contrastive examples)
        if score > 3: return 1.5
        else:         return 1.0

weights = np.array([score_to_weight(s, l)
                    for s, l in zip(tox_scores, y_train)])

print(f"Sample weight distribution:")
print(f"  Min: {weights.min():.1f} | Max: {weights.max():.1f} | "
      f"Mean: {weights.mean():.2f}")

train_C = pd.DataFrame(X_train, columns=ALL_FEATURE_COLS)
train_C['toxic']          = y_train
train_C['sample_weights'] = weights

test_C = pd.DataFrame(X_test, columns=ALL_FEATURE_COLS)
test_C['toxic'] = y_test

predictor_C = TabularPredictor(
    label='toxic',
    eval_metric='roc_auc',
    path='model_C_sample_weighted',
    sample_weight='sample_weights'
).fit(
    train_data=train_C,
    time_limit=300,
    presets='best_quality',
    verbosity=1
)

results_C = predictor_C.evaluate(test_C)
print(f"Model C — ClinTox AUROC: {results_C['roc_auc']:.4f}")

# ══════════════════════════════════════════════════════════
# COMPARISON SUMMARY
# ══════════════════════════════════════════════════════════
print("\n" + "="*60)
print("FINAL COMPARISON — ClinTox Test Set AUROC")
print("="*60)

comparison = pd.DataFrame([
    {
        'Model'      : 'A — Baseline',
        'Description': 'All 2048 bits, no chemical emphasis',
        'AUROC'      : results_A['roc_auc'],
        'Features'   : 2048
    },
    {
        'Model'      : 'B — Feature-selected',
        'Description': f'Top {TOP_K} toxicity-correlated bits',
        'AUROC'      : results_B['roc_auc'],
        'Features'   : TOP_K
    },
    {
        'Model'      : 'C — Sample-weighted',
        'Description': 'All bits, UMAP cluster weights applied',
        'AUROC'      : results_C['roc_auc'],
        'Features'   : 2048
    },
])
print(comparison.to_string(index=False))
comparison.to_csv('model_comparison.csv', index=False)

# ── Plot comparison ───────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 5))
colors = ['steelblue', 'darkorange', 'seagreen']
bars = ax.bar(
    comparison['Model'],
    comparison['AUROC'],
    color=colors, width=0.5, edgecolor='black'
)
ax.set_ylim(0.5, 1.0)
ax.axhline(0.5, color='red', linestyle='--', linewidth=1, label='Random (0.5)')
ax.set_ylabel('AUROC on ClinTox', fontsize=12)
ax.set_title('Model Comparison: Baseline vs Chemistry-Informed\n'
             '(Trained on ChEMBL, Tested on ClinTox)', fontsize=12)
ax.legend()

for bar, val in zip(bars, comparison['AUROC']):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
            f'{val:.3f}', ha='center', va='bottom', fontsize=11, fontweight='bold')

plt.tight_layout()
plt.savefig('model_comparison.png', dpi=150, bbox_inches='tight')
plt.close()
print("\nSaved model_comparison.png")

# ── Feature importance for Model A vs B ──────────────────
print("\nComputing feature importance...")
imp_A = predictor_A.feature_importance(test_A, silent=True)
imp_B = predictor_B.feature_importance(test_B, silent=True)
imp_A.to_csv('importance_model_A.csv')
imp_B.to_csv('importance_model_B.csv')
print("Saved feature importance files")
print("\nAll done.")