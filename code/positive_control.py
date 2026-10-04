import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
from sklearn.model_selection import train_test_split
from autogluon.tabular import TabularPredictor
from sklearn.metrics import roc_auc_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ── fingerprint helper ─────────────────────────────────────────────────────
gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)

def smiles_to_fp(smi):
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None
    arr = np.zeros(2048, dtype=np.uint8)
    gen.GetFingerprintAsNumPy(mol, arr)
    return arr

# ── load Tox21 + ClinTox ───────────────────────────────────────────────────
print("Loading datasets...")
clintox = pd.read_csv('clintox.csv')
tox21   = pd.read_csv('tox21.csv')

records = []
for df, col in [(clintox, 'CT_TOX'), (tox21, 'SR-ARE')]:
    smiles_col = 'smiles' if 'smiles' in df.columns else df.columns[0]
    for _, row in df.iterrows():
        smi = str(row[smiles_col])
        label = row.get(col, np.nan)
        if pd.isna(label):
            continue
        fp = smiles_to_fp(smi)
        if fp is not None:
            records.append({'smiles': smi, 'label': int(label), 'fp': fp})

combined = pd.DataFrame(records).drop_duplicates('smiles')
print(f"Combined: {len(combined)} compounds | Toxic: {combined['label'].sum()} | Non-toxic: {(combined['label']==0).sum()}")

# ── build feature matrix ───────────────────────────────────────────────────
X = np.vstack(combined['fp'].values)
y = combined['label'].values
cols = [f'fp_{i}' for i in range(2048)]
feat_df = pd.DataFrame(X, columns=cols)
feat_df['toxic'] = y

# ── split 60/20/20 ─────────────────────────────────────────────────────────
train_val, test_df = train_test_split(feat_df, test_size=0.2, random_state=42, stratify=y)
train_df, val_df   = train_test_split(train_val, test_size=0.25, random_state=42,
                                       stratify=train_val['toxic'])

print(f"Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")

# ── train ──────────────────────────────────────────────────────────────────
print("\nTraining positive control model (Tox21+ClinTox → ClinTox)...")
predictor = TabularPredictor(
    label='toxic',
    eval_metric='roc_auc',
    path='model_D_positive_control'
).fit(
    train_data=train_df,
    tuning_data=val_df,
    time_limit=1800,
    presets='best_quality'
)

# ── leaderboard ────────────────────────────────────────────────────────────
print("\n=== Model D Leaderboard ===")
lb = predictor.leaderboard(silent=True)
print(lb[['model','score_val']].to_string())

# ── evaluate on held-out test ──────────────────────────────────────────────
test_X = test_df.drop(columns=['toxic'])
test_y = test_df['toxic'].values
probs  = predictor.predict_proba(test_X)[1].values
auroc  = roc_auc_score(test_y, probs)
print(f"\nModel D — In-distribution Test AUROC: {auroc:.4f}")

# ── combined comparison plot ───────────────────────────────────────────────
results = {
    'A — ChEMBL baseline\n(2048 bits → ClinTox transfer)':      0.5171,
    'B — ChEMBL feature-selected\n(300 bits → ClinTox transfer)': 0.4826,
    'C — ChEMBL UMAP-weighted\n(→ ClinTox transfer)':            0.4988,
    'D — Tox21+ClinTox\n(in-distribution)':                       auroc,
}

fig, ax = plt.subplots(figsize=(9, 5))
colors = ['#d62728', '#d62728', '#d62728', '#2ca02c']
bars = ax.barh(list(results.keys()), list(results.values()), color=colors, edgecolor='black', linewidth=0.5)
ax.axvline(0.5,  color='gray', linestyle='--', linewidth=1, label='Random (0.50)')
ax.axvline(0.91, color='steelblue', linestyle=':', linewidth=1.5, label='ChEMBL CV score (0.91)')
ax.set_xlabel('AUROC on ClinTox Test Set', fontsize=12)
ax.set_title('Distribution Shift: ChEMBL Bioactivity → Clinical Toxicity', fontsize=13)
ax.set_xlim(0.3, 1.0)
for bar, val in zip(bars, results.values()):
    ax.text(val + 0.005, bar.get_y() + bar.get_height()/2,
            f'{val:.3f}', va='center', fontsize=10)
ax.legend(loc='lower right')
plt.tight_layout()
plt.savefig('distribution_shift_comparison.png', dpi=150, bbox_inches='tight')
print("Saved distribution_shift_comparison.png")