import matplotlib
matplotlib.use('Agg')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
from rdkit import RDLogger
from sklearn.decomposition import IncrementalPCA
import umap
import pickle
import warnings
warnings.filterwarnings('ignore')

RDLogger.DisableLog('rdApp.*')


CHOSEN_CONFIG = {
    'fp'         : 'fcfp4',
    'metric'     : 'jaccard',
    'n_neighbors': 15,
    'min_dist'   : 0.1,
    'pca_dim'    : 100,
}

# ── Load and merge data (same as search script) ───────
def load_target(fname, target_name):
    df = pd.read_csv(fname)
    df = df[['canonical_smiles', 'pIC50']].copy()
    df = df[df['canonical_smiles'].notna() & df['pIC50'].notna()]
    return df.groupby('canonical_smiles')['pIC50'].median()\
             .reset_index()\
             .rename(columns={'pIC50': f'pIC50_{target_name}'})

print("Loading data...")
TARGET_FILES = {
    'hERG'  : 'hERG_human.csv',
    '5HT2B' : '5-HT2B_human.csv',
    'Nav15' : 'Nav1.5_human.csv',
    'Cav12' : 'Cav1.2_human.csv',
    'CYP3A4': 'CYP3A4_human.csv',
}

dfs = [load_target(f, name) for name, f in TARGET_FILES.items()]
combined = dfs[0]
for df in dfs[1:]:
    combined = combined.merge(df, on='canonical_smiles', how='outer')

withdrawn = pd.read_csv('withdrawn_drugs_clean.csv')
combined['withdrawn'] = combined['canonical_smiles'].isin(
    set(withdrawn['smiles'].dropna())
).astype(int)

WEIGHTS = {'hERG': 3.0, '5HT2B': 2.5, 'Nav15': 2.0, 'Cav12': 1.0, 'CYP3A4': 1.0}

def compute_tox_score(row):
    score = 0
    for target, weight in WEIGHTS.items():
        col = f'pIC50_{target}'
        if col in row and pd.notna(row[col]):
            if row[col] > 6:   score += weight * 1.0
            elif row[col] > 5: score += weight * 0.5
    if row.get('withdrawn', 0) == 1:
        score += 5.0
    return score

combined['tox_score'] = combined.apply(compute_tox_score, axis=1)
combined['toxic']     = (combined['tox_score'] > 2.5).astype(int)

# ── Fingerprints ──────────────────────────────────────
fp_name = CHOSEN_CONFIG['fp']
if fp_name == 'fcfp4':
    gen = rdFingerprintGenerator.GetMorganGenerator(
        radius=2, fpSize=2048,
        atomInvariantsGenerator=rdFingerprintGenerator.GetMorganFeatureAtomInvGen()
    )
elif fp_name == 'ecfp6':
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=3, fpSize=2048)
else:
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)

def safe_fp(smi):
    try:
        mol = Chem.MolFromSmiles(str(smi))
        if mol:
            return np.array(gen.GetFingerprint(mol), dtype=np.uint8)
    except:
        pass
    return None

print(f"Computing {fp_name} fingerprints...")
fps = combined['canonical_smiles'].apply(safe_fp)
mask = fps.notna()
X   = np.vstack(fps[mask].values)
df  = combined[mask].reset_index(drop=True)
print(f"  {len(df):,} valid compounds")

# ── PCA ───────────────────────────────────────────────
pca_dim = CHOSEN_CONFIG['pca_dim']
print(f"PCA {X.shape[1]} → {pca_dim} dims...")
pca = IncrementalPCA(n_components=pca_dim, batch_size=1000)
X_pca = pca.fit_transform(X.astype(float))
print(f"  Explained variance: {pca.explained_variance_ratio_.sum():.1%}")

# ── UMAP ──────────────────────────────────────────────
print(f"Fitting UMAP (metric={CHOSEN_CONFIG['metric']}, "
      f"n_neighbors={CHOSEN_CONFIG['n_neighbors']}, "
      f"min_dist={CHOSEN_CONFIG['min_dist']})...")
reducer = umap.UMAP(
    n_components=2,
    metric=CHOSEN_CONFIG['metric'],
    n_neighbors=CHOSEN_CONFIG['n_neighbors'],
    min_dist=CHOSEN_CONFIG['min_dist'],
    random_state=42,
    low_memory=True
)
embedding = reducer.fit_transform(X_pca)

df['UMAP_1'] = embedding[:, 0]
df['UMAP_2'] = embedding[:, 1]

# Save reducer for projecting ClinTox later
with open('umap_reducer.pkl', 'wb') as f:
    pickle.dump({'reducer': reducer, 'pca': pca, 'fp': fp_name}, f)
print("Saved UMAP reducer to umap_reducer.pkl")

# Save full embedding
df.to_csv('final_embedding.csv', index=False)
print("Saved embedding to final_embedding.csv")

# ── Plots ─────────────────────────────────────────────
TARGETS = ['hERG', '5HT2B', 'Nav15', 'Cav12', 'CYP3A4']
n_plots  = len(TARGETS) + 2  # one per target + overall tox score + binary
ncols    = 3
nrows    = (n_plots + ncols - 1) // ncols

fig, axes = plt.subplots(nrows, ncols, figsize=(18, nrows * 5))
axes = axes.flatten()

# Plot 1: overall toxicity score
sc = axes[0].scatter(
    df['UMAP_1'], df['UMAP_2'],
    c=df['tox_score'], cmap='RdYlGn_r',
    s=1, alpha=0.5, vmin=0, vmax=8
)
plt.colorbar(sc, ax=axes[0], label='Toxicity Score')
axes[0].set_title('Overall Toxicity Score', fontsize=11)

# Plot 2: binary + withdrawn
colors = np.where(df['withdrawn'] == 1, 'red',
         np.where(df['toxic'] == 1, 'orange', 'steelblue'))
axes[1].scatter(df['UMAP_1'], df['UMAP_2'], c=colors, s=1, alpha=0.4)
legend_els = [
    Line2D([0],[0], marker='o', color='w', markerfacecolor='red',      markersize=8, label=f"Withdrawn ({df['withdrawn'].sum()})"),
    Line2D([0],[0], marker='o', color='w', markerfacecolor='orange',    markersize=8, label=f"Toxic ({df['toxic'].sum()})"),
    Line2D([0],[0], marker='o', color='w', markerfacecolor='steelblue', markersize=8, label=f"Non-toxic ({(df['toxic']==0).sum()})"),
]
axes[1].legend(handles=legend_els, loc='upper right', fontsize=8)
axes[1].set_title('Toxic vs Non-toxic', fontsize=11)

# Plots 3-7: one per target
for j, target in enumerate(TARGETS):
    ax  = axes[j + 2]
    col = f'pIC50_{target}'
    has_data = df[col].notna()

    # Grey background for compounds not tested
    ax.scatter(
        df.loc[~has_data, 'UMAP_1'],
        df.loc[~has_data, 'UMAP_2'],
        c='lightgrey', s=1, alpha=0.2, label='Not tested'
    )
    # Colored for tested compounds
    sc = ax.scatter(
        df.loc[has_data, 'UMAP_1'],
        df.loc[has_data, 'UMAP_2'],
        c=df.loc[has_data, col],
        cmap='RdYlGn_r', s=2, alpha=0.7,
        vmin=4, vmax=9
    )
    plt.colorbar(sc, ax=ax, label='pIC50')
    ax.set_title(f'{target} pIC50\n({has_data.sum():,} compounds tested)', fontsize=11)

# Hide unused axes
for k in range(n_plots, len(axes)):
    axes[k].set_visible(False)

for ax in axes[:n_plots]:
    ax.set_xlabel('UMAP_1', fontsize=8)
    ax.set_ylabel('UMAP_2', fontsize=8)
    ax.tick_params(labelsize=7)

cfg = CHOSEN_CONFIG
fig.suptitle(
    f'Cardiac Toxicity Chemical Space — {fp_name} | metric={cfg["metric"]} | '
    f'n_neighbors={cfg["n_neighbors"]} | min_dist={cfg["min_dist"]} | pca={cfg["pca_dim"]}',
    fontsize=13, y=1.01
)
plt.tight_layout()
plt.savefig('final_umap.png', dpi=200, bbox_inches='tight')
plt.close()
print("Saved final_umap.png")
print("\nDone. Next step: run supervised ML using final_embedding.csv")