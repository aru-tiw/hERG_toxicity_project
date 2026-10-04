import matplotlib
matplotlib.use('Agg')  # no display needed on cluster

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
from rdkit import RDLogger
from sklearn.decomposition import IncrementalPCA
import umap
import os
import warnings
warnings.filterwarnings('ignore')

RDLogger.DisableLog('rdApp.*')

# ── Fingerprint generators ────────────────────────────────
FP_GENERATORS = {
    'ecfp4': rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048),
    'ecfp6': rdFingerprintGenerator.GetMorganGenerator(radius=3, fpSize=2048),
    'fcfp4': rdFingerprintGenerator.GetMorganGenerator(
                 radius=2, fpSize=2048,
                 atomInvariantsGenerator=rdFingerprintGenerator.GetMorganFeatureAtomInvGen()
             ),
}

# ── Hyperparameter grid to search ─────────────────────────
CONFIGS = [
    {'fp': 'ecfp4', 'metric': 'jaccard',   'n_neighbors': 15,  'min_dist': 0.1,  'pca_dim': 100},
    {'fp': 'ecfp4', 'metric': 'jaccard',   'n_neighbors': 30,  'min_dist': 0.1,  'pca_dim': 100},
    {'fp': 'ecfp4', 'metric': 'jaccard',   'n_neighbors': 15,  'min_dist': 0.5,  'pca_dim': 100},
    {'fp': 'ecfp4', 'metric': 'euclidean', 'n_neighbors': 15,  'min_dist': 0.1,  'pca_dim': 100},
    {'fp': 'ecfp6', 'metric': 'jaccard',   'n_neighbors': 15,  'min_dist': 0.1,  'pca_dim': 100},
    {'fp': 'fcfp4', 'metric': 'jaccard',   'n_neighbors': 15,  'min_dist': 0.1,  'pca_dim': 100},
    {'fp': 'fcfp4', 'metric': 'jaccard',   'n_neighbors': 30,  'min_dist': 0.1,  'pca_dim': 50 },
    {'fp': 'ecfp4', 'metric': 'jaccard',   'n_neighbors': 15,  'min_dist': 0.1,  'pca_dim': 50 },
]

# ── Load and merge all ChEMBL targets ────────────────────
def load_target(fname, target_name):
    df = pd.read_csv(fname)
    df = df[['canonical_smiles', 'pIC50']].copy()
    df = df[df['canonical_smiles'].notna() & df['pIC50'].notna()]
    # Median pIC50 per compound
    return df.groupby('canonical_smiles')['pIC50'].median()\
             .reset_index()\
             .rename(columns={'pIC50': f'pIC50_{target_name}'})

print("Loading target data...")
TARGET_FILES = {
    'hERG'  : 'hERG_human.csv',
    '5HT2B' : '5-HT2B_human.csv',
    'Nav15' : 'Nav1.5_human.csv',
    'Cav12' : 'Cav1.2_human.csv',
    'CYP3A4': 'CYP3A4_human.csv',
}

dfs = [load_target(f, name) for name, f in TARGET_FILES.items()]

# Outer merge — keeps all compounds even if only tested on one target
combined = dfs[0]
for df in dfs[1:]:
    combined = combined.merge(df, on='canonical_smiles', how='outer')

# Add withdrawn drugs
withdrawn = pd.read_csv('withdrawn_drugs_clean.csv')
withdrawn_smiles = set(withdrawn['smiles'].dropna())
combined['withdrawn'] = combined['canonical_smiles'].isin(withdrawn_smiles).astype(int)

# ── Toxicity score ────────────────────────────────────────
WEIGHTS = {'hERG': 3.0, '5HT2B': 2.5, 'Nav15': 2.0, 'Cav12': 1.0, 'CYP3A4': 1.0}

def compute_tox_score(row):
    score = 0
    for target, weight in WEIGHTS.items():
        col = f'pIC50_{target}'
        if col in row and pd.notna(row[col]):
            if row[col] > 6:
                score += weight * 1.0
            elif row[col] > 5:
                score += weight * 0.5
    if row.get('withdrawn', 0) == 1:
        score += 5.0
    return score

combined['tox_score'] = combined.apply(compute_tox_score, axis=1)
combined['toxic']     = (combined['tox_score'] > 2.5).astype(int)

print(f"Total compounds: {len(combined):,}")
print(f"Toxic (high concern): {combined['toxic'].sum():,}")
print(f"Non-toxic:            {(combined['toxic']==0).sum():,}")

# ── Fingerprints ──────────────────────────────────────────
def safe_fp(smi, gen):
    try:
        mol = Chem.MolFromSmiles(str(smi))
        if mol:
            return np.array(gen.GetFingerprint(mol), dtype=np.uint8)
    except:
        pass
    return None

# Pre-compute all fingerprint types
print("\nComputing fingerprints...")
fp_cache = {}
for fp_name, gen in FP_GENERATORS.items():
    fps = combined['canonical_smiles'].apply(lambda s: safe_fp(s, gen))
    mask = fps.notna()
    fp_cache[fp_name] = {
        'fps'  : np.vstack(fps[mask].values),
        'mask' : mask,
        'df'   : combined[mask].reset_index(drop=True)
    }
    print(f"  {fp_name}: {mask.sum():,} valid compounds")

# ── Run hyperparameter search ─────────────────────────────
os.makedirs('umap_search', exist_ok=True)
results = []

print(f"\nRunning {len(CONFIGS)} UMAP configurations...")

for i, cfg in enumerate(CONFIGS):
    fp_name   = cfg['fp']
    metric    = cfg['metric']
    n_neigh   = cfg['n_neighbors']
    min_dist  = cfg['min_dist']
    pca_dim   = cfg['pca_dim']

    label = f"cfg{i+1:02d}_{fp_name}_m{metric[:3]}_n{n_neigh}_d{min_dist}_pca{pca_dim}"
    print(f"\n[{i+1}/{len(CONFIGS)}] {label}")

    X   = fp_cache[fp_name]['fps']
    df  = fp_cache[fp_name]['df']

    # PCA pre-processing (as per lab assignment)
    print(f"  PCA {X.shape[1]} → {pca_dim} dims...")
    pca = IncrementalPCA(n_components=pca_dim, batch_size=1000)
    X_pca = pca.fit_transform(X.astype(float))

    # UMAP
    print(f"  UMAP: metric={metric}, n_neighbors={n_neigh}, min_dist={min_dist}...")
    reducer = umap.UMAP(
        n_components=2,
        metric=metric,
        n_neighbors=n_neigh,
        min_dist=min_dist,
        random_state=42,
        low_memory=True
    )
    embedding = reducer.fit_transform(X_pca)

    # Save embedding
    emb_df = df[['canonical_smiles', 'tox_score', 'toxic', 'withdrawn']].copy()
    emb_df['UMAP_1'] = embedding[:, 0]
    emb_df['UMAP_2'] = embedding[:, 1]
    emb_df.to_csv(f'umap_search/{label}_embedding.csv', index=False)

    # Plot — colored by toxicity score
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle(f'Config {i+1}: {fp_name} | metric={metric} | '
                 f'n_neighbors={n_neigh} | min_dist={min_dist} | pca={pca_dim}',
                 fontsize=12)

    # Left: continuous toxicity score
    sc = axes[0].scatter(
        embedding[:, 0], embedding[:, 1],
        c=df['tox_score'], cmap='RdYlGn_r',
        s=1, alpha=0.5
    )
    plt.colorbar(sc, ax=axes[0], label='Toxicity Score')
    axes[0].set_title('Colored by Toxicity Score')
    axes[0].set_xlabel('UMAP_1')
    axes[0].set_ylabel('UMAP_2')

    # Right: binary toxic/non-toxic + withdrawn highlighted
    colors = np.where(df['withdrawn'] == 1, 'red',
             np.where(df['toxic'] == 1, 'orange', 'steelblue'))
    axes[1].scatter(embedding[:, 0], embedding[:, 1],
                    c=colors, s=1, alpha=0.4)
    # Legend
    from matplotlib.lines import Line2D
    legend = [
        Line2D([0],[0], marker='o', color='w', markerfacecolor='red',    markersize=8, label='Withdrawn'),
        Line2D([0],[0], marker='o', color='w', markerfacecolor='orange',  markersize=8, label='Toxic'),
        Line2D([0],[0], marker='o', color='w', markerfacecolor='steelblue',markersize=8, label='Non-toxic'),
    ]
    axes[1].legend(handles=legend, loc='upper right')
    axes[1].set_title('Toxic vs Non-toxic')
    axes[1].set_xlabel('UMAP_1')
    axes[1].set_ylabel('UMAP_2')

    plt.tight_layout()
    plt.savefig(f'umap_search/{label}.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved plot: umap_search/{label}.png")

    results.append({**cfg, 'label': label, 'n_compounds': len(df)})

# Save summary
pd.DataFrame(results).to_csv('umap_search/search_summary.csv', index=False)
print(f"\nDone. {len(CONFIGS)} embeddings saved to umap_search/")
print("Download the PNG files to compare visually, then edit CHOSEN_CONFIG in generate_final_umap.py")