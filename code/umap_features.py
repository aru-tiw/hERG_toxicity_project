import numpy as np
import pandas as pd
import pickle
import sklearn.decomposition
import umap
import hdbscan
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from rdkit import Chem, RDLogger
from rdkit.Chem import Draw
from rdkit.Chem.Scaffolds import MurckoScaffold
from scipy.stats import pointbiserialr
import warnings
warnings.filterwarnings('ignore')
RDLogger.DisableLog('rdApp.*')

# ── Load data ──────────────────────────────────────────────────────────────
print("Loading data...")
X    = np.load('herg_X.npy').astype(np.float32)
meta = pd.read_csv('herg_clean.csv')
y    = meta['toxic'].values
pic50 = meta['pIC50'].values
n = len(X)
print(f"Compounds: {n} | Blockers: {y.sum()} | Non-blockers: {(y==0).sum()}")

# ── IncrementalPCA ─────────────────────────────────────────────────────────
print("Fitting IncrementalPCA (100 components)...")
pca = sklearn.decomposition.IncrementalPCA(n_components=100)
batch = 1000
for i in range(0, n, batch):
    pca.partial_fit(X[i:i+batch])
pca_X = np.vstack([pca.transform(X[i:i+batch]) for i in range(0, n, batch)])
print(f"PCA shape: {pca_X.shape}")

# ── UMAP ───────────────────────────────────────────────────────────────────
print("Fitting UMAP...")
reducer = umap.UMAP(n_neighbors=15, min_dist=0.1,
                    metric='euclidean', random_state=42, n_jobs=4)
emb = reducer.fit_transform(pca_X)
print(f"UMAP embedding shape: {emb.shape}")

# Save reducers
with open('herg_pca.pkl', 'wb') as f: pickle.dump(pca, f)
with open('herg_umap.pkl', 'wb') as f: pickle.dump(reducer, f)

# ── UMAP plots (pIC50 + binary) ────────────────────────────────────────────
print("Plotting UMAP...")
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

sc = axes[0].scatter(emb[:,0], emb[:,1], c=pic50, cmap='RdYlGn_r',
                     s=4, alpha=0.6, vmin=4, vmax=10)
plt.colorbar(sc, ax=axes[0], label='pIC50')
axes[0].set_title('ChEMBL hERG — pIC50 gradient')
axes[0].set_xlabel('UMAP 1'); axes[0].set_ylabel('UMAP 2')

colors_bin = ['#d62728' if t==1 else '#2ca02c' for t in y]
axes[1].scatter(emb[:,0], emb[:,1], c=colors_bin, s=4, alpha=0.5)
axes[1].set_title('ChEMBL hERG — Blocker vs Non-blocker')
axes[1].set_xlabel('UMAP 1'); axes[1].set_ylabel('UMAP 2')
axes[1].legend(handles=[
    mpatches.Patch(color='#d62728', label=f'hERG blocker (n={y.sum()})'),
    mpatches.Patch(color='#2ca02c', label=f'Non-blocker (n={(y==0).sum()})')
])
plt.suptitle('ChEMBL hERG Chemical Space (FCFP4 → PCA → UMAP)', fontsize=13)
plt.tight_layout()
plt.savefig('herg_umap.png', dpi=150, bbox_inches='tight')
print("Saved herg_umap.png")
plt.close()

# ── HDBSCAN clustering ─────────────────────────────────────────────────────
print("\nClustering UMAP embedding with HDBSCAN...")
clusterer = hdbscan.HDBSCAN(min_cluster_size=30, min_samples=10)
cluster_labels = clusterer.fit_predict(emb)
n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
print(f"Found {n_clusters} clusters | Noise points: {(cluster_labels==-1).sum()}")

# Toxicity rate per cluster
cluster_df = pd.DataFrame({'cluster': cluster_labels, 'toxic': y})
cluster_stats = (cluster_df[cluster_df['cluster'] >= 0]
                 .groupby('cluster')
                 .agg(count=('toxic','count'), n_toxic=('toxic','sum'))
                 .reset_index())
cluster_stats['toxic_rate'] = cluster_stats['n_toxic'] / cluster_stats['count']
cluster_stats.to_csv('herg_cluster_stats.csv', index=False)
print("\nCluster toxicity rates:")
print(cluster_stats.sort_values('toxic_rate', ascending=False).to_string())

# Label clusters with >60% blockers as toxic regions
TOXIC_THRESH = 0.6
toxic_clusters = set(cluster_stats[cluster_stats['toxic_rate'] >= TOXIC_THRESH]['cluster'])
print(f"\nToxic clusters (>{TOXIC_THRESH*100:.0f}% blockers): {toxic_clusters}")

# Assign region label — noise points fall back to their actual label
in_toxic_region = np.array([
    1 if cl in toxic_clusters else (0 if cl >= 0 else int(t))
    for cl, t in zip(cluster_labels, y)
])
print(f"Compounds in toxic UMAP regions: {in_toxic_region.sum()}")

# ── Cluster UMAP plot ──────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

axes[0].scatter(emb[:,0], emb[:,1], c=cluster_labels, cmap='tab20',
                s=4, alpha=0.6)
axes[0].set_title(f'HDBSCAN Clusters (n={n_clusters})')
axes[0].set_xlabel('UMAP 1'); axes[0].set_ylabel('UMAP 2')

region_colors = ['#d62728' if r==1 else '#2ca02c' for r in in_toxic_region]
axes[1].scatter(emb[:,0], emb[:,1], c=region_colors, s=4, alpha=0.5)
axes[1].set_title('Toxic vs Non-toxic UMAP Regions')
axes[1].set_xlabel('UMAP 1'); axes[1].set_ylabel('UMAP 2')
axes[1].legend(handles=[
    mpatches.Patch(color='#d62728', label='Toxic UMAP region'),
    mpatches.Patch(color='#2ca02c', label='Non-toxic UMAP region')
])
plt.suptitle('HDBSCAN Clustering of ChEMBL hERG Chemical Space', fontsize=13)
plt.tight_layout()
plt.savefig('herg_umap_clusters.png', dpi=150, bbox_inches='tight')
print("Saved herg_umap_clusters.png")
plt.close()

# ── UMAP-informed bit enrichment ───────────────────────────────────────────
print("\nFinding fingerprint bits enriched in toxic UMAP regions...")
umap_corrs = []
for i in range(2048):
    col = X[:, i]
    if col.std() > 0:
        r, _ = pointbiserialr(in_toxic_region, col)
        umap_corrs.append((i, abs(r), r))
    else:
        umap_corrs.append((i, 0.0, 0.0))

umap_corr_df = pd.DataFrame(umap_corrs, columns=['bit','abs_corr','corr'])
umap_corr_df = umap_corr_df.sort_values('abs_corr', ascending=False)
umap_corr_df.to_csv('herg_umap_bit_enrichment.csv', index=False)

# Save top 300 bits for training script
top300 = umap_corr_df.head(300)['bit'].tolist()
pd.Series(top300).to_csv('herg_top300_umap_bits.csv', index=False, header=False)
print(f"Saved top 300 UMAP-enriched bits to herg_top300_umap_bits.csv")

# Bar chart of top 20 bits
top20_bits = umap_corr_df.head(20)
fig, ax = plt.subplots(figsize=(10, 5))
bit_labels = [f"bit_{int(r['bit'])}" for _, r in top20_bits.iterrows()]
bit_colors = ['#d62728' if r['corr'] > 0 else '#1f77b4' for _, r in top20_bits.iterrows()]
ax.bar(bit_labels, top20_bits['abs_corr'], color=bit_colors, edgecolor='black')
ax.set_xticklabels(bit_labels, rotation=45, ha='right', fontsize=8)
ax.set_ylabel('|Point-biserial correlation with toxic UMAP region|')
ax.set_title('Top 20 FCFP4 bits enriched in toxic hERG UMAP clusters\n'
             '(red = enriched in toxic regions, blue = enriched in non-toxic)')
ax.legend(handles=[mpatches.Patch(color='#d62728', label='→ toxic region'),
                   mpatches.Patch(color='#1f77b4', label='→ non-toxic region')])
plt.tight_layout()
plt.savefig('herg_top20_bits.png', dpi=150, bbox_inches='tight')
print("Saved herg_top20_bits.png")
plt.close()

# ── Murcko scaffold enrichment (top 20 toxic scaffolds) ───────────────────
print("\nComputing Murcko scaffold enrichment...")
scaffolds = []
for smi in meta['canonical_smiles']:
    try:
        mol = Chem.MolFromSmiles(str(smi))
        if mol:
            sca = MurckoScaffold.GetScaffoldForMol(mol)
            scaffolds.append(Chem.MolToSmiles(sca))
        else:
            scaffolds.append(None)
    except:
        scaffolds.append(None)

meta['scaffold'] = scaffolds
meta_sca = meta[meta['scaffold'].notna()].copy()

stats = meta_sca.groupby('scaffold').agg(
    count      = ('toxic', 'count'),
    n_toxic    = ('toxic', 'sum'),
    mean_pIC50 = ('pIC50',  'mean')
).reset_index()
stats['toxic_rate'] = stats['n_toxic'] / stats['count']

top20_sca = (stats[stats['count'] >= 5]
             .sort_values('toxic_rate', ascending=False)
             .head(20)
             .reset_index(drop=True))
top20_sca.to_csv('herg_top20_scaffolds.csv', index=False)
print("Top 20 toxic scaffolds:")
print(top20_sca[['scaffold','count','n_toxic','toxic_rate','mean_pIC50']].to_string())

mols = [Chem.MolFromSmiles(s) for s in top20_sca['scaffold']]
legends = [
    f"#{i+1} | n={row['count']} | {row['toxic_rate']:.0%} blockers | pIC50={row['mean_pIC50']:.1f}"
    for i, (_, row) in enumerate(top20_sca.iterrows())
]
img = Draw.MolsToGridImage(mols, molsPerRow=4, subImgSize=(350, 250), legends=legends)
img.save('herg_top20_scaffolds.png')
print("Saved herg_top20_scaffolds.png")

# ── Save final embedding CSV ───────────────────────────────────────────────
emb_df = pd.DataFrame({
    'UMAP_1':         emb[:,0],
    'UMAP_2':         emb[:,1],
    'pIC50':          pic50,
    'toxic':          y,
    'cluster':        cluster_labels,
    'in_toxic_region': in_toxic_region,
    'smiles':         meta['canonical_smiles'].values
})
emb_df.to_csv('herg_embedding.csv', index=False)
print("\nSaved herg_embedding.csv")
print("\nAll done.")