import numpy as np
import pandas as pd
import pickle
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from rdkit import Chem, RDLogger
from rdkit.Chem import rdFingerprintGenerator
from sklearn.neighbors import NearestNeighbors
from scipy.stats import mannwhitneyu
import warnings
warnings.filterwarnings('ignore')
RDLogger.DisableLog('rdApp.*')

# ── Fingerprint generator (must match training) ────────────────────────────
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

# ── Load ChEMBL UMAP embedding (background) ───────────────────────────────
print("Loading ChEMBL hERG embedding...")
chembl_emb = pd.read_csv('herg_embedding.csv')
print(f"ChEMBL: {len(chembl_emb)} compounds")

# ── Load saved PCA and UMAP reducers ──────────────────────────────────────
print("Loading reducers...")
with open('herg_pca.pkl', 'rb') as f:
    pca = pickle.load(f)
with open('herg_umap.pkl', 'rb') as f:
    umap_reducer = pickle.load(f)

# ── Load and featurize TDC hERG ───────────────────────────────────────────
print("Loading TDC hERG...")
tdc = pd.read_csv('herg_dataset.csv').dropna(subset=['Drug', 'Y'])

fps, labels, names = [], [], []
for _, row in tdc.iterrows():
    fp = safe_fp(row['Drug'])
    if fp is not None:
        fps.append(fp)
        labels.append(int(row['Y']))
        names.append(str(row['Drug_ID']))

X_tdc = np.array(fps, dtype=np.float32)
y_tdc = np.array(labels)
print(f"TDC: {len(y_tdc)} compounds | Blockers: {y_tdc.sum()} | Non-blockers: {(y_tdc==0).sum()}")

# ── Project TDC into ChEMBL UMAP space ────────────────────────────────────
print("Projecting TDC compounds into ChEMBL UMAP space...")
pca_tdc  = pca.transform(X_tdc)
umap_tdc = umap_reducer.transform(pca_tdc)
print(f"Projection done: {umap_tdc.shape}")

tdc_neg = y_tdc == 0
tdc_pos = y_tdc == 1

# ── Plot 1: TDC overlaid on ChEMBL (colored by label) ────────────────────
fig, axes = plt.subplots(1, 2, figsize=(16, 7))

# Panel 1: ChEMBL grey background + TDC colored by blocker/non-blocker
axes[0].scatter(chembl_emb['UMAP_1'], chembl_emb['UMAP_2'],
                c='lightgrey', s=3, alpha=0.3, label='ChEMBL (background)')
axes[0].scatter(umap_tdc[tdc_neg, 0], umap_tdc[tdc_neg, 1],
                c='#1f77b4', s=18, alpha=0.7,
                edgecolors='white', linewidths=0.3,
                label=f'TDC non-blocker (n={tdc_neg.sum()})')
axes[0].scatter(umap_tdc[tdc_pos, 0], umap_tdc[tdc_pos, 1],
                c='#d62728', s=18, alpha=0.8,
                edgecolors='white', linewidths=0.3, zorder=5,
                label=f'TDC blocker (n={tdc_pos.sum()})')
axes[0].set_title('TDC hERG projected onto ChEMBL chemical space\n(grey = ChEMBL background)',
                  fontsize=11)
axes[0].set_xlabel('UMAP 1'); axes[0].set_ylabel('UMAP 2')
axes[0].legend(fontsize=9, markerscale=2)

# Panel 2: ChEMBL colored by actual toxic label + TDC overlay
chembl_colors = ['#ffb3b3' if t==1 else '#c8e6c9' for t in chembl_emb['toxic'].values]
axes[1].scatter(chembl_emb['UMAP_1'], chembl_emb['UMAP_2'],
                c=chembl_colors, s=3, alpha=0.4)
axes[1].scatter(umap_tdc[tdc_neg, 0], umap_tdc[tdc_neg, 1],
                c='#1f77b4', s=18, alpha=0.7,
                edgecolors='white', linewidths=0.3,
                label=f'TDC non-blocker (n={tdc_neg.sum()})')
axes[1].scatter(umap_tdc[tdc_pos, 0], umap_tdc[tdc_pos, 1],
                c='#d62728', s=18, alpha=0.9,
                edgecolors='white', linewidths=0.3, zorder=5,
                label=f'TDC blocker (n={tdc_pos.sum()})')
axes[1].set_title('TDC hERG overlaid on ChEMBL hERG blocker regions\n'
                  '(pink = ChEMBL blocker, green = ChEMBL non-blocker)', fontsize=11)
axes[1].set_xlabel('UMAP 1'); axes[1].set_ylabel('UMAP 2')
axes[1].legend(handles=[
    mpatches.Patch(color='#ffb3b3', label='ChEMBL hERG blocker'),
    mpatches.Patch(color='#c8e6c9', label='ChEMBL non-blocker'),
    mpatches.Patch(color='#d62728', label=f'TDC blocker (n={tdc_pos.sum()})'),
    mpatches.Patch(color='#1f77b4', label=f'TDC non-blocker (n={tdc_neg.sum()})'),
], fontsize=9)

plt.suptitle('Cross-dataset Projection: TDC hERG onto ChEMBL Chemical Space',
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('herg_tdc_overlay.png', dpi=150, bbox_inches='tight')
print("Saved herg_tdc_overlay.png")

# ── Quantify overlap using actual ChEMBL toxic labels ─────────────────────
print("\nQuantifying TDC compound overlap with ChEMBL hERG blocker regions...")

chembl_coords = chembl_emb[['UMAP_1', 'UMAP_2']].values
tdc_coords    = umap_tdc
chembl_toxic  = chembl_emb['toxic'].values  # actual blocker labels, not sparse cluster flag

nn = NearestNeighbors(n_neighbors=10).fit(chembl_coords)
distances, indices = nn.kneighbors(tdc_coords)

# Local toxicity score = fraction of 10 nearest ChEMBL neighbors that are blockers
tdc_local_tox = np.array([chembl_toxic[idx].mean() for idx in indices])

tdc_df = pd.DataFrame({
    'drug_id':         names,
    'tdc_label':       y_tdc,
    'local_tox_score': tdc_local_tox,
    'umap_1':          umap_tdc[:, 0],
    'umap_2':          umap_tdc[:, 1],
})
tdc_df.to_csv('herg_tdc_projection.csv', index=False)

blocker_scores    = tdc_local_tox[y_tdc == 1]
nonblocker_scores = tdc_local_tox[y_tdc == 0]
print(f"Mean fraction of hERG blocker neighbors in ChEMBL UMAP space:")
print(f"  TDC blockers:     {blocker_scores.mean():.3f}")
print(f"  TDC non-blockers: {nonblocker_scores.mean():.3f}")

stat, pval = mannwhitneyu(blocker_scores, nonblocker_scores, alternative='greater')
print(f"Mann-Whitney U test (blockers > non-blockers): p = {pval:.4e}")

# ── Distribution plot of local toxicity scores ─────────────────────────────
fig, ax = plt.subplots(figsize=(8, 5))
bins = np.linspace(0, 1, 25)
ax.hist(blocker_scores,    bins=bins, alpha=0.6, color='#d62728',
        label=f'TDC blockers (n={tdc_pos.sum()})',     density=True)
ax.hist(nonblocker_scores, bins=bins, alpha=0.6, color='#1f77b4',
        label=f'TDC non-blockers (n={tdc_neg.sum()})', density=True)
ax.axvline(blocker_scores.mean(),    color='#d62728', linestyle='--', linewidth=1.5,
           label=f'Blocker mean = {blocker_scores.mean():.3f}')
ax.axvline(nonblocker_scores.mean(), color='#1f77b4', linestyle='--', linewidth=1.5,
           label=f'Non-blocker mean = {nonblocker_scores.mean():.3f}')
ax.set_xlabel('Fraction of ChEMBL hERG blocker neighbors (k=10 in UMAP space)', fontsize=11)
ax.set_ylabel('Density', fontsize=11)
ax.set_title(f'TDC compounds: local ChEMBL blocker density in UMAP space\n'
             f'Mann-Whitney p = {pval:.2e}', fontsize=11)
ax.legend(fontsize=9)
plt.tight_layout()
plt.savefig('herg_tdc_local_tox.png', dpi=150, bbox_inches='tight')
print("Saved herg_tdc_local_tox.png")
print("Saved herg_tdc_projection.csv")