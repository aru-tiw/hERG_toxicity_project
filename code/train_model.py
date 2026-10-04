# save as train_model.py
import matplotlib
matplotlib.use('Agg')
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
from rdkit import RDLogger
from sklearn.model_selection import train_test_split
from autogluon.tabular import TabularPredictor
import pickle
import warnings
warnings.filterwarnings('ignore')
RDLogger.DisableLog('rdApp.*')

# ── Fingerprint generator (FCFP4 — same as UMAP) ─────────
gen = rdFingerprintGenerator.GetMorganGenerator(
    radius=2, fpSize=2048,
    atomInvariantsGenerator=rdFingerprintGenerator.GetMorganFeatureAtomInvGen()
)
FEATURE_COLS = [f'fp_{i}' for i in range(2048)]

def safe_fp(smi):
    try:
        mol = Chem.MolFromSmiles(str(smi))
        if mol:
            return np.array(gen.GetFingerprint(mol), dtype=np.uint8)
    except:
        pass
    return None

def to_df(smiles_list, labels, feature_cols):
    fps = [safe_fp(s) for s in smiles_list]
    mask = [fp is not None for fp in fps]
    fps_clean = np.vstack([fp for fp in fps if fp is not None])
    labels_clean = np.array(labels)[mask]
    df = pd.DataFrame(fps_clean, columns=feature_cols)
    df['toxic'] = labels_clean
    return df

# ── Build ChEMBL training data ────────────────────────────
print("Loading ChEMBL embedding...")
chembl = pd.read_csv('final_embedding.csv')
chembl = chembl[chembl['canonical_smiles'].notna()].reset_index(drop=True)

print(f"ChEMBL compounds: {len(chembl):,}")
print(f"Toxic: {chembl['toxic'].sum():,} | Non-toxic: {(chembl['toxic']==0).sum():,}")

print("Computing FCFP4 fingerprints for training data...")
train_df = to_df(
    chembl['canonical_smiles'].tolist(),
    chembl['toxic'].tolist(),
    FEATURE_COLS
)
print(f"Training set: {len(train_df):,} compounds")

# ── Build ClinTox test data ───────────────────────────────
print("\nLoading ClinTox test data...")
clintox = pd.read_csv('clintox.csv')
clintox = clintox[clintox['smiles'].notna()].reset_index(drop=True)

# CT_TOX=1 means failed human trials due to toxicity
test_df = to_df(
    clintox['smiles'].tolist(),
    clintox['CT_TOX'].tolist(),
    FEATURE_COLS
)
print(f"Test set (ClinTox): {len(test_df):,} compounds")
print(f"CT_TOX=1 (failed humans): {test_df['toxic'].sum():,}")
print(f"CT_TOX=0 (passed):        {(test_df['toxic']==0).sum():,}")

# ── Train AutoGluon ───────────────────────────────────────
print("\nTraining AutoGluon...")
predictor = TabularPredictor(
    label='toxic',
    eval_metric='roc_auc',
    path='cardiac_tox_model'
).fit(
    train_data=train_df,
    time_limit=600,
    presets='best_quality',
    verbosity=2
)

# ── Evaluate on ClinTox ───────────────────────────────────
print("\n--- Leaderboard (validation) ---")
lb = predictor.leaderboard(test_df, silent=True)
print(lb[['model', 'score_test', 'score_val']].to_string(index=False))

print("\n--- Evaluation on ClinTox (held-out test) ---")
results = predictor.evaluate(test_df)
print(results)

# ── Save predictions + probabilities ─────────────────────
probs = predictor.predict_proba(test_df)
clintox_results = clintox[['smiles']].copy().iloc[:len(probs)]
clintox_results['CT_TOX_true']       = test_df['toxic'].values
clintox_results['pred_toxic_prob']   = probs[1].values
clintox_results['pred_toxic_binary'] = (probs[1].values > 0.5).astype(int)
clintox_results.to_csv('clintox_predictions.csv', index=False)
print("\nSaved predictions to clintox_predictions.csv")

# ── Feature importance ────────────────────────────────────
print("\n--- Top 20 most important fingerprint bits ---")
importance = predictor.feature_importance(test_df, silent=True)
print(importance.head(20))
importance.to_csv('feature_importance.csv')
print("Saved feature importance to feature_importance.csv")

# ── Project ClinTox onto UMAP ─────────────────────────────
print("\nProjecting ClinTox onto ChEMBL UMAP...")
with open('umap_reducer.pkl', 'rb') as f:
    saved = pickle.load(f)
reducer = saved['reducer']
pca     = saved['pca']

fps_clintox = []
valid_idx   = []
for i, smi in enumerate(clintox['smiles']):
    fp = safe_fp(smi)
    if fp is not None:
        fps_clintox.append(fp)
        valid_idx.append(i)

X_clintox     = np.vstack(fps_clintox).astype(float)
X_clintox_pca = pca.transform(X_clintox)
clintox_emb   = reducer.transform(X_clintox_pca)

clintox_umap = clintox.iloc[valid_idx].copy()
clintox_umap['UMAP_1']          = clintox_emb[:, 0]
clintox_umap['UMAP_2']          = clintox_emb[:, 1]
clintox_umap['pred_toxic_prob'] = probs[1].values[:len(valid_idx)]
clintox_umap.to_csv('clintox_umap.csv', index=False)
print("Saved ClinTox UMAP projection to clintox_umap.csv")
print("\nDone.")