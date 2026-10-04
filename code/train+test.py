import matplotlib
matplotlib.use('Agg')
import numpy as np
import pandas as pd
import shutil, os
from rdkit import Chem, RDLogger
from rdkit.Chem import rdFingerprintGenerator
from autogluon.tabular import TabularPredictor
from sklearn.model_selection import train_test_split
from sklearn.metrics import (roc_auc_score, roc_curve, confusion_matrix,
                              precision_score, recall_score, f1_score)
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
RDLogger.DisableLog('rdApp.*')

# ── Clean up old model directories ────────────────────────────────────────
for model_dir in ['herg_model_A', 'herg_model_B', 'herg_model_C']:
    if os.path.exists(model_dir):
        shutil.rmtree(model_dir)
        print(f"Removed old {model_dir}/")

# ── Fingerprint generator ─────────────────────────────────────────────────
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

FEAT_COLS = [f'fp_{i}' for i in range(2048)]

# ── Evaluation helper ─────────────────────────────────────────────────────
def full_eval(predictor, test_features, y_true, model_name, label_col='toxic'):
    """Compute AUROC, Youden threshold, confusion matrix, FP/FN breakdown."""
    probs = predictor.predict_proba(test_features)[1].values
    auroc = roc_auc_score(y_true, probs)

    # ROC curve + Youden's index
    fpr, tpr, thresholds = roc_curve(y_true, probs)
    j_scores  = tpr - fpr
    best_idx  = np.argmax(j_scores)
    best_thresh = thresholds[best_idx]
    best_j    = j_scores[best_idx]

    y_pred = (probs >= best_thresh).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    precision = precision_score(y_true, y_pred, zero_division=0)
    recall    = recall_score(y_true, y_pred, zero_division=0)
    f1        = f1_score(y_true, y_pred, zero_division=0)
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

    print(f"\n{'='*55}")
    print(f"  {model_name} — TDC External Evaluation")
    print(f"{'='*55}")
    print(f"  AUROC:              {auroc:.4f}")
    print(f"  Youden's J:         {best_j:.4f}  (threshold = {best_thresh:.4f})")
    print(f"  Sensitivity/Recall: {recall:.4f}  ({tp} of {tp+fn} blockers found)")
    print(f"  Specificity:        {specificity:.4f}")
    print(f"  Precision:          {precision:.4f}")
    print(f"  F1 Score:           {f1:.4f}")
    print(f"\n  Confusion Matrix (at Youden threshold):")
    print(f"  {'':20s}  Pred Non-blocker  Pred Blocker")
    print(f"  {'True Non-blocker':20s}  {tn:>16}  {fp:>12}  ← False Positives")
    print(f"  {'True Blocker':20s}  {fn:>16}  {tp:>12}")
    print(f"                                      ↑ False Negatives")
    print(f"\n  False Positive Rate: {fp/(fp+tn):.4f}  ({fp} non-blockers misclassified as toxic)")
    print(f"  False Negative Rate: {fn/(fn+tp):.4f}  ({fn} blockers missed — safety risk)")

    return auroc, probs, best_thresh, {'tp':tp,'tn':tn,'fp':fp,'fn':fn,
                                        'recall':recall,'precision':precision,
                                        'specificity':specificity,'f1':f1,
                                        'youden_j':best_j,'threshold':best_thresh}

# ── Load ChEMBL hERG ──────────────────────────────────────────────────────
print("Loading ChEMBL hERG...")
X_all = np.load('herg_X.npy').astype(np.float32)
meta  = pd.read_csv('herg_clean.csv')
y_all = meta['toxic'].values
print(f"Total: {len(y_all)} | Blockers: {y_all.sum()} | Non-blockers: {(y_all==0).sum()}")

# ── Train/val split (80/20 stratified) ────────────────────────────────────
X_tr, X_val, y_tr, y_val = train_test_split(
    X_all, y_all, test_size=0.2, random_state=42, stratify=y_all)
print(f"Train: {len(y_tr)} | Val: {len(y_val)}")

train_df = pd.DataFrame(X_tr, columns=FEAT_COLS)
train_df['toxic'] = y_tr
val_df = pd.DataFrame(X_val, columns=FEAT_COLS)
val_df['toxic'] = y_val

# ── Load UMAP-enriched bits + continuous correlation weights ───────────────
print("\nLoading UMAP-enriched bits and correlation weights...")
top300_umap   = pd.read_csv('herg_top300_umap_bits.csv', header=None)[0].tolist()
top_cols_umap = [f'fp_{i}' for i in top300_umap]

# Full correlation scores for all 2048 bits (from herg_umap_features.py)
corr_df = pd.read_csv('herg_umap_bit_enrichment.csv')
corr_weights = (corr_df.set_index('bit')['abs_corr']
                .reindex(range(2048)).fillna(0).values.astype(np.float32))

# Continuous weight: w_i = 1 + 5 * correlation_i
# Top bits get up to 6x their original binary value; uncorrelated bits stay at 1x
WEIGHT_SCALE = 5.0
feature_weights = 1.0 + WEIGHT_SCALE * corr_weights
print(f"Feature weight range: {feature_weights.min():.3f} – {feature_weights.max():.3f}")
print(f"Top 5 bit weights: {sorted(feature_weights, reverse=True)[:5]}")

# Apply continuous weights to all splits
X_tr_C   = X_tr  * feature_weights[np.newaxis, :]
X_val_C  = X_val * feature_weights[np.newaxis, :]

WFEAT_COLS = [f'wfp_{i}' for i in range(2048)]
train_C = pd.DataFrame(X_tr_C,  columns=WFEAT_COLS)
train_C['toxic'] = y_tr
val_C   = pd.DataFrame(X_val_C, columns=WFEAT_COLS)
val_C['toxic'] = y_val

# ── Load TDC hERG (deduplicated if available, else original) ──────────────
tdc_file = 'herg_tdc_deduped.csv' if os.path.exists('herg_tdc_deduped.csv') else 'herg_dataset.csv'
print(f"\nLoading TDC hERG from {tdc_file}...")
tdc = pd.read_csv(tdc_file).dropna(subset=['Drug', 'Y'])

fps_tdc, y_tdc = [], []
for _, row in tdc.iterrows():
    fp = safe_fp(row['Drug'])
    if fp is not None:
        fps_tdc.append(fp)
        y_tdc.append(int(row['Y']))

X_tdc  = np.array(fps_tdc, dtype=np.float32)
y_tdc  = np.array(y_tdc)
test_df   = pd.DataFrame(X_tdc, columns=FEAT_COLS)
test_df_C = pd.DataFrame(X_tdc * feature_weights[np.newaxis, :], columns=WFEAT_COLS)
print(f"TDC hERG: {len(y_tdc)} | Blockers: {y_tdc.sum()} | Non-blockers: {(y_tdc==0).sum()}")

# ══════════════════════════════════════════════════════════════════════════
# MODEL A — Baseline (all 2048 bits, binary)
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "="*55)
print("MODEL A: Baseline — all 2048 bits")
print("="*55)

pred_A = TabularPredictor(
    label='toxic', eval_metric='roc_auc', path='herg_model_A'
).fit(
    train_data=train_df, tuning_data=val_df,
    use_bag_holdout=True, time_limit=1800,
    presets='best_quality', verbosity=1
)

val_A_score = pred_A.evaluate(val_df)['roc_auc']
print(f"\nModel A — ChEMBL Val AUROC: {val_A_score:.4f}")
print(pred_A.leaderboard(silent=True)[['model','score_val']].to_string())
tdc_A_score, probs_A, thresh_A, metrics_A = full_eval(
    pred_A, test_df, y_tdc, "Model A — Baseline (2048 bits)")

# ══════════════════════════════════════════════════════════════════════════
# MODEL B — UMAP feature selection (top 300 bits)
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "="*55)
print("MODEL B: UMAP-informed — top 300 bits")
print("="*55)

train_B = train_df[top_cols_umap + ['toxic']].copy()
val_B   = val_df[top_cols_umap + ['toxic']].copy()
test_B  = test_df[top_cols_umap].copy()

pred_B = TabularPredictor(
    label='toxic', eval_metric='roc_auc', path='herg_model_B'
).fit(
    train_data=train_B, tuning_data=val_B,
    use_bag_holdout=True, time_limit=1800,
    presets='best_quality', verbosity=1
)

val_B_score = pred_B.evaluate(val_B)['roc_auc']
print(f"\nModel B — ChEMBL Val AUROC: {val_B_score:.4f}")
print(pred_B.leaderboard(silent=True)[['model','score_val']].to_string())
tdc_B_score, probs_B, thresh_B, metrics_B = full_eval(
    pred_B, test_B, y_tdc, "Model B — UMAP feature selection (300 bits)")

# ══════════════════════════════════════════════════════════════════════════
# MODEL C — Continuous UMAP reweighting (all 2048 bits, scaled)
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "="*55)
print("MODEL C: Continuous reweighting — all 2048 bits, UMAP-scaled")
print("="*55)
print(f"Each bit i is multiplied by (1 + {WEIGHT_SCALE} × corr_i)")
print(f"Top UMAP bits receive up to {1+WEIGHT_SCALE:.0f}x their binary value")

pred_C = TabularPredictor(
    label='toxic', eval_metric='roc_auc', path='herg_model_C'
).fit(
    train_data=train_C, tuning_data=val_C,
    use_bag_holdout=True, time_limit=1800,
    presets='best_quality', verbosity=1
)

val_C_score = pred_C.evaluate(val_C)['roc_auc']
print(f"\nModel C — ChEMBL Val AUROC: {val_C_score:.4f}")
print(pred_C.leaderboard(silent=True)[['model','score_val']].to_string())
tdc_C_score, probs_C, thresh_C, metrics_C = full_eval(
    pred_C, test_df_C, y_tdc, "Model C — Continuous UMAP reweighting (2048 bits)")

# ══════════════════════════════════════════════════════════════════════════
# SUMMARY TABLE
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "="*65)
print("FINAL COMPARISON")
print("="*65)

summary = pd.DataFrame([
    {'Model': 'A — Baseline (2048 bits)',
     'Val AUROC': val_A_score, 'TDC AUROC': tdc_A_score, 'Features': 2048,
     'Precision': metrics_A['precision'], 'Recall': metrics_A['recall'],
     'F1': metrics_A['f1'], 'FP': metrics_A['fp'], 'FN': metrics_A['fn'],
     "Youden's J": metrics_A['youden_j']},
    {'Model': 'B — UMAP selection (300 bits)',
     'Val AUROC': val_B_score, 'TDC AUROC': tdc_B_score, 'Features': 300,
     'Precision': metrics_B['precision'], 'Recall': metrics_B['recall'],
     'F1': metrics_B['f1'], 'FP': metrics_B['fp'], 'FN': metrics_B['fn'],
     "Youden's J": metrics_B['youden_j']},
    {'Model': 'C — Continuous reweighting (2048 bits)',
     'Val AUROC': val_C_score, 'TDC AUROC': tdc_C_score, 'Features': 2048,
     'Precision': metrics_C['precision'], 'Recall': metrics_C['recall'],
     'F1': metrics_C['f1'], 'FP': metrics_C['fp'], 'FN': metrics_C['fn'],
     "Youden's J": metrics_C['youden_j']},
])
print(summary.to_string(index=False))
summary.to_csv('herg_model_comparison.csv', index=False)

# ══════════════════════════════════════════════════════════════════════════
# PLOTS
# ══════════════════════════════════════════════════════════════════════════

# ── Plot 1: AUROC grouped bar chart ───────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 5))
x = np.arange(3)
w = 0.3
ax.bar(x - w/2, summary['Val AUROC'], width=w, color='steelblue',
       edgecolor='black', label='ChEMBL val (in-distribution)')
ax.bar(x + w/2, summary['TDC AUROC'], width=w, color='darkorange',
       edgecolor='black', label='TDC hERG (external test)')
ax.set_xticks(x)
ax.set_xticklabels(['A — Baseline\n(2048 bits)',
                    'B — UMAP selection\n(300 bits)',
                    'C — Continuous\nreweighting'], fontsize=9)
ax.set_ylim(0.4, 1.0)
ax.axhline(0.5, color='red', linestyle='--', linewidth=1, label='Random (0.50)')
ax.set_ylabel('AUROC', fontsize=12)
ax.set_title('hERG Toxicity Prediction: Model Comparison', fontsize=12)
ax.legend()
for i, row in summary.iterrows():
    ax.text(i - w/2, row['Val AUROC'] + 0.008, f"{row['Val AUROC']:.3f}",
            ha='center', va='bottom', fontsize=8)
    ax.text(i + w/2, row['TDC AUROC'] + 0.008, f"{row['TDC AUROC']:.3f}",
            ha='center', va='bottom', fontsize=8)
plt.tight_layout()
plt.savefig('herg_model_comparison.png', dpi=150, bbox_inches='tight')
plt.close()
print("\nSaved herg_model_comparison.png")

# ── Plot 2: ROC curves for all three models ───────────────────────────────
fig, ax = plt.subplots(figsize=(7, 6))
for probs, label, color, thresh in [
    (probs_A, f'Model A AUROC={tdc_A_score:.3f}', 'steelblue',  thresh_A),
    (probs_B, f'Model B AUROC={tdc_B_score:.3f}', 'darkorange', thresh_B),
    (probs_C, f'Model C AUROC={tdc_C_score:.3f}', 'seagreen',   thresh_C),
]:
    fpr, tpr, thresholds = roc_curve(y_tdc, probs)
    ax.plot(fpr, tpr, color=color, linewidth=2, label=label)
    # Mark Youden threshold
    j = tpr - fpr
    best = np.argmax(j)
    ax.scatter(fpr[best], tpr[best], color=color, s=80, zorder=5, marker='D')

ax.plot([0,1],[0,1], 'k--', linewidth=1, label='Random')
ax.set_xlabel('False Positive Rate', fontsize=12)
ax.set_ylabel('True Positive Rate', fontsize=12)
ax.set_title('ROC Curves — TDC hERG External Test\n(diamonds = Youden optimal threshold)', fontsize=11)
ax.legend(fontsize=9)
plt.tight_layout()
plt.savefig('herg_roc_curves.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved herg_roc_curves.png")

# ── Plot 3: FP / FN comparison ────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
model_names = ['A\nBaseline', 'B\nUMAP\nselection', 'C\nContinuous\nreweighting']
colors_fp = ['#d62728', '#ff7f0e', '#2ca02c']

axes[0].bar(model_names, [metrics_A['fp'], metrics_B['fp'], metrics_C['fp']],
            color=colors_fp, edgecolor='black')
axes[0].set_title('False Positives\n(non-blockers predicted as toxic)', fontsize=10)
axes[0].set_ylabel('Count')
for i, v in enumerate([metrics_A['fp'], metrics_B['fp'], metrics_C['fp']]):
    axes[0].text(i, v + 0.5, str(v), ha='center', va='bottom', fontsize=10)

axes[1].bar(model_names, [metrics_A['fn'], metrics_B['fn'], metrics_C['fn']],
            color=colors_fp, edgecolor='black')
axes[1].set_title('False Negatives\n(blockers missed — safety risk)', fontsize=10)
axes[1].set_ylabel('Count')
for i, v in enumerate([metrics_A['fn'], metrics_B['fn'], metrics_C['fn']]):
    axes[1].text(i, v + 0.5, str(v), ha='center', va='bottom', fontsize=10)

plt.suptitle('False Positive vs False Negative Trade-off (Youden threshold)', fontsize=11)
plt.tight_layout()
plt.savefig('herg_fp_fn_comparison.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved herg_fp_fn_comparison.png")

print("\nAll done.")