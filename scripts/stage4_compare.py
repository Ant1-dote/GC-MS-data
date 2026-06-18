import pandas as pd, numpy as np, time, warnings
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
warnings.filterwarnings('ignore')
OUT = Path('output')
t0 = time.time()
print('[1] Loading data...')
combined = pd.read_parquet(OUT / 'combined.parquet')
mask = combined['DBE'].notna() & (combined['n_fragments'] >= 3)
model_df = combined[mask].copy()
n_total = len(model_df)
print('  Valid samples: %d' % n_total)
spec_mat = np.load(OUT / 'spectral_matrix.npy')
spec_mat = spec_mat[mask.values]
X_feat = model_df[['MW','n_fragments','base_peak_mz','base_peak_int','tic_log','spectral_entropy']].values
X = np.hstack([spec_mat, X_feat])
y = model_df['DBE'].values
mw_vals = model_df['MW'].values
print('  Feature matrix: %s, MW range: %.1f ~ %.1f' % (str(X.shape), mw_vals.min(), mw_vals.max()))
from sklearn.model_selection import train_test_split
idx = np.arange(n_total)
X_tr, X_te, y_tr, y_te, ix_tr, ix_te = train_test_split(X, y, idx, test_size=0.2, random_state=42)
mw_te = mw_vals[ix_te]
print('  Train: %d, Test: %d' % (X_tr.shape[0], X_te.shape[0]))
# LightGBM baseline (same as Stage 3)
print('\n[2] LightGBM baseline...')
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
lgb = LGBMRegressor(n_estimators=300, max_depth=12, num_leaves=63, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, min_child_samples=20, random_state=42, verbose=-1)
lgb.fit(X_tr, y_tr)
y_pred_lgb = lgb.predict(X_te)
def metrics(y_true, y_pred, name=''):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    print('  %s  MAE=%.3f  RMSE=%.3f  R2=%.4f' % (name, mae, rmse, r2))
    return {'model':name, 'MAE':round(mae,4), 'RMSE':round(rmse,4), 'R2':round(r2,4)}
res_lgb = metrics(y_te, y_pred_lgb, 'LGB')
# XGBoost + hyperparameter tuning (subsampled)
print('\n[3] XGBoost with hyperparameter tuning (subsampled)...')
import xgboost as xgb
from sklearn.model_selection import RandomizedSearchCV
np.random.seed(42)
n_tune = min(30000, len(X_tr))
ti = np.random.choice(len(X_tr), n_tune, replace=False)
X_tune = X_tr[ti]; y_tune = y_tr[ti]
print('  Tuning subset: %d samples' % n_tune)
param_dist = {'n_estimators':[200,300,500], 'max_depth':[6,8,10,12], 'learning_rate':[0.01,0.03,0.05], 'subsample':[0.6,0.8], 'colsample_bytree':[0.6,0.8], 'min_child_weight':[1,3,5], 'gamma':[0,0.1], 'reg_alpha':[0,0.1], 'reg_lambda':[0,0.1]}
n_iter = 15
print('  RandomizedSearchCV (%d iters, 2-fold CV)...' % n_iter)
xgb_base = xgb.XGBRegressor(random_state=42, verbosity=0, n_jobs=-1)
rs = RandomizedSearchCV(xgb_base, param_dist, n_iter=n_iter, scoring='neg_mean_absolute_error', cv=2, random_state=42, verbose=0, n_jobs=-1)
rs.fit(X_tune, y_tune)
print('  Best params: %s' % str(rs.best_params_))
print('  Best CV MAE: %.3f' % -rs.best_score_)
print('  Retraining best model on full training set...')
rs.best_estimator_.fit(X_tr, y_tr)
y_pred_xgb = rs.predict(X_te)
res_xgb = metrics(y_te, y_pred_xgb, 'XGB')
# Model comparison
print('\n[4] Model comparison:')
summary = pd.DataFrame([res_lgb, res_xgb])
print(summary.to_string(index=False))
summary.to_parquet(OUT / 'model_comparison.parquet')
print('  Saved: model_comparison.parquet')
# Residual analysis
print('\n[5] Residual analysis...')
rl = y_te - y_pred_lgb; rx = y_te - y_pred_xgb
for lbl, r in [('LGB',rl),('XGB',rx)]:
    sk = float(pd.Series(r).skew())
    p95 = float(np.quantile(np.abs(r), 0.95))
    print('  %s: mean=%+.3f  std=%.3f  skew=%.3f  |res|_p95=%.3f' % (lbl, r.mean(), r.std(), sk, p95))
residual_df = pd.DataFrame({'true_dbe':y_te, 'pred_lgb':y_pred_lgb, 'pred_xgb':y_pred_xgb, 'res_lgb':rl, 'res_xgb':rx, 'MW':mw_te})
residual_df.to_parquet(OUT / 'residuals.parquet')
print('  Saved: residuals.parquet (%d rows)' % len(residual_df))
# MW-stratified evaluation
print('\n[6] MW-stratified evaluation...')
mw_labels = ['<100','100-150','150-200','200-300','300-500','>=500']
mw_bins = [-float('inf'), 100, 150, 200, 300, 500, float('inf')]
mw_group = pd.cut(mw_te, bins=mw_bins, labels=mw_labels)
rows = []
for rv, yp, rn in [(rl, y_pred_lgb, 'LGB'), (rx, y_pred_xgb, 'XGB')]:
    for l in mw_labels:
        sel = mw_group == l
        if sel.sum() < 3: continue
        ys = y_te[sel]; yps = yp[sel]
        rows.append({'model':rn, 'MW_range':l, 'n':int(sel.sum()), 'MAE':mean_absolute_error(ys, yps), 'RMSE':float(np.sqrt(mean_squared_error(ys, yps))), 'R2':r2_score(ys, yps)})
sdf = pd.DataFrame(rows)
print(sdf.round(3).to_string(index=False))
sdf.to_parquet(OUT / 'strata_by_mw.parquet')
print('  Saved: strata_by_mw.parquet')
# Visualizations
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Residual Analysis: LGB vs XGBoost', fontsize=14, y=0.98)
ax = axes[0,0]
ax.scatter(y_pred_lgb, rl, s=3, alpha=0.3, c='#2196F3', label='LGB')
ax.scatter(y_pred_xgb, rx, s=3, alpha=0.3, c='#FF5722', label='XGB')
ax.axhline(0, c='gray', ls='--', lw=0.8)
ax.set_xlabel('Predicted DBE'); ax.set_ylabel('Residual'); ax.legend()
ax = axes[0,1]; bins = np.linspace(-10, 10, 61)
ax.hist(rl, bins=bins, alpha=0.5, label='LGB', color='#2196F3', edgecolor='white', lw=0.3)
ax.hist(rx, bins=bins, alpha=0.5, label='XGB', color='#FF5722', edgecolor='white', lw=0.3)
ax.axvline(0, c='gray', ls='--', lw=0.8); ax.legend()
ax.set_xlabel('Residual'); ax.set_title('Residual Distribution')
ax = axes[1,0]
bl = [np.abs(rl[mw_group==l]) for l in mw_labels]
bx = [np.abs(rx[mw_group==l]) for l in mw_labels]
pl = np.arange(len(mw_labels))*2 - 0.3; px = np.arange(len(mw_labels))*2 + 0.3
bp1 = ax.boxplot(bl, positions=pl, widths=0.5, patch_artist=True, boxprops=dict(facecolor='#2196F3', alpha=0.6), medianprops=dict(color='white'))
bp2 = ax.boxplot(bx, positions=px, widths=0.5, patch_artist=True, boxprops=dict(facecolor='#FF5722', alpha=0.6), medianprops=dict(color='white'))
ax.set_xticks(np.arange(len(mw_labels))*2); ax.set_xticklabels(mw_labels, fontsize=9)
ax.set_xlabel('MW Range'); ax.set_ylabel('|Residual|'); ax.set_title('|Residual| by MW Stratum')
ax.legend([bp1['boxes'][0], bp2['boxes'][0]], ['LGB','XGB'], loc='upper left')
ax = axes[1,1]
ax.scatter(y_te, y_pred_lgb, s=3, alpha=0.3, c='#2196F3', label='LGB')
ax.scatter(y_te, y_pred_xgb, s=3, alpha=0.3, c='#FF5722', label='XGB')
lims = [min(y_te.min(),y_pred_lgb.min(),y_pred_xgb.min())-1, max(y_te.max(),y_pred_lgb.max(),y_pred_xgb.max())+1]
ax.plot(lims, lims, c='gray', ls='--', lw=0.8)
ax.set_xlim(lims); ax.set_ylim(lims)
ax.set_xlabel('True DBE'); ax.set_ylabel('Predicted DBE'); ax.legend(); ax.set_aspect('equal')
plt.tight_layout(rect=[0,0,1,0.96])
fig.savefig(OUT / 'viz_residual_analysis.png', dpi=150, bbox_inches='tight')
plt.close()
print('  Saved: viz_residual_analysis.png')
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('MW-Stratified Performance', fontsize=14, y=1.02)
for mi, metric in enumerate(['MAE','R2']):
    ax = axes[mi]
    for mij, (mo, co) in enumerate([('LGB','#2196F3'),('XGB','#FF5722')]):
        sub = sdf[sdf['model'] == mo]
        x = np.arange(len(sub)) + (mij - 0.5) * 0.3
        vals = sub[metric].values
        ax.bar(x, vals, width=0.3, color=co, alpha=0.8, label=mo)
        for xi, v in zip(x, vals):
            ax.text(xi, v+(0.02 if metric=='R2' else 0.005), '%.2f' % v, ha='center', fontsize=7)
    ax.set_xticks(np.arange(len(sub)))
    ax.set_xticklabels(sub['MW_range'].values, fontsize=9)
    ax.set_xlabel('MW Range'); ax.set_ylabel(metric); ax.legend()
plt.tight_layout()
fig.savefig(OUT / 'viz_strata_mw.png', dpi=150, bbox_inches='tight')
plt.close()
print('  Saved: viz_strata_mw.png')
# Save best model
if res_xgb['MAE'] < res_lgb['MAE']:
    print('\n[7] XGBoost outperforms LGB -- saving model...')
    rs.best_estimator_.save_model(str(OUT / 'xgb_dbe_model.json'))
    print('  Saved: xgb_dbe_model.json')
else:
    print('\n[7] LightGBM remains best -- no new model saved.')
print('\nDone! Total time: %.1fs' % (time.time()-t0))
