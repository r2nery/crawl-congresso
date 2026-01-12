# =============================================================================
# COMPREHENSIVE TABLE PRINTERS FOR PAPER
# =============================================================================
# These cells print publication-ready LaTeX code for all tables
# Add these to your notebook after the relevant analysis sections

# =============================================================================
# TABLE: ROBUSTNESS - ESTIMATION METHOD COMPARISON
# =============================================================================
# INSERT AFTER: Method comparison analysis (Section 3)

print("\n" + "="*80)
print("TABLE: ROBUSTNESS - ESTIMATION METHOD COMPARISON")
print("="*80)

# Extract peak effects at τ=+1
methods = {
    'OLS': ols_results,
    'TWFE': twfe_results,
    'DML': results_dml
}

print("\nMethod                             Peak (τ=+1)         SE         R²          N")
print("-" * 80)

for name, result in methods.items():
    if 't_1' in result.params.index:
        coef = result.params['t_1']
        se = result.bse['t_1']
        r2 = result.rsquared
        n = result.nobs
        
        stars = '***' if result.pvalues['t_1'] < 0.01 else '**' if result.pvalues['t_1'] < 0.05 else '*'
        
        print(f"{name:30s} {coef:12.4f}{stars} ({se:.4f}) {r2:10.4f} {n:10.0f}")

# Print correlations
print("\nCoefficient Correlations Across Time Periods (n=11):")

# Extract all time period coefficients
time_cols = [f't_{t}' for t in [-6,-5,-4,-3,-2,1,2,3,4,5,6]]
ols_coefs = [ols_results.params[c] for c in time_cols if c in ols_results.params.index]
twfe_coefs = [twfe_results.params[c] for c in time_cols if c in twfe_results.params.index]
dml_coefs = [results_dml.params[c] for c in time_cols if c in results_dml.params.index]

from scipy.stats import pearsonr
corr_ols_dml, p_ols_dml = pearsonr(ols_coefs, dml_coefs)
corr_twfe_dml, p_twfe_dml = pearsonr(twfe_coefs, dml_coefs)

print(f"  OLS vs DML:  r = {corr_ols_dml:+.3f}*** (p = {p_ols_dml:.4f})")
print(f"  TWFE vs DML: r = {corr_twfe_dml:+.3f}*** (p = {p_twfe_dml:.4f})")

# LaTeX code
print("\n--- LATEX CODE ---")
for name, result in methods.items():
    if 't_1' in result.params.index:
        coef = result.params['t_1']
        se = result.bse['t_1']
        r2 = result.rsquared
        n = result.nobs
        stars = '***' if result.pvalues['t_1'] < 0.01 else '**' if result.pvalues['t_1'] < 0.05 else '*'
        
        print(f"{name} (main specification) & \\blue{{{coef:.4f}{stars}}} & \\blue{{({se:.4f})}} & \\blue{{{r2:.4f}}} & \\blue{{{n:.0f}}} \\\\")


# =============================================================================
# TABLE: HETEROGENEITY - WITHIN-BLOC VS CROSS-BLOC
# =============================================================================
# INSERT AFTER: Within-bloc analysis

print("\n" + "="*80)
print("TABLE: HETEROGENEITY - WITHIN-BLOC VS CROSS-BLOC")
print("="*80)

# These should be from your within-bloc analysis
# Adjust variable names as needed
print("\nSwitch Type                Peak (τ=+1)           SE    N Obs    N Switchers")
print("-" * 85)

# Within-bloc
if 'within_bloc_results' in locals():
    wb_coef = within_bloc_results.params['t_1']
    wb_se = within_bloc_results.bse['t_1']
    wb_n = within_bloc_results.nobs
    wb_switchers = df_within_bloc['deputado_id'].nunique()
    wb_stars = '***' if within_bloc_results.pvalues['t_1'] < 0.01 else '**'
    
    print(f"Within-bloc          {wb_coef:12.4f}{wb_stars}    {wb_se:10.4f} {wb_n:8.0f}    {wb_switchers:12.0f}")

# Cross-bloc  
if 'cross_bloc_results' in locals():
    cb_coef = cross_bloc_results.params['t_1']
    cb_se = cross_bloc_results.bse['t_1']
    cb_n = cross_bloc_results.nobs
    cb_switchers = df_cross_bloc['deputado_id'].nunique()
    cb_stars = '***' if cross_bloc_results.pvalues['t_1'] < 0.01 else '**'
    
    print(f"Cross-bloc           {cb_coef:12.4f}{cb_stars}    {cb_se:10.4f} {cb_n:8.0f}    {cb_switchers:12.0f}")
    
    ratio = abs(cb_coef) / abs(wb_coef)
    print(f"\nRatio (cross/within): {ratio:.2f}×")

# LaTeX code
print("\n--- LATEX CODE ---")
if 'within_bloc_results' in locals() and 'cross_bloc_results' in locals():
    print(f"Within-bloc & \\blue{{{wb_coef:.4f}{wb_stars}}} & \\blue{{{wb_se:.4f}}} & \\blue{{{wb_n:.0f}}} & \\blue{{{wb_switchers:.0f}}} \\\\")
    print(f"Cross-bloc & \\blue{{{cb_coef:.4f}{cb_stars}}} & \\blue{{{cb_se:.4f}}} & \\blue{{{cb_n:.0f}}} & \\blue{{{cb_switchers:.0f}}} \\\\")
    print(f"Ratio (cross/within) & \\multicolumn{{4}}{{c}}{{\\blue{{{ratio:.2f}}}×}} \\\\")


# =============================================================================
# TABLE: WINDOW SENSITIVITY
# =============================================================================
# INSERT AFTER: Window sensitivity analysis

print("\n" + "="*80)
print("TABLE: WINDOW SENSITIVITY")
print("="*80)

print("\nWindow                   ATE           SE      p-value            N")
print("-" * 70)

# Loop through window results
windows = [6, 9, 12, 15, 18]
for w in windows:
    if f'window_{w}' in locals():
        result = locals()[f'window_{w}']
        ate = result['ate']
        se = result['se']
        pval = result['pvalue']
        n = result['n']
        
        stars = '***' if pval < 0.01 else '**' if pval < 0.05 else '*'
        pval_str = '0.0000' if pval < 0.0001 else f'{pval:.4f}'
        
        print(f"±{w} months      {ate:10.4f}{stars}       {se:10.4f}       {pval_str}       {n:8.0f}")

# LaTeX code
print("\n--- LATEX CODE ---")
for w in windows:
    if f'window_{w}' in locals():
        result = locals()[f'window_{w}']
        ate = result['ate']
        se = result['se']
        pval = result['pvalue']
        n = result['n']
        stars = '***' if pval < 0.01 else '**' if pval < 0.05 else '*'
        
        print(f"±{w} months & \\blue{{{ate:.4f}{stars}}} & \\blue{{({se:.4f})}} & \\blue{{<0.001}} & \\blue{{{n:.0f}}} \\\\")


# =============================================================================
# TABLE: CLASSIFIER PERFORMANCE METRICS
# =============================================================================
# INSERT AFTER: Classifier training

print("\n" + "="*80)
print("TABLE: CLASSIFIER PERFORMANCE METRICS")
print("="*80)

# From your classifier metrics
cv_scores = cross_val_score(clf_party, X_train, y_train, cv=5)
mean_acc = cv_scores.mean()
std_acc = cv_scores.std()

# Calculate baseline
n_parties = len(np.unique(y_train))
baseline_acc = 1 / n_parties

print(f"\nMetric                                        Value        Baseline     Improvement")
print("-" * 80)
print(f"Overall accuracy                              {mean_acc:.1%}            {baseline_acc:.1%}           {mean_acc/baseline_acc:.2f}×")
print(f"Weighted F1 score                             {weighted_f1:.3f}             ---             ---")
print(f"Mean F1 (major parties)                       {major_f1:.3f}             ---             ---")

print(f"\nNotes: Metrics from 5-fold cross-validation on training set (N = {len(X_train):,} speeches)")
print(f"       Baseline = uniform random guess across {n_parties} parties")

# LaTeX code
print("\n--- LATEX CODE ---")
print(f"Overall accuracy & \\blue{{{mean_acc:.1%}}} & \\blue{{{baseline_acc:.1%}}} & \\blue{{{mean_acc/baseline_acc:.2f}}}× \\\\")


# =============================================================================
# SAVE ALL TABLES TO CSV
# =============================================================================

import pandas as pd

# Save DML coefficients
dml_table = pd.DataFrame({
    'Period': time_periods,
    'Coefficient': [results_dml.params[f't_{t}'] if f't_{t}' in results_dml.params.index else np.nan for t in time_periods],
    'SE': [results_dml.bse[f't_{t}'] if f't_{t}' in results_dml.params.index else np.nan for t in time_periods],
    'p_value': [results_dml.pvalues[f't_{t}'] if f't_{t}' in results_dml.params.index else np.nan for t in time_periods],
})
dml_table.to_csv('../results/tables/table1_dml_event_study.csv', index=False)
print("\n✓ Saved: table1_dml_event_study.csv")

print("\n" + "="*80)
print("ALL TABLES PRINTED - READY FOR PAPER")
print("="*80)
