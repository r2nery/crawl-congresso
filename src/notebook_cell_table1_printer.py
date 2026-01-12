# =============================================================================
# PRINT TABLE 1: DML EVENT STUDY (LATEX-READY FORMAT)
# =============================================================================
# INSERT THIS CELL AFTER THE PRE-TRENDS F-TEST (after cell 16)

print("\n" + "="*80)
print("TABLE 1: DML EVENT STUDY COEFFICIENTS (LATEX-READY)")
print("="*80)

# Extract coefficients for all time periods
time_periods = [-6, -5, -4, -3, -2, 1, 2, 3, 4, 5, 6]

print("\n--- COEFFICIENTS FOR LATEX TABLE ---")
print("Period | Coef      | SE       | p-value  | CI Lower  | CI Upper")
print("-" * 70)

for t in time_periods:
    col_name = f't_{t}'
    if col_name in results_dml.params.index:
        coef = results_dml.params[col_name]
        se = results_dml.bse[col_name]
        pval = results_dml.pvalues[col_name]
        ci = results_dml.conf_int().loc[col_name]
        
        # Format for significance stars
        stars = '***' if pval < 0.01 else '**' if pval < 0.05 else '*' if pval < 0.1 else ''
        
        print(f"τ = {t:+2d} | {coef:9.4f}{stars:3s} | {se:8.4f} | {pval:8.3f} | {ci[0]:9.4f} | {ci[1]:9.4f}")

# Print summary statistics
print("\n--- SUMMARY STATISTICS ---")
print(f"N (speech-events):   {len(df_es):,}")
print(f"N (deputies):        {df_es['deputado_id'].nunique():,}")
print(f"N (clusters):        {df_es['deputado_id'].nunique():,}")
print(f"R²:                  {results_dml.rsquared:.4f}")

# Print pre-trends test - CORRECTED VERSION
print("\n--- PRE-TRENDS TEST ---")
pre_period_cols = [f't_{t}' for t in [-6, -5, -4, -3, -2] if f't_{t}' in results_dml.params.index]

# Get coefficients and covariance matrix for pre-period
pre_coefs = results_dml.params[pre_period_cols].values
pre_cov = results_dml.cov_params().loc[pre_period_cols, pre_period_cols].values

# F-statistic for joint test that all pre-period coefficients = 0
# F = (R'R)^{-1} * R' * theta / k
# where k = number of restrictions
k = len(pre_period_cols)
f_stat = (pre_coefs @ np.linalg.inv(pre_cov) @ pre_coefs) / k
df_denom = results_dml.df_resid

# p-value from F-distribution
from scipy.stats import f as f_dist
f_pval = 1 - f_dist.cdf(f_stat, k, df_denom)

print(f"F-statistic:         {f_stat:.3f}")
print(f"p-value:             {f_pval:.4f}")
print(f"Degrees of freedom:  F({k}, {df_denom})")

print("\n" + "="*80)
print("LATEX CODE (copy-paste ready):")
print("="*80)
print()

# Generate LaTeX table rows
for t in time_periods:
    col_name = f't_{t}'
    if col_name in results_dml.params.index:
        coef = results_dml.params[col_name]
        se = results_dml.bse[col_name]
        pval = results_dml.pvalues[col_name]
        ci = results_dml.conf_int().loc[col_name]
        
        stars = '***' if pval < 0.01 else '**' if pval < 0.05 else '*' if pval < 0.1 else ''
        
        # Format p-value
        pval_str = '<0.001' if pval < 0.001 else f'{pval:.3f}'
        
        # Add + sign for positive coefficients
        coef_str = f'+{coef:.4f}' if coef > 0 else f'{coef:.4f}'
        
        latex_row = f"$\\tau = {t:+d}$ & \\blue{{{coef_str}{stars}}} & \\blue{{{se:.4f}}} & \\blue{{{pval_str}}} & \\blue{{{ci[0]:.3f}}} & \\blue{{{ci[1]:.3f}}} \\\\"
        print(latex_row)

# Print table footer
print()
print(f"\\multicolumn{{6}}{{l}}{{N (speech-events) = \\blue{{{len(df_es):,}}}, \\quad N (deputies) = \\blue{{{df_es['deputado_id'].nunique():,}}}, \\quad N (clusters) = \\blue{{{df_es['deputado_id'].nunique():,}}}}} \\\\")
print(f"\\multicolumn{{6}}{{l}}{{R² = \\blue{{{results_dml.rsquared:.4f}}}, \\quad Joint pre-trend test: F(\\blue{{{k}}}, \\blue{{{df_denom}}}) = \\blue{{{f_stat:.3f}}}, p = \\blue{{{f_pval:.3f}}}}} \\\\")

print("\n" + "="*80)
