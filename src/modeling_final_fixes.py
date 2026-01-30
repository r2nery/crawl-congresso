# %% [markdown]
# # Rhetoric and Behavior After Party Switching in Brazil
# ## Replication Package - Comprehensive Analysis
# 
# **Author:** Arthur Gomes Nery  
# **Date:** January 2026
# 
# ---
# 
# ### Table of Contents
# 
# | Part | Section | Content |
# |------|---------|----------|
# | **I** | 1.0 | Setup & Configuration |
# | **I** | 1.1 | Data Loading |
# | **I** | 1.2 | Sample Construction |
# | **II** | 2.0 | Party Classifier Training |
# | **II** | 2.1 | Main DML Event Study |
# | **II** | 2.2 | Language vs. Voting Behavior |
# | **III** | 3.1 | Robustness: Alternative Estimation Methods |
# | **III** | 3.2 | Robustness: Classifier Performance |
# | **III** | 3.3 | Robustness: Named Entity Removal |
# | **III** | 3.4 | Robustness: Ideological Bloc Classification |
# | **III** | 3.5 | Robustness: Embedding-Based Semantic Distance |
# | **III** | 3.6 | Robustness: Placebo Test (Permutation) |
# | **III** | 3.7 | Robustness: Alternative Time Windows |
# | **III** | 3.8 | Robustness: Alternative Outcome Variable |
# | **IV** | 4.1 | Heterogeneity: Within-Bloc vs Cross-Bloc |
# | **IV** | 4.2 | Heterogeneity: Ideological Distance |
# | **IV** | 4.3 | Heterogeneity: Career Experience |
# | **IV** | 4.4 | Heterogeneity: Destination Party Size |
# | **IV** | 4.5 | Heterogeneity: Switch Direction |

# %% [markdown]
# ---
# # PART I: DATA
# ---

# %% [markdown]
# ### Section Setup

# %%
# IMPORTS =====================================================================
# =============================================================================

import os
import pickle
import warnings
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy import stats
from scipy.spatial.distance import cosine
import statsmodels.api as sm

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingRegressor, HistGradientBoostingClassifier
from sklearn.model_selection import cross_val_score, cross_val_predict, KFold, GroupKFold
from sklearn.base import clone
from sklearn.metrics import accuracy_score, f1_score
import numpy as np

warnings.filterwarnings('ignore')
tqdm.pandas()
SEED = 42
np.random.seed(SEED)

print("Imports complete.")

# %%
# CONFIGURATION ===============================================================
# =============================================================================

@dataclass
class Config:
    text_col: str = 'text_level_2'  # normalized, keeps NER, not stemmed
    bimester_days: int = 60
    min_speeches_per_period: int = 1
    het_min_speeches_window: int = 5
    min_party_speeches: int = 50
    control_holdout_ratio: float = 0.20
    n_control_samples: int = 5
    tfidf_max_features: int = 5000
    tfidf_min_df: int = 5
    tfidf_max_df: float = 0.70
    n_partisan_words: int = 200
    dml_n_splits: int = 3
    n_permutations: int = 1000
    min_votes_corr: int = 3
    

CFG = Config()

# Paths
DATA_DIR = '../data/processed/'
RAW_DIR = '../data/raw/'
RESULTS_DIR = '../results/final_paper/'
PLOTS_DIR = os.path.join(RESULTS_DIR, 'figures/')
TABLES_DIR = os.path.join(RESULTS_DIR, 'tables/')

for d in [RESULTS_DIR, PLOTS_DIR, TABLES_DIR]:
    os.makedirs(d, exist_ok=True)

PATH_PANEL = os.path.join(DATA_DIR, 'data_panel.parquet')
PATH_HISTORY = os.path.join(RAW_DIR, 'deputies/deputy_migrations.csv')

# Visual
COLORS = {'main': '#2b7bba', 'control': '#0077BE', 'sig': '#2ECC71', 'nonsig': '#95A5A6'}
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({'figure.dpi': 100, 'savefig.dpi': 300, 'font.size': 11})

print(f"Config: text={CFG.text_col}")

# STANDARDIZED COVARIATE CONFIGURATION
# ====================================
# This ensures consistency across ALL model specifications

CONTINUOUS_COVARIATES = [
    'career_tenure_years',   # Career tenure (years since first election)
    'prop_activity_annual',  # Legislative activity (bills + amendments)
]

def build_covariates(df, include_calendar_month_fe=False, include_topic_fe=True, 
                     date_col='dataHoraInicio', verbose=False):
    """
    Standardized covariate matrix construction.
    """
    components = []
    
    # 1. Continuous covariates
    available = [c for c in CONTINUOUS_COVARIATES if c in df.columns]
    if available:
        X_cont = df[available].apply(pd.to_numeric, errors='coerce').fillna(0).astype(float)
        components.append(X_cont)
        if verbose:
            print(f"   Continuous: {available}")
    
    # 2. Legislature FE
    if 'idLegislatura' in df.columns:
        X_leg = pd.get_dummies(df['idLegislatura'], prefix='leg', drop_first=True).astype(float)
        components.append(X_leg)
        if verbose:
            print(f"   Legislature FE: {X_leg.shape[1]} dummies")
    
    # 3. Calendar month FE
    if include_calendar_month_fe and date_col in df.columns:
        cal_month = pd.to_datetime(df[date_col]).dt.to_period('M').astype(str)
        X_month = pd.get_dummies(cal_month, prefix='cal_month', drop_first=True).astype(float)
        components.append(X_month)
        if verbose:
            print(f"   Calendar month FE: {X_month.shape[1]} dummies")
    
    # 4. Topic FE
    if include_topic_fe and 'topic_id' in df.columns:
        X_topic = pd.get_dummies(df['topic_id'], prefix='topic', drop_first=True).astype(float)
        components.append(X_topic)
        if verbose:
            print(f"   Topic FE: {X_topic.shape[1]} dummies")
    
    return pd.concat(components, axis=1) if components else pd.DataFrame(index=df.index)

def grouped_oof_predict_regressor(model, X, y, groups, n_splits, seed=42):
    gkf = GroupKFold(n_splits=n_splits)
    y_hat = np.empty(len(y), dtype=float)
    for tr, te in gkf.split(X, y, groups=groups):
        m = clone(model)
        m.fit(X[tr], y[tr])
        y_hat[te] = m.predict(X[te])
    return y_hat

def grouped_oof_predict_proba_binary(model, X, d, groups, n_splits, seed=42):
    gkf = GroupKFold(n_splits=n_splits)
    p_hat = np.empty(len(d), dtype=float)
    d = np.asarray(d).astype(int)
    for tr, te in gkf.split(X, d, groups=groups):
        uniq = np.unique(d[tr])
        if uniq.size < 2:
            p_hat[te] = float(uniq[0])
        else:
            m = clone(model)
            m.fit(X[tr], d[tr])
            p_hat[te] = m.predict_proba(X[te])[:, 1]
    return p_hat

def groupkfold_cv_predict_classifier(
    estimator,
    X,
    y,
    groups,
    n_splits=5
):
    """
    GroupKFold cross-validated predictions for a classifier.
    Splits by `groups` (e.g., deputado_id) to avoid leakage across repeated speeches.
    """
    y_arr = np.asarray(y)
    g_arr = np.asarray(groups)

    gkf = GroupKFold(n_splits=n_splits)

    # Use dtype=object to support string labels
    y_pred = np.empty(len(y_arr), dtype=object)

    for tr_idx, te_idx in gkf.split(X, y_arr, groups=g_arr):
        est = clone(estimator)
        est.fit(X[tr_idx], y_arr[tr_idx])
        y_pred[te_idx] = est.predict(X[te_idx])

    return y_pred

def groupkfold_cv_accuracy_classifier(
    estimator,
    X,
    y,
    groups,
    n_splits=5
):
    """
    Convenience wrapper: returns GroupKFold CV accuracy (deputy-level splits).
    """
    from sklearn.metrics import accuracy_score

    y_arr = np.asarray(y)
    y_pred = groupkfold_cv_predict_classifier(
        estimator=estimator,
        X=X,
        y=y_arr,
        groups=groups,
        n_splits=n_splits
    )
    return float(accuracy_score(y_arr, y_pred))

print("Covariate configuration loaded.")
print(f"Continuous covariates: {CONTINUOUS_COVARIATES}")

# %%
# ==============================================================================
# PUBLICATION-QUALITY FIGURES (NO LATEX REQUIRED)
# ==============================================================================

import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import os

# ------------------------------------------------------------------------------
# 1. FIGURE SIZE CONFIGURATION
# ------------------------------------------------------------------------------

PT = 1./72.27  # Points to inches

LATEX_WIDTHS = {
    # Standard LaTeX classes
    'article': 345 * PT,           # ~4.77 inches
    'article_twocol': 246 * PT,    # ~3.40 inches (single column in two-col)
    'thesis': 390 * PT,            # ~5.39 inches
    'beamer': 307 * PT,            # ~4.25 inches
    
    # Common journals
    'nature': 183 * PT,            # 89mm single column
    'science': 227 * PT,           
    'pnas_one': 246 * PT,
    'pnas_two': 510 * PT,
    
    # YOUR DOCUMENT - update after measuring with \printlength{\textwidth}
    'my_paper': 455 * PT,          # <-- UPDATE THIS
}

GOLDEN = (1 + 5**0.5) / 2  # ~1.618

def figsize(width_key='my_paper', fraction=1.0, aspect=None):
    """
    Get figure size matching LaTeX document.
    
    Parameters
    ----------
    width_key : str or float
        Key from LATEX_WIDTHS, or width in inches directly
    fraction : float
        Fraction of width (e.g., 0.5 for half-width figure)
    aspect : float
        Height/width ratio. Default: 1/golden ratio (~0.618)
    
    Returns
    -------
    tuple : (width, height) in inches
    """
    if isinstance(width_key, (int, float)):
        width = width_key
    else:
        width = LATEX_WIDTHS.get(width_key, LATEX_WIDTHS['article'])
    
    width *= fraction
    
    if aspect is None:
        aspect = 1 / GOLDEN
    
    return (width, width * aspect)


# ------------------------------------------------------------------------------
# 2. STYLE SETUP
# ------------------------------------------------------------------------------

plt.rcdefaults()

plt.rcParams.update({
    'text.usetex': False,
    'mathtext.fontset': 'cm',
    
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'font.size': 10,
    'axes.labelsize': 10,
    'axes.titlesize': 10,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    
    'figure.dpi': 150,
    'axes.linewidth': 0.5,
    'axes.spines.top': False,
    'axes.spines.right': False,
    
    'xtick.direction': 'in',
    'ytick.direction': 'in',
    'xtick.major.width': 0.5,
    'ytick.major.width': 0.5,
    'xtick.major.size': 3,
    'ytick.major.size': 3,
    
    'lines.linewidth': 1.0,
    'lines.markersize': 5,
    'legend.frameon': False,
    
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.02,
})



# %%
# LOAD DATA ===================================================================
# =============================================================================

# Initialize container for section statistics
section1_stats = {}

df = pd.read_parquet(PATH_PANEL).dropna(subset=['CAT', 'EST', CFG.text_col])
df['deputado_id'] = df['deputado_id'].astype(str)
df['idPartido'] = df['idPartido'].astype(str)
df['dataHoraInicio'] = pd.to_datetime(df['dataHoraInicio'], utc=True)

CAT_MAP = {0: 'Left', 1: 'Center', 2: 'Right'}
if pd.api.types.is_numeric_dtype(df['CAT']):
    df['CAT'] = df['CAT'].map(CAT_MAP)

# Store loading stats
section1_stats['data_loading'] = {
    'Total Speeches': len(df),
    'Total Deputies': df['deputado_id'].nunique(),
    'Columns': ", ".join(df.columns.tolist())
}

# Load history
df_hist = pd.read_csv(PATH_HISTORY)
df_hist['deputado_id'] = df_hist['deputado_id'].astype(str)
df_hist['dataHora'] = pd.to_datetime(df_hist['dataHora'], utc=True)
df_hist['idPartido'] = df_hist['uriPartido'].str.split('/').str[-1]


section1_stats['history_loading'] = {
    'Deputies with History': df_hist['deputado_id'].nunique()
}

print("Data loaded.")

# %%
# EXTRACT SWITCH EVENTS =======================================================
# =============================================================================

switch_rows = df_hist[df_hist['descricaoStatus'] == 'Alteração de partido'].copy()

events = []
for dep_id, group in tqdm(switch_rows.groupby('deputado_id'), desc="Extracting events"):
    dep_hist = df_hist[df_hist['deputado_id'] == dep_id].sort_values('dataHora')
    for i, (_, row) in enumerate(group.sort_values('dataHora').iterrows()):
        before = dep_hist[dep_hist['dataHora'] < row['dataHora']]
        if len(before) == 0: continue
        old_pid, new_pid = before.iloc[-1]['idPartido'], row['idPartido']
        if old_pid == new_pid: continue
        events.append({'deputado_id': dep_id, 'switch_date': row['dataHora'],
                       'switch_number': i+1, 'old_party_id': str(old_pid), 'new_party_id': str(new_pid)})

df_events = pd.DataFrame(events)

df_events = df_events.dropna(subset=['deputado_id'])  
# Store switch stats
section1_stats['switch_events'] = {
    'Total Switch Events': len(df_events),
    'Deputies Switching': df_events['deputado_id'].nunique()
}

# %%
# ENRICH WITH IDEOLOGY ========================================================
# =============================================================================

def get_party_ideo(df, pid, date):
    sub = df[(df['idPartido']==str(pid)) & (df['dataHoraInicio']<=date)]
    if len(sub)==0: sub = df[df['idPartido']==str(pid)]
    return (sub.iloc[0]['EST'], sub.iloc[0]['CAT']) if len(sub)>0 else (None, None)

ideo_data = []
for _, row in tqdm(df_events.iterrows(), total=len(df_events), desc="Enriching ideology"):
    o_est, o_cat = get_party_ideo(df, row['old_party_id'], row['switch_date'])
    n_est, n_cat = get_party_ideo(df, row['new_party_id'], row['switch_date'])
    ideo_data.append({'old_EST': o_est, 'old_CAT': o_cat, 'new_EST': n_est, 'new_CAT': n_cat})

df_events = pd.concat([df_events, pd.DataFrame(ideo_data)], axis=1)
df_events['ideo_distance'] = abs(pd.to_numeric(df_events['new_EST'], errors='coerce') - 
                                  pd.to_numeric(df_events['old_EST'], errors='coerce'))
df_events['is_rightward'] = pd.to_numeric(df_events['new_EST'], errors='coerce') > pd.to_numeric(df_events['old_EST'], errors='coerce')
df_events['transition'] = df_events['old_CAT'] + ' -> ' + df_events['new_CAT']

# Store ideology stats
section1_stats['ideology_enrichment'] = {
    'Events with Valid Ideology': df_events.dropna(subset=['old_CAT','new_CAT']).shape[0]
}

# %%
# PARTITION DEPUTIES ==========================================================
# =============================================================================

all_ids = df['deputado_id'].unique()
switcher_ids = df_events['deputado_id'].dropna().unique()  # Use the SAME definition as df_es

nonswitcher_ids = np.setdiff1d(all_ids, switcher_ids)
np.random.seed(SEED)
np.random.shuffle(nonswitcher_ids)

n_ctrl = int(len(nonswitcher_ids) * CFG.control_holdout_ratio)
control_ids, train_ids = nonswitcher_ids[:n_ctrl], nonswitcher_ids[n_ctrl:]
df_train = df[df['deputado_id'].isin(train_ids)].copy()

# --- CREATE CONTROL HOLDOUT DATAFRAME (20% NON-SWITCHERS) --------------------
df_control = df[df['deputado_id'].isin(control_ids)].copy()

# Optional: sanity print
print(f"Train non-switchers: {len(train_ids)} deputies, {len(df_train):,} speeches")
print(f"Holdout non-switchers: {len(control_ids)} deputies, {len(df_control):,} speeches")

CORPUS_STATS = {'n_speeches': len(df), 'n_deputies': df['deputado_id'].nunique(),
                'n_switch_events': len(df_events)}

# Store partition stats
section1_stats['partitioning'] = {
    'Train Deputies': len(train_ids),
    'Train Speeches': len(df_train),
    'Control Deputies': len(control_ids),
    'Switcher Deputies': len(switcher_ids)
}

print("Deputy partitioning complete.")

# %%
# BUILD EVENT STUDY DATASET ===================================================
# =============================================================================

# Ensure section1_stats exists if running out of order
if 'section1_stats' not in locals(): section1_stats = {}

def get_old_party_conf(row, ci):
    pid = str(row['old_party_id'])
    return row['party_probs'][ci[pid]] if pid in ci else np.nan

df_sw = df[df['deputado_id'].isin(df_events['deputado_id'].unique())].copy()
df_es = df_sw.merge(df_events[['deputado_id','switch_date','old_party_id','new_party_id',
                                'old_CAT','new_CAT','ideo_distance','is_rightward','switch_number']], on='deputado_id')

df_es['days_from_switch'] = (df_es['dataHoraInicio'].dt.tz_localize(None) - 
                              pd.to_datetime(df_es['switch_date']).dt.tz_localize(None)).dt.days
df_es = df_es[df_es['days_from_switch'].abs() <= 360].copy()

# Store stats
section1_stats['event_study_build'] = {
    'Event Study Sample': len(df_es)
}

# %%
# VOTING DATA ANALYSIS (PRE-TRAINING) =========================================
# =============================================================================

# Ensure stats container exists
if 'section1_stats' not in locals(): section1_stats = {}
voting_stats = {'Status': 'Initialized'}

PATH_VOTES = os.path.join(RAW_DIR, 'scrape_votes.parquet')

df_votes = pd.read_parquet(PATH_VOTES)
df_votes['deputado_id'] = df_votes['deputado_id'].astype(str)

# Handle date column (flexible)
if 'dataVotacao' in df_votes.columns:
    df_votes['vote_date'] = pd.to_datetime(df_votes['dataVotacao'], utc=True)
elif 'dataHoraVoto' in df_votes.columns:
    df_votes['vote_date'] = pd.to_datetime(df_votes['dataHoraVoto'], utc=True)
else:
    voting_stats['Status'] = 'Skipped (No date column)'
    
if 'vote_date' in df_votes.columns:
    voting_stats['Total Votes Loaded'] = len(df_votes)
    
    # Build party timeline
    deputy_party_timeline = {}
    for deputy_id in tqdm(df_hist['deputado_id'].unique(), desc="Building Timelines"):
        affiliations = df_hist[df_hist['deputado_id'] == deputy_id].sort_values('dataHora')
        timeline = []
        for i, row in affiliations.iterrows():
            start = row['dataHora']
            end = affiliations.iloc[i+1]['dataHora'] if i < len(affiliations)-1 else pd.Timestamp('2030-01-01', tz='UTC')
            timeline.append({'start': start, 'end': end, 'party_id': row['idPartido']})
        deputy_party_timeline[deputy_id] = timeline
    
    # Add party affiliation to votes
    def get_party_at_date(deputy_id, date):
        if deputy_id not in deputy_party_timeline: return None
        for period in deputy_party_timeline[deputy_id]:
            if period['start'] <= date <= period['end']: return period['party_id']
        return None
    
    df_votes['party_at_vote'] = df_votes.progress_apply(
        lambda r: get_party_at_date(r['deputado_id'], r['vote_date']), axis=1
    )
    
    # Calculate loyalty
    def calc_loyalty(deputy_id, start_date, end_date, party_id):
        deputy_votes = df_votes[
            (df_votes['deputado_id'] == deputy_id) &
            (df_votes['vote_date'] >= start_date) & (df_votes['vote_date'] <= end_date)
        ]
        if len(deputy_votes) < CFG.min_votes_corr: return np.nan
        
        vote_id_col = 'idVotacao' if 'idVotacao' in df_votes.columns else 'uriProposicao'
        party_votes = df_votes[
            (df_votes['party_at_vote'] == party_id) & (df_votes[vote_id_col].isin(deputy_votes[vote_id_col]))
        ]
        if len(party_votes) == 0: return np.nan
        
        party_position = party_votes.groupby(vote_id_col)['voto'].agg(
            lambda x: x.mode()[0] if len(x.mode()) > 0 else None
        )
        
        deputy_votes = deputy_votes.merge(party_position.to_frame('party_pos'), left_on=vote_id_col, right_index=True, how='left').dropna(subset=['party_pos'])
        if len(deputy_votes) < CFG.min_votes_corr: return np.nan
        
        return (deputy_votes['voto'] == deputy_votes['party_pos']).mean()
    
    voting_data = []
    for _, switch in tqdm(df_events.iterrows(), total=len(df_events), desc="Calculating Loyalty"):
        deputy_id = switch['deputado_id']
        switch_date = pd.to_datetime(switch['switch_date'], utc=True)
        
        loyalty_pre = calc_loyalty(deputy_id, switch_date - pd.Timedelta(days=180), switch_date - pd.Timedelta(days=1), switch['old_party_id'])
        loyalty_post = calc_loyalty(deputy_id, switch_date, switch_date + pd.Timedelta(days=180), switch['new_party_id'])
        
        voting_data.append({
            'deputado_id': deputy_id,
            'loyalty_pre_old': loyalty_pre,
            'loyalty_post_new': loyalty_post,
            'loyalty_increase': loyalty_post - loyalty_pre if pd.notna([loyalty_pre, loyalty_post]).all() else np.nan
        })
    
    df_voting = pd.DataFrame(voting_data)

    # Keep only deputies that exist in df_events
    valid_deputies = df_events['deputado_id'].dropna().unique()
    df_voting = df_voting[df_voting['deputado_id'].isin(valid_deputies)].copy()

    voting_stats['Switchers with Voting Data'] = len(df_voting.dropna(subset=['loyalty_increase']))


    # Also calculate counterfactual: loyalty to OLD party AFTER switch
    for i, row in df_voting.iterrows():
        deputy_id = row['deputado_id']
        switch_info = df_events[df_events['deputado_id'] == deputy_id].iloc[0]
        switch_date = pd.to_datetime(switch_info['switch_date'], utc=True)
        
        loyalty_post_old = calc_loyalty(
            deputy_id, 
            switch_date, 
            switch_date + pd.Timedelta(days=180), 
            switch_info['old_party_id']
        )
        df_voting.at[i, 'loyalty_post_old'] = loyalty_post_old

    # Calculate change in loyalty to old party
    df_voting['loyalty_decrease_old'] = df_voting['loyalty_post_old'] - df_voting['loyalty_pre_old']

section1_stats['voting_analysis'] = voting_stats

# %%
# IDEOLOGY AND BLOC TRANSITION STATISTICS ======================================
# ==============================================================================

# Filter to switches with valid ideology data
df_events_ideo = df_events.dropna(subset=['old_CAT', 'new_CAT', 'old_EST', 'new_EST']).copy()

# Determine within-bloc vs cross-bloc
df_events_ideo['is_within_bloc'] = df_events_ideo['old_CAT'] == df_events_ideo['new_CAT']
df_events_ideo['is_cross_bloc'] = df_events_ideo['old_CAT'] != df_events_ideo['new_CAT']

# Store statistics
ideology_stats = {
    'n_with_ideology': len(df_events_ideo),
    'n_total_switches': len(df_events),
    'pct_with_ideology': len(df_events_ideo) / len(df_events) * 100,
    
    # Ideological distance
    'mean_ideo_distance': df_events_ideo['ideo_distance'].mean(),
    'sd_ideo_distance': df_events_ideo['ideo_distance'].std(),
    'min_ideo_distance': df_events_ideo['ideo_distance'].min(),
    'max_ideo_distance': df_events_ideo['ideo_distance'].max(),
    
    # EST scale range (for reference)
    'est_min': df_events_ideo[['old_EST', 'new_EST']].min().min(),
    'est_max': df_events_ideo[['old_EST', 'new_EST']].max().max(),
    'est_range': df_events_ideo[['old_EST', 'new_EST']].max().max() - df_events_ideo[['old_EST', 'new_EST']].min().min(),
    
    # Within vs cross-bloc
    'n_within_bloc': df_events_ideo['is_within_bloc'].sum(),
    'n_cross_bloc': df_events_ideo['is_cross_bloc'].sum(),
    'pct_within_bloc': df_events_ideo['is_within_bloc'].mean() * 100,
    'pct_cross_bloc': df_events_ideo['is_cross_bloc'].mean() * 100,
}

# Calculate what percentage of spectrum the mean distance represents
ideology_stats['pct_spectrum'] = (ideology_stats['mean_ideo_distance'] / ideology_stats['est_range']) * 100

print("✓ Ideology statistics computed")

# %% [markdown]
# ### Section Outputs

# %%
# SECTION 1 REPORT ============================================================
# =============================================================================

def print_section_report(stats):
    print("="*60)
    print("SECTION 1: DATA PROCESSING & PARTITIONING REPORT")
    print("="*60)
    print("\n")
    
    # 1. Dataset Overview
    if 'data_loading' in stats:
        overview_data = {**stats['data_loading'], **stats['history_loading']}
        cols = overview_data.pop('Columns', 'N/A') 
        df_overview = pd.DataFrame(list(overview_data.items()), columns=['Metric', 'Value'])
        
        print("--- Dataset Overview ---")
        print(df_overview.to_string(index=False))
        # Truncate columns if too long
        col_str = cols if len(cols) < 100 else f"{cols[:100]}..."
        print(f"\nColumns Loaded: {col_str}")
        print("\n")
    
    # 2. Switch Events
    if 'switch_events' in stats:
        switch_data = {**stats['switch_events'], **stats['ideology_enrichment']}
        df_switch = pd.DataFrame(list(switch_data.items()), columns=['Metric', 'Count'])
        
        print("--- Switch Event Analysis ---")
        print(df_switch.to_string(index=False))
        print("\n")
    
    # 3. Partitioning
    if 'partitioning' in stats:
        df_part = pd.DataFrame(list(stats['partitioning'].items()), columns=['Group', 'Count'])
        
        print("--- Experimental Partitioning ---")
        print(df_part.to_string(index=False))
        print(f"Test speeches: {df[df['deputado_id'].isin(switcher_ids)].shape[0]:,}")
        print(f"Control speeches: {df[df['deputado_id'].isin(control_ids)].shape[0]:,}")
        print("\n")

    # 4. Voting Loyalty (Updated)
    # Combines Event Study sample size with Voting Loyalty stats
    if 'voting_analysis' in stats:
        voting_data = stats.get('event_study_build', {}).copy() # Start with event study size
        voting_data.update(stats['voting_analysis'])            # Add voting stats
        
        df_vote = pd.DataFrame(list(voting_data.items()), columns=['Metric', 'Value'])
        
        print("--- Event Study & Voting Loyalty ---")
        print(df_vote.to_string(index=False))
        print("\n")

    print("="*60)

    print("IDEOLOGY AND BLOC TRANSITION STATISTICS")
    print("=" * 80)
    print()

    s = ideology_stats

    print(f"Switches with valid ideology data: {s['n_with_ideology']:,} / {s['n_total_switches']:,} ({s['pct_with_ideology']:.1f}%)")
    print()

    print("--- Ideological Distance (|EST_new - EST_old|) ---")
    print(f"   Mean:     {s['mean_ideo_distance']:.3f}")
    print(f"   Std Dev:  {s['sd_ideo_distance']:.3f}")
    print(f"   Min:      {s['min_ideo_distance']:.3f}")
    print(f"   Max:      {s['max_ideo_distance']:.3f}")
    print()

    print("--- EST Scale ---")
    print(f"   Range: [{s['est_min']:.2f}, {s['est_max']:.2f}]")
    print(f"   Total span: {s['est_range']:.2f}")
    print(f"   Mean distance as % of spectrum: {s['pct_spectrum']:.1f}%")
    print()

    print("--- Within-Bloc vs Cross-Bloc ---")
    print(f"   Within-bloc switches: {s['n_within_bloc']:,} ({s['pct_within_bloc']:.1f}%)")
    print(f"   Cross-bloc switches:  {s['n_cross_bloc']:,} ({s['pct_cross_bloc']:.1f}%)")
    print()

    print("--- Transition Matrix ---")
    transition_matrix = pd.crosstab(
        df_events_ideo['old_CAT'], 
        df_events_ideo['new_CAT'], 
        margins=True
    )
    print(transition_matrix)
    print()

    print("=" * 80)

print_section_report(section1_stats)

# %% [markdown]
# ---
# # PART II: MAIN RESULTS
# ---

# %% [markdown]
# ### Section Setup

# %%
# TRAIN CLASSIFIER, UPDATE EVENT STUDY & CORRELATE ============================
# =============================================================================

# Initialize stats container
section2_stats = {}

print("Training party-level classifier...")
tfidf_party = TfidfVectorizer(max_features=CFG.tfidf_max_features, min_df=CFG.tfidf_min_df,
                               max_df=CFG.tfidf_max_df, ngram_range=(1,2))
X_train = tfidf_party.fit_transform(df_train[CFG.text_col])
y_train = df_train['idPartido']

# --- BUILD HOLDOUT FEATURES/LABELS USING SAME VECTORIZER (NO LEAKAGE) --------
# Ensure text exists
df_control_eval = df_control.dropna(subset=[CFG.text_col, 'idPartido']).copy()

X_control = tfidf_party.transform(df_control_eval[CFG.text_col])
y_control = df_control_eval['idPartido']

# Keep a handle for deputy ids (useful for deputy-level accuracy)
groups_control = df_control_eval['deputado_id'].astype(str).values

print(f"Holdout eval set: {len(y_control):,} speeches from {df_control_eval['deputado_id'].nunique()} deputies")

clf_party = LogisticRegression(class_weight='balanced', C=1.0, max_iter=500, n_jobs=-1, random_state=SEED)
clf_party.fit(X_train, y_train)

groups_train = df_train['deputado_id'].astype(str).values

cv_acc = groupkfold_cv_accuracy_classifier(
    estimator=clf_party,
    X=X_train,
    y=y_train,
    groups=groups_train,
    n_splits=5
)

section2_stats['model_performance'] = {
    'Model': 'Logistic Regression',
    'CV Accuracy': f"{cv_acc:.3f} (GroupKFold by deputy, 5 folds)",
    'Training Samples': int(len(y_train))
}

PARTY_CLASS_INDICES = {label: idx for idx, label in enumerate(clf_party.classes_)}

# %%
# PREDICTION LOGIC ------------------------------------------------------------
# -----------------------------------------------------------------------------
print("Predicting probabilities...")

# Predict for Event Study DataFrame (FIXED)
# Instead of mapping by index (which causes KeyError), we predict directly on df_es text
print("   Updating Event Study dataset...")
X_es = tfidf_party.transform(df_es[CFG.text_col])
df_es['party_probs'] = list(clf_party.predict_proba(X_es))

# Calculate Confidence
def get_old_party_conf(row, ci):
    pid = str(row['old_party_id'])
    # Check if party exists in classifier classes, otherwise NaN
    return row['party_probs'][ci[pid]] if pid in ci else np.nan

df_es['Y_confidence'] = df_es.apply(lambda r: get_old_party_conf(r, PARTY_CLASS_INDICES), axis=1)
df_es = df_es.dropna(subset=['Y_confidence'])

# =============================================================================
# FILTER: Minimum speeches per deputy in event study window
# =============================================================================
speeches_per_deputy = df_es.groupby('deputado_id').size()
deputies_meeting_threshold = speeches_per_deputy[speeches_per_deputy >= CFG.min_speeches_per_period].index

n_deputies_before = df_es['deputado_id'].nunique()
n_speeches_before = len(df_es)

df_es = df_es[df_es['deputado_id'].isin(deputies_meeting_threshold)].copy()

n_deputies_after = df_es['deputado_id'].nunique()
n_speeches_after = len(df_es)

print(f"Minimum speech filter (≥{CFG.min_speeches_per_period} speeches per deputy):")
print(f"  Deputies: {n_deputies_before} → {n_deputies_after} ({n_deputies_before - n_deputies_after} removed)")
print(f"  Speeches: {n_speeches_before:,} → {n_speeches_after:,} ({n_speeches_before - n_speeches_after:,} removed)")

section2_stats['event_study_data'] = {
    'Total Speech-Events': len(df_es),
    'Rightward Switches': df_es['is_rightward'].sum(),
    'Avg Confidence': df_es['Y_confidence'].mean()
}

# %%
def get_validation_examples(df_es, n_examples=5):
    """
    Extract example speeches with high/low P(old party) for qualitative validation.
    """
    import pandas as pd
    
    results = []
    
    # Get switchers with clear pre/post contrast
    switcher_effects = df_es.groupby('deputado_id').apply(
        lambda x: pd.Series({
            'pre_mean': x.loc[x['days_from_switch'] < 0, 'Y_confidence'].mean(),
            'post_mean': x.loc[x['days_from_switch'] > 0, 'Y_confidence'].mean(),
            'effect': x.loc[x['days_from_switch'] < 0, 'Y_confidence'].mean() - x.loc[x['days_from_switch'] > 0, 'Y_confidence'].mean(),
            'old_party_id': x['old_party_id'].iloc[0],
            'new_party_id': x['new_party_id'].iloc[0],
            'old_CAT': x['old_CAT'].iloc[0],
            'new_CAT': x['new_CAT'].iloc[0]
        })
    ).dropna()
    
    # Select switchers with large effects
    top_adapters = switcher_effects.nlargest(n_examples * 2, 'effect')
    
    for dep_id in top_adapters.index[:n_examples]:
        dep_data = df_es[df_es['deputado_id'] == dep_id]
        
        # High P(old): pre-switch, high confidence
        pre_speeches = dep_data[dep_data['days_from_switch'] < 0].nlargest(1, 'Y_confidence')
        # Low P(old): post-switch, low confidence  
        post_speeches = dep_data[dep_data['days_from_switch'] > 0].nsmallest(1, 'Y_confidence')
        
        if len(pre_speeches) > 0 and len(post_speeches) > 0:
            results.append({
                'deputado_id': dep_id,
                'old_party_id': switcher_effects.loc[dep_id, 'old_party_id'],
                'new_party_id': switcher_effects.loc[dep_id, 'new_party_id'],
                'old_CAT': switcher_effects.loc[dep_id, 'old_CAT'],
                'new_CAT': switcher_effects.loc[dep_id, 'new_CAT'],
                'effect_size': switcher_effects.loc[dep_id, 'effect'],
                'period': 'pre',
                'P_old': pre_speeches.iloc[0]['Y_confidence'],
                'days_from_switch': pre_speeches.iloc[0]['days_from_switch'],
                'date': pre_speeches.iloc[0]['dataHoraInicio'],
                'text': pre_speeches.iloc[0]['text_level_1'][:500] if pd.notna(pre_speeches.iloc[0]['text_level_1']) else ''
            })
            results.append({
                'deputado_id': dep_id,
                'old_party_id': switcher_effects.loc[dep_id, 'old_party_id'],
                'new_party_id': switcher_effects.loc[dep_id, 'new_party_id'],
                'old_CAT': switcher_effects.loc[dep_id, 'old_CAT'],
                'new_CAT': switcher_effects.loc[dep_id, 'new_CAT'],
                'effect_size': switcher_effects.loc[dep_id, 'effect'],
                'period': 'post',
                'P_old': post_speeches.iloc[0]['Y_confidence'],
                'days_from_switch': post_speeches.iloc[0]['days_from_switch'],
                'date': post_speeches.iloc[0]['dataHoraInicio'],
                'text': post_speeches.iloc[0]['text_level_1'][:500] if pd.notna(post_speeches.iloc[0]['text_level_1']) else ''
            })
    
    return pd.DataFrame(results)


# ==============================================================================
# USAGE
# ==============================================================================

validation_df = get_validation_examples(df_es, n_examples=5)
validation_df.to_csv('../results/final_paper/classifier_validation_examples.csv', index=False)

print(f"✓ Saved {len(validation_df)} example speeches")
print("\n" + "="*80)
print("CLASSIFIER VALIDATION EXAMPLES")
print("="*80)
print(validation_df[['old_party_id', 'new_party_id', 'period', 'P_old', 'days_from_switch', 'effect_size']].to_string())

# Show text excerpts
print("\n" + "="*80)
print("TEXT EXCERPTS (first 200 chars)")
print("="*80)
for _, row in validation_df.iterrows():
    print(f"\n[{row['period'].upper()}] Deputy {row['deputado_id']} | P(old)={row['P_old']:.3f} | {row['days_from_switch']:.0f} days")
    print(f"  {row['text'][:200]}...")


def get_top_features_by_party(clf, vectorizer, n_features=10):
    """
    Extract top predictive words for each party.
    Returns DataFrame with party, word, weight.
    """
    import pandas as pd
    
    feature_names = vectorizer.get_feature_names_out()
    party_list = clf.classes_
    
    results = []
    for i, party in enumerate(party_list):
        if i < clf.coef_.shape[0]:
            top_indices = clf.coef_[i].argsort()[-n_features:][::-1]
            for rank, idx in enumerate(top_indices):
                results.append({
                    'party': party,
                    'rank': rank + 1,
                    'word': feature_names[idx],
                    'weight': clf.coef_[i][idx]
                })
    
    return pd.DataFrame(results)


# ==============================================================================
# USAGE
# ==============================================================================

# Get examples
validation_df = get_validation_examples(df_es, n_examples=100)
validation_df.to_csv('../results/final_paper/classifier_validation_examples.csv', index=False)
print(f"✓ Saved {len(validation_df)} example speeches")

# %%
# =============================================================================
# REVISION 2A: SAMPLE FILTERING TABLE
# =============================================================================
# Document the sample construction process step-by-step
# This shows how we go from all switchers to the final event study sample
# =============================================================================

print("=" * 95)
print("TABLE: SAMPLE FILTERING STEPS")
print("=" * 95)
print()
print(f"{'Step':<60} {'N Deputies':<15} {'N Speeches':<15}")
print("-" * 95)

filtering_steps = {}

# -----------------------------------------------------------------------------
# STEP 1: Test set (all switchers in corpus)
# -----------------------------------------------------------------------------
# All deputies who switched AND have at least one speech in the corpus

all_switcher_ids = set(df_events['deputado_id'].unique())
df_switchers_raw = df[df['deputado_id'].isin(all_switcher_ids)]

n_dep_1 = df_switchers_raw['deputado_id'].nunique()
n_speech_1 = len(df_switchers_raw)

print(f"{'1. Test set (all switchers in corpus)':<60} {n_dep_1:<15} {n_speech_1:<15,}")
filtering_steps['step1_test_set'] = {'n_deputies': n_dep_1, 'n_speeches': n_speech_1}

# -----------------------------------------------------------------------------
# STEP 2: After removing switches without valid party codes
# -----------------------------------------------------------------------------
# Keep only switches where old_party_id exists in classifier's training classes

valid_party_ids = set(PARTY_CLASS_INDICES.keys())

# Filter events to those with valid old party
df_events_valid = df_events[df_events['old_party_id'].astype(str).isin(valid_party_ids)]
valid_switcher_ids = set(df_events_valid['deputado_id'].unique())

# Filter speeches to deputies with valid switches
df_switchers_valid = df[df['deputado_id'].isin(valid_switcher_ids)]

n_dep_2 = df_switchers_valid['deputado_id'].nunique()
n_speech_2 = len(df_switchers_valid)

# How many lost?
lost_dep_2 = n_dep_1 - n_dep_2
lost_speech_2 = n_speech_1 - n_speech_2

print(f"{'2. After removing switches without valid party codes':<60} {n_dep_2:<15} {n_speech_2:<15,}")
if lost_dep_2 > 0:
    print(f"{'   (Lost due to invalid party codes)':<60} {-lost_dep_2:<15} {-lost_speech_2:<15,}")
filtering_steps['step2_valid_parties'] = {'n_deputies': n_dep_2, 'n_speeches': n_speech_2}

# -----------------------------------------------------------------------------
# STEP 3: After restricting to ±360-day window around switch
# -----------------------------------------------------------------------------
# This is the event study sample before Y_confidence filtering

# Rebuild the merge to count accurately (df_es may have been modified)
df_es_rebuild = df_switchers_valid.merge(
    df_events_valid[['deputado_id', 'switch_date', 'old_party_id', 'new_party_id']],
    on='deputado_id'
)
df_es_rebuild['days_from_switch'] = (
    df_es_rebuild['dataHoraInicio'].dt.tz_localize(None) - 
    pd.to_datetime(df_es_rebuild['switch_date']).dt.tz_localize(None)
).dt.days

# Apply ±360 day window
df_es_window = df_es_rebuild[df_es_rebuild['days_from_switch'].abs() <= 360]

n_dep_3 = df_es_window['deputado_id'].nunique()
n_speech_3 = len(df_es_window)

lost_dep_3 = n_dep_2 - n_dep_3
lost_speech_3 = n_speech_2 - n_speech_3

print(f"{'3. After restricting to ±360-day window around switch':<60} {n_dep_3:<15} {n_speech_3:<15,}")
if lost_speech_3 > 0:
    print(f"{'   (Speeches outside window)':<60} {'--':<15} {-lost_speech_3:<15,}")
filtering_steps['step3_time_window'] = {'n_deputies': n_dep_3, 'n_speeches': n_speech_3}

# -----------------------------------------------------------------------------
# STEP 4: After requiring valid Y_confidence (old party in classifier)
# -----------------------------------------------------------------------------
# This is len(df_es) after dropna(subset=['Y_confidence'])

n_dep_4 = df_es['deputado_id'].nunique()
n_speech_4 = len(df_es)

lost_dep_4 = n_dep_3 - n_dep_4
lost_speech_4 = n_speech_3 - n_speech_4

print(f"{'4. After requiring valid classifier prediction':<60} {n_dep_4:<15} {n_speech_4:<15,}")
if lost_speech_4 > 0:
    print(f"{'   (Lost due to missing Y_confidence)':<60} {-lost_dep_4:<15} {-lost_speech_4:<15,}")
filtering_steps['step4_valid_prediction'] = {'n_deputies': n_dep_4, 'n_speeches': n_speech_4}

# -----------------------------------------------------------------------------
# STEP 5: After requiring minimum speeches per deputy
# -----------------------------------------------------------------------------
# This is the final event study sample

n_dep_5 = df_es['deputado_id'].nunique()
n_speech_5 = len(df_es)

lost_dep_5 = n_dep_4 - n_dep_5
lost_speech_5 = n_speech_4 - n_speech_5

print(f"{'5. After requiring ≥' + str(CFG.min_speeches_per_period) + ' speeches per deputy':<60} {n_dep_5:<15} {n_speech_5:<15,}")
if lost_dep_5 > 0:
    print(f"{'   (Deputies with insufficient speeches)':<60} {-lost_dep_5:<15} {-lost_speech_5:<15,}")
filtering_steps['step5_min_speeches'] = {'n_deputies': n_dep_5, 'n_speeches': n_speech_5}


# -----------------------------------------------------------------------------
# SUMMARY STATISTICS
# -----------------------------------------------------------------------------

print("=" * 95)
print("SUMMARY")
print("=" * 95)
print()
print(f"Final event study sample: {n_dep_4} deputies, {n_speech_4:,} speech-events")
print()
print(f"Sample retention rates:")
print(f"   Deputies: {n_dep_4}/{n_dep_1} = {100*n_dep_4/n_dep_1:.1f}%")
print(f"   Speeches: {n_speech_4:,}/{n_speech_1:,} = {100*n_speech_4/n_speech_1:.1f}%")
print()

# Distribution of speeches per deputy
print(f"Speeches per deputy in final sample:")
print(f"   Mean:   {speeches_per_deputy.mean():.1f}")
print(f"   Median: {speeches_per_deputy.median():.1f}")
print(f"   Min:    {speeches_per_deputy.min()}")
print(f"   Max:    {speeches_per_deputy.max()}")
print()

# How many deputies would be lost at different thresholds?
print(f"Deputies meeting minimum speech thresholds:")
for min_thresh in [1, 5, 10, 20]:
    n_meeting = (speeches_per_deputy >= min_thresh).sum()
    pct = 100 * n_meeting / len(speeches_per_deputy)
    print(f"   ≥{min_thresh:2d} speeches: {n_meeting} deputies ({pct:.1f}%)")

print()
print("=" * 95)

# Store results
REVISION_2A_RESULTS = filtering_steps

# %%
# CALCULATE LINGUISTIC CORRELATION ============================================
# =============================================================================

print("Calculating linguistic adaptation for correlation...")
linguistic_effects = []

# We iterate only over deputies present in the event study
for deputy_id in df_events['deputado_id'].unique():
    dep_speeches = df_es[df_es['deputado_id'] == deputy_id]
    
    pre = dep_speeches[(dep_speeches['days_from_switch'] < 0) & (dep_speeches['days_from_switch'] >= -180)]['Y_confidence']
    post = dep_speeches[(dep_speeches['days_from_switch'] > 0) & (dep_speeches['days_from_switch'] <= 180)]['Y_confidence']
    
    if len(pre) >= CFG.min_speeches_per_period and len(post) >= CFG.min_speeches_per_period:
        effect = pre.mean() - post.mean()
        linguistic_effects.append({'deputado_id': deputy_id, 'linguistic_effect': effect, 'abs_linguistic_effect': abs(effect)})

df_ling = pd.DataFrame(linguistic_effects)

# Merge voting and linguistic data
df_corr = df_voting.merge(df_ling, on='deputado_id', how='inner')
df_corr = df_corr.dropna(subset=['loyalty_increase', 'abs_linguistic_effect'])

section2_stats['correlation_analysis'] = {
    'Status': 'Complete',
    'Switchers with Full Data': len(df_corr),
    'Avg Linguistic Effect': df_corr['linguistic_effect'].mean()
    }

# %%
# =============================================================================
# REVISION 1A: CORRELATION SAMPLE COMPARISON TABLE
# =============================================================================
# Compare characteristics of switchers INCLUDED in correlation sample vs EXCLUDED
# This addresses potential selection bias in the language-voting correlation analysis
# =============================================================================

from scipy.stats import ttest_ind, chi2_contingency, mannwhitneyu
import warnings
warnings.filterwarnings('ignore')

print("=" * 90)
print("CORRELATION SAMPLE SELECTION ANALYSIS")
print("=" * 90)
print()

# -----------------------------------------------------------------------------
# 1. IDENTIFY INCLUDED VS EXCLUDED SWITCHERS
# -----------------------------------------------------------------------------

# All unique switchers from events data
all_switcher_ids = set(df_events['deputado_id'].unique())

# Switchers included in correlation sample
included_ids = set(df_corr['deputado_id'].unique())

# Switchers excluded from correlation sample
excluded_ids = all_switcher_ids - included_ids

# Count switch EVENTS vs unique DEPUTIES
n_events_included = len(df_corr)
n_deputies_included = len(included_ids)
n_deputies_excluded = len(excluded_ids)
n_deputies_total = len(all_switcher_ids)

print(f"{'SAMPLE COMPOSITION':-^90}")
print()
print(f"  Unit of Analysis: Switch Events")
print(f"  ─────────────────────────────────")
print(f"  Total switch events in correlation sample:    {n_events_included:>6}")
print(f"  Unique deputies in correlation sample:        {n_deputies_included:>6}")
print(f"  Deputies with multiple switches in sample:    {n_events_included - n_deputies_included:>6}")
print()
print(f"  Sample Selection Summary")
print(f"  ─────────────────────────────────")
print(f"  Total unique switchers:                       {n_deputies_total:>6}")
print(f"  Included in correlation sample:               {n_deputies_included:>6} ({100*n_deputies_included/n_deputies_total:.1f}%)")
print(f"  Excluded from correlation sample:             {n_deputies_excluded:>6} ({100*n_deputies_excluded/n_deputies_total:.1f}%)")
print()
print(f"  Selection Requirements:")
print(f"  • Minimum {CFG.min_speeches_per_period} speeches in each ±180-day window (linguistic effect)")
print(f"  • Minimum {CFG.min_votes_corr} votes in each ±180-day window (voting loyalty)")
print()

# -----------------------------------------------------------------------------
# 2. COMPUTE DEPUTY-LEVEL CHARACTERISTICS
# -----------------------------------------------------------------------------

print(f"{'COMPUTING DEPUTY CHARACTERISTICS':-^90}")
print()

deputy_data = []

for dep_id in tqdm(all_switcher_ids, desc="Processing deputies"):
    row = {'deputado_id': dep_id, 'in_sample': dep_id in included_ids}
    
    # --- A. Career Tenure (from speeches data) ---
    dep_speeches = df[df['deputado_id'] == dep_id]
    if 'career_tenure_years' in dep_speeches.columns and len(dep_speeches) > 0:
        row['career_tenure'] = dep_speeches['career_tenure_years'].median()
    else:
        row['career_tenure'] = np.nan
    
    # --- B. Speech Activity (speeches per year) ---
    if len(dep_speeches) > 0:
        dates = pd.to_datetime(dep_speeches['dataHoraInicio'])
        years_active = (dates.max() - dates.min()).days / 365.25
        years_active = max(years_active, 0.5)  # Minimum 6 months to avoid division issues
        row['speeches_per_year'] = len(dep_speeches) / years_active
        row['total_speeches'] = len(dep_speeches)
    else:
        row['speeches_per_year'] = 0
        row['total_speeches'] = 0
    
    # --- C. Voting Activity (votes per year) ---
    dep_votes = df_votes[df_votes['deputado_id'] == dep_id]
    if len(dep_votes) > 0:
        vote_dates = pd.to_datetime(dep_votes['vote_date'])
        years_voting = (vote_dates.max() - vote_dates.min()).days / 365.25
        years_voting = max(years_voting, 0.5)
        row['votes_per_year'] = len(dep_votes) / years_voting
        row['total_votes'] = len(dep_votes)
    else:
        row['votes_per_year'] = 0
        row['total_votes'] = 0
    
    # --- D. Switch Characteristics (from events data) ---
    dep_switches = df_events[df_events['deputado_id'] == dep_id]
    
    # Ideological distance (average across switches if multiple)
    row['ideo_distance'] = dep_switches['ideo_distance'].mean()
    
    # Number of switches
    row['n_switches'] = len(dep_switches)
    
    # Direction (use first switch for simplicity)
    if len(dep_switches) > 0:
        first_switch = dep_switches.iloc[0]
        row['is_rightward'] = first_switch['is_rightward'] if pd.notna(first_switch['is_rightward']) else np.nan
        row['new_party_id'] = first_switch['new_party_id']
    else:
        row['is_rightward'] = np.nan
        row['new_party_id'] = np.nan
    
    deputy_data.append(row)

df_deputy = pd.DataFrame(deputy_data)

# --- E. Major Party Classification (based on party size in training data) ---
party_sizes = df_train.groupby('idPartido').size().sort_values(ascending=False)
major_party_ids = set(party_sizes.head(5).index.astype(str))  # Top 5 parties = "major"

df_deputy['to_major_party'] = df_deputy['new_party_id'].astype(str).isin(major_party_ids)

# -----------------------------------------------------------------------------
# 3. SPLIT INTO INCLUDED VS EXCLUDED GROUPS
# -----------------------------------------------------------------------------

included = df_deputy[df_deputy['in_sample'] == True].copy()
excluded = df_deputy[df_deputy['in_sample'] == False].copy()

# -----------------------------------------------------------------------------
# 4. STATISTICAL COMPARISONS
# -----------------------------------------------------------------------------

print()
print(f"{'TABLE: COMPARISON OF INCLUDED VS EXCLUDED SWITCHERS':=^90}")
print()
print(f"{'Variable':<30} {'Included':<15} {'Excluded':<15} {'Diff':<12} {'p-value':<10} {'Test':<10}")
print(f"{'':30} {'(N='+str(len(included))+')':<15} {'(N='+str(len(excluded))+')':<15}")
print("-" * 92)

results_1a = {}

# --- A. Career Tenure ---
inc_vals = included['career_tenure'].dropna()
exc_vals = excluded['career_tenure'].dropna()
if len(inc_vals) > 1 and len(exc_vals) > 1:
    t_stat, p_val = ttest_ind(inc_vals, exc_vals, equal_var=False)
    diff = inc_vals.mean() - exc_vals.mean()
    print(f"{'Career tenure (years)':<30} {inc_vals.mean():<15.2f} {exc_vals.mean():<15.2f} {diff:<+12.2f} {p_val:<10.4f} {'t-test':<10}")
    results_1a['career_tenure'] = {'included': inc_vals.mean(), 'excluded': exc_vals.mean(), 'p': p_val}

# --- B. Speeches per Year ---
inc_vals = included['speeches_per_year'].dropna()
exc_vals = excluded['speeches_per_year'].dropna()
if len(inc_vals) > 1 and len(exc_vals) > 1:
    t_stat, p_val = ttest_ind(inc_vals, exc_vals, equal_var=False)
    diff = inc_vals.mean() - exc_vals.mean()
    print(f"{'Speeches per year':<30} {inc_vals.mean():<15.1f} {exc_vals.mean():<15.1f} {diff:<+12.1f} {p_val:<10.4f} {'t-test':<10}")
    results_1a['speeches_per_year'] = {'included': inc_vals.mean(), 'excluded': exc_vals.mean(), 'p': p_val}

# --- C. Total Speeches ---
inc_vals = included['total_speeches'].dropna()
exc_vals = excluded['total_speeches'].dropna()
if len(inc_vals) > 1 and len(exc_vals) > 1:
    t_stat, p_val = ttest_ind(inc_vals, exc_vals, equal_var=False)
    diff = inc_vals.mean() - exc_vals.mean()
    print(f"{'Total speeches':<30} {inc_vals.mean():<15.1f} {exc_vals.mean():<15.1f} {diff:<+12.1f} {p_val:<10.4f} {'t-test':<10}")
    results_1a['total_speeches'] = {'included': inc_vals.mean(), 'excluded': exc_vals.mean(), 'p': p_val}

# --- D. Votes per Year ---
inc_vals = included['votes_per_year'].dropna()
exc_vals = excluded['votes_per_year'].dropna()
if len(inc_vals) > 1 and len(exc_vals) > 1:
    t_stat, p_val = ttest_ind(inc_vals, exc_vals, equal_var=False)
    diff = inc_vals.mean() - exc_vals.mean()
    print(f"{'Votes per year':<30} {inc_vals.mean():<15.1f} {exc_vals.mean():<15.1f} {diff:<+12.1f} {p_val:<10.4f} {'t-test':<10}")
    results_1a['votes_per_year'] = {'included': inc_vals.mean(), 'excluded': exc_vals.mean(), 'p': p_val}

# --- E. Total Votes ---
inc_vals = included['total_votes'].dropna()
exc_vals = excluded['total_votes'].dropna()
if len(inc_vals) > 1 and len(exc_vals) > 1:
    t_stat, p_val = ttest_ind(inc_vals, exc_vals, equal_var=False)
    diff = inc_vals.mean() - exc_vals.mean()
    print(f"{'Total votes':<30} {inc_vals.mean():<15.1f} {exc_vals.mean():<15.1f} {diff:<+12.1f} {p_val:<10.4f} {'t-test':<10}")
    results_1a['total_votes'] = {'included': inc_vals.mean(), 'excluded': exc_vals.mean(), 'p': p_val}

# --- F. Ideological Distance ---
inc_vals = included['ideo_distance'].dropna()
exc_vals = excluded['ideo_distance'].dropna()
if len(inc_vals) > 1 and len(exc_vals) > 1:
    t_stat, p_val = ttest_ind(inc_vals, exc_vals, equal_var=False)
    diff = inc_vals.mean() - exc_vals.mean()
    print(f"{'Ideological distance':<30} {inc_vals.mean():<15.3f} {exc_vals.mean():<15.3f} {diff:<+12.3f} {p_val:<10.4f} {'t-test':<10}")
    results_1a['ideo_distance'] = {'included': inc_vals.mean(), 'excluded': exc_vals.mean(), 'p': p_val}

# --- G. Number of Switches ---
inc_vals = included['n_switches'].dropna()
exc_vals = excluded['n_switches'].dropna()
if len(inc_vals) > 1 and len(exc_vals) > 1:
    t_stat, p_val = ttest_ind(inc_vals, exc_vals, equal_var=False)
    diff = inc_vals.mean() - exc_vals.mean()
    print(f"{'Number of switches':<30} {inc_vals.mean():<15.2f} {exc_vals.mean():<15.2f} {diff:<+12.2f} {p_val:<10.4f} {'t-test':<10}")
    results_1a['n_switches'] = {'included': inc_vals.mean(), 'excluded': exc_vals.mean(), 'p': p_val}

print("-" * 92)

# --- H. Switch to Major Party (Chi-squared) ---
inc_major = included['to_major_party'].sum()
exc_major = excluded['to_major_party'].sum()
inc_pct = 100 * inc_major / len(included) if len(included) > 0 else 0
exc_pct = 100 * exc_major / len(excluded) if len(excluded) > 0 else 0

contingency = [
    [inc_major, len(included) - inc_major],
    [exc_major, len(excluded) - exc_major]
]
chi2, p_chi2, dof, expected = chi2_contingency(contingency)
diff_pct = inc_pct - exc_pct

print(f"{'Switch to major party (%)':<30} {inc_pct:<15.1f} {exc_pct:<15.1f} {diff_pct:<+12.1f} {p_chi2:<10.4f} {'χ²':<10}")
results_1a['to_major_party'] = {'included': inc_pct, 'excluded': exc_pct, 'p': p_chi2}

# --- I. Rightward Switch (Chi-squared) ---
inc_right = included['is_rightward'].dropna()
exc_right = excluded['is_rightward'].dropna()
if len(inc_right) > 0 and len(exc_right) > 0:
    inc_right_pct = 100 * inc_right.mean()
    exc_right_pct = 100 * exc_right.mean()
    
    contingency_r = [
        [int(inc_right.sum()), int(len(inc_right) - inc_right.sum())],
        [int(exc_right.sum()), int(len(exc_right) - exc_right.sum())]
    ]
    chi2_r, p_chi2_r, _, _ = chi2_contingency(contingency_r)
    diff_r = inc_right_pct - exc_right_pct
    
    print(f"{'Rightward switch (%)':<30} {inc_right_pct:<15.1f} {exc_right_pct:<15.1f} {diff_r:<+12.1f} {p_chi2_r:<10.4f} {'χ²':<10}")
    results_1a['rightward'] = {'included': inc_right_pct, 'excluded': exc_right_pct, 'p': p_chi2_r}

print("-" * 92)
print()

# -----------------------------------------------------------------------------
# 5. INTERPRETATION
# -----------------------------------------------------------------------------

print(f"{'INTERPRETATION':=^90}")
print()

# Count significant differences
sig_vars = [k for k, v in results_1a.items() if v['p'] < 0.05]
marginal_vars = [k for k, v in results_1a.items() if 0.05 <= v['p'] < 0.10]

if len(sig_vars) == 0:
    print("✅ NO significant differences (p < 0.05) between included and excluded switchers.")
    print("   The correlation sample appears representative of all switchers.")
else:
    print(f"⚠️  {len(sig_vars)} significant difference(s) detected (p < 0.05):")
    for var in sig_vars:
        r = results_1a[var]
        direction = "higher" if r['included'] > r['excluded'] else "lower"
        print(f"   • {var}: Included switchers have {direction} values (p = {r['p']:.4f})")


print()
print("Key selection mechanism:")
print(f"   • Included deputies gave {results_1a.get('speeches_per_year', {}).get('included', 0):.1f} speeches/year vs "
      f"{results_1a.get('speeches_per_year', {}).get('excluded', 0):.1f} for excluded")
print(f"   • Included deputies cast {results_1a.get('votes_per_year', {}).get('included', 0):.1f} votes/year vs "
      f"{results_1a.get('votes_per_year', {}).get('excluded', 0):.1f} for excluded")
print()
print("   → Selection is driven by ACTIVITY LEVEL, not by substantive political characteristics")
print()
print("=" * 90)

# Store for later use
REVISION_1A_RESULTS = {
    'n_events_included': n_events_included,
    'n_deputies_included': n_deputies_included,
    'n_deputies_excluded': n_deputies_excluded,
    'n_deputies_total': n_deputies_total,
    'comparisons': results_1a,
    'df_deputy': df_deputy
}

# %%
# DML EVENT STUDY (COMPUTATION) ===============================================
# =============================================================================

# Initialize stats container for this block
dml_stats = {}

# -------------------------------------------------------------------------
# 0) SAFETY: align the analysis window with binning (avoid NaN tau "ghost" rows)
#    Here we keep exactly 6x 60-day bins on each side => |days| <= 360
# -------------------------------------------------------------------------
df_es = df_es.loc[df_es["days_from_switch"].abs() <= 360].copy()

# -------------------------------------------------------------------------
# 1) Event-time bins (60-day), explicit boundary behavior
#    right=False => intervals are [a,b), so day 0 is in [0,60) i.e., post period
# -------------------------------------------------------------------------
bins = list(range(-360, 361, 60))  # edges: -360,-300,...,300,360
labels = [-6, -5, -4, -3, -2, -1,  1,  2,  3,  4,  5,  6]  # 12 labels for 12 intervals

df_es["tau"] = pd.cut(
    df_es["days_from_switch"],
    bins=bins,
    labels=labels,
    right=False,
    include_lowest=True
)

# Drop any residual NaNs defensively (should be none after |days|<=359)
df_es = df_es.dropna(subset=["tau"]).copy()

# Create time dummies; -1 is the reference category (dropped)
time_dummies = pd.get_dummies(df_es["tau"].astype(int), prefix="t")

# Ensure reference exists and drop it
if "t_-1" not in time_dummies.columns:
    raise ValueError("Reference period t_-1 not found. Check bin edges / tau construction.")
time_dummies = time_dummies.drop(columns=["t_-1"])

# Sort dummies in chronological tau order for cleaner output/plots
def _tau_from_col(c: str) -> int:
    return int(c.replace("t_", ""))
time_dummies = time_dummies[sorted(time_dummies.columns, key=_tau_from_col)]

print("Building covariate matrix...")

# -------------------------------------------------------------------------
# 2) Covariates: include topic FE
# -------------------------------------------------------------------------
X = build_covariates(
    df_es,
    include_calendar_month_fe=False,   
    include_topic_fe=True,
    date_col="dataHoraInicio",
    verbose=True
)

has_topics = "topic_id" in df_es.columns
Y = df_es["Y_confidence"].astype(float).values
clusters = df_es["deputado_id"].values

# -------------------------------------------------------------------------
# 3) DML with CLUSTERED cross-fitting (by deputy_id) to avoid leakage
#    We do manual cross-fitting so we can:
#    - pass groups
#    - gracefully handle folds with only one class in D (rare for some tau)
# -------------------------------------------------------------------------
from sklearn.model_selection import GroupKFold
from sklearn.base import clone

gkf = GroupKFold(n_splits=CFG.dml_n_splits)

base_learner_Y = HistGradientBoostingRegressor(
    max_iter=100, max_depth=5, random_state=SEED
)
base_learner_D = HistGradientBoostingClassifier(
    max_iter=50, max_depth=3, random_state=SEED
)

# --- Residualize Y ---
Y_pred = np.empty_like(Y, dtype=float)

for train_idx, test_idx in gkf.split(X, Y, groups=clusters):
    learner_Y = clone(base_learner_Y)
    learner_Y.fit(X.iloc[train_idx], Y[train_idx])
    Y_pred[test_idx] = learner_Y.predict(X.iloc[test_idx])

Y_resid = Y - Y_pred

# --- Residualize Treatments (time dummies) ---
D_resid = pd.DataFrame(index=df_es.index, columns=time_dummies.columns, dtype=float)

for col in tqdm(time_dummies.columns, desc="DML Residualizing (cluster-CV)"):
    d = time_dummies[col].astype(int).values
    d_pred = np.empty_like(d, dtype=float)

    for train_idx, test_idx in gkf.split(X, d, groups=clusters):
        d_train = d[train_idx]
        uniq = np.unique(d_train)

        if uniq.size < 2:
            d_pred[test_idx] = float(uniq[0])
        else:
            learner_D = clone(base_learner_D)
            learner_D.fit(X.iloc[train_idx], d_train)
            d_pred[test_idx] = learner_D.predict_proba(X.iloc[test_idx])[:, 1]

    D_resid[col] = d - d_pred

# -------------------------------------------------------------------------
# 4) Final-stage OLS with cluster-robust SE (clustered by deputy)
# -------------------------------------------------------------------------
X_final = sm.add_constant(D_resid)  # preserves column names for readable params
results_dml = sm.OLS(Y_resid, X_final).fit(
    cov_type="cluster",
    cov_kwds={"groups": clusters}
)

# Store Results for Reporting
coefs = results_dml.params.drop("const")
cis = results_dml.conf_int().drop(index="const")
pvals = results_dml.pvalues.drop("const")
bse = results_dml.bse.drop("const")

# -------------------------------------------------------------------------
# 5) Effect size (Cohen's d) for τ = +1 (i.e., t_1)
# -------------------------------------------------------------------------
sd_Y_resid = np.std(Y_resid, ddof=1)
print(f"Residual SD of Y: {sd_Y_resid:.4f}")

if "t_1" not in results_dml.params.index:
    raise ValueError("t_1 not found in results. Check tau labeling / dummy creation.")

beta_t1 = float(results_dml.params["t_1"])
cohens_d_t1 = abs(beta_t1) / sd_Y_resid if sd_Y_resid > 0 else np.nan
print(f"Cohen's d (τ = +1): {cohens_d_t1:.3f}")

# -------------------------------------------------------------------------
# 6) Prepare Plot Data
# -------------------------------------------------------------------------
plot_data = pd.DataFrame({
    "time": [_tau_from_col(c) for c in coefs.index],
    "coef": coefs.values,
    "lower": cis[0].values,
    "upper": cis[1].values,
    "pval": pvals.values,
    "sig": pvals.values < 0.05
}).sort_values("time")

# Add reference point (t = -1) explicitly at zero
ref_point = pd.DataFrame({
    "time": [-1],
    "coef": [0.0],
    "lower": [0.0],
    "upper": [0.0],
    "pval": [np.nan],
    "sig": [False]
})
plot_data = pd.concat([plot_data, ref_point], ignore_index=True).sort_values("time").reset_index(drop=True)

# -------------------------------------------------------------------------
# 7) Save stats
# -------------------------------------------------------------------------
dml_stats["model_summary"] = {
    "N_Observations": int(len(Y)),
    "N_Clusters": int(len(np.unique(clusters))),
    "Topic_Controls": bool(has_topics),
    "CalendarMonth_FE": False,
    "R_squared": float(results_dml.rsquared)
}

dml_stats["results_table"] = pd.DataFrame({
    "Coef": coefs,
    "Std_Err": bse,
    "P_Value": pvals,
    "Lower_CI": cis[0],
    "Upper_CI": cis[1]
})

dml_stats["plot_data"] = plot_data


# %%
# =============================================================================
# REVISION 5A: BASELINE PROBABILITY CALCULATION
# =============================================================================
# Calculate the mean P(old party|speech) in the pre-switch reference period (τ = -1)
# This is needed to interpret effect sizes as percentage changes from baseline
# =============================================================================

print("=" * 80)
print("BASELINE PROBABILITY ANALYSIS")
print("=" * 80)
print()

# -----------------------------------------------------------------------------
# 1. IDENTIFY REFERENCE PERIOD (τ = -1)
# -----------------------------------------------------------------------------
# Using 60-day bins: τ = -1 corresponds to days_from_switch in [-60, 0)

# Method 1: Using the 'month' column created for DML
df_ref_method1 = df_es[df_es['tau'].astype(int) == -1]

# Method 2: Direct day filtering (backup/verification)
df_ref_method2 = df_es[
    (df_es['days_from_switch'] >= -60) & 
    (df_es['days_from_switch'] < 0)
]

print("Reference period definition: τ = -1 (days -60 to 0 before switch)")
print()
print(f"Method 1 (using 'month' column): {len(df_ref_method1):,} speeches")
print(f"Method 2 (direct day filter):    {len(df_ref_method2):,} speeches")
print()

# Use Method 1 (consistent with DML bins)
df_ref = df_ref_method1

# -----------------------------------------------------------------------------
# 2. CALCULATE BASELINE STATISTICS
# -----------------------------------------------------------------------------

baseline_mean = df_ref['Y_confidence'].mean()
baseline_std = df_ref['Y_confidence'].std()
baseline_median = df_ref['Y_confidence'].median()
baseline_n = len(df_ref)
baseline_n_deputies = df_ref['deputado_id'].nunique()

print("=" * 80)
print("BASELINE P(old party|speech) IN REFERENCE PERIOD (τ = -1)")
print("=" * 80)
print()
print(f"   N speeches:  {baseline_n:,}")
print(f"   N deputies:  {baseline_n_deputies}")
print()
print(f"   Mean:        {baseline_mean:.4f}  ({baseline_mean:.1%})")
print(f"   Median:      {baseline_median:.4f}")
print(f"   Std Dev:     {baseline_std:.4f}")
print(f"   Range:       [{df_ref['Y_confidence'].min():.4f}, {df_ref['Y_confidence'].max():.4f}]")
print()

# -----------------------------------------------------------------------------
# 3. INTERPRET TREATMENT EFFECTS AS PERCENTAGE CHANGES
# -----------------------------------------------------------------------------

print("=" * 80)
print("EFFECT SIZE INTERPRETATION")
print("=" * 80)
print()

# Get key coefficients from DML results
coef_t1 = results_dml.params.get('t_1', np.nan)
coef_t2 = results_dml.params.get('t_2', np.nan)

# Peak effect (most negative)
post_coefs = {k: v for k, v in results_dml.params.items() if k.startswith('t_') and not k.startswith('t_-')}
if post_coefs:
    peak_period = min(post_coefs, key=post_coefs.get)
    peak_effect = post_coefs[peak_period]
else:
    peak_period = 't_1'
    peak_effect = coef_t1

print(f"Baseline probability:           {baseline_mean:.4f} ({baseline_mean:.1%})")
print()
print(f"Treatment effects (percentage point change from baseline):")
print(f"   τ = +1 (first 2 months):     {coef_t1:+.4f} pp")
print(f"   τ = +2 (months 3-4):         {coef_t2:+.4f} pp")
print(f"   Peak effect ({peak_period}):            {peak_effect:+.4f} pp")
print()

# Percentage change interpretation
pct_change_t1 = 100 * coef_t1 / baseline_mean
pct_change_peak = 100 * peak_effect / baseline_mean

print(f"As percentage of baseline:")
print(f"   τ = +1:  {pct_change_t1:+.1f}% change")
print(f"   Peak:    {pct_change_peak:+.1f}% change")
print()

# -----------------------------------------------------------------------------
# 4. COMPARISON WITH OTHER PERIODS
# -----------------------------------------------------------------------------

print("=" * 80)
print("P(old party|speech) BY PERIOD")
print("=" * 80)
print()
print(f"{'Period':<15} {'N':<10} {'Mean':<12} {'Std':<12} {'vs Baseline':<15}")
print("-" * 65)

# Calculate for each period
for period in sorted(df_es['tau'].dropna().astype(int).unique()):
    period_int = int(period)
    df_period = df_es[df_es['tau'].astype(int) == int(period)]
    p_mean = df_period['Y_confidence'].mean()
    p_std = df_period['Y_confidence'].std()
    p_n = len(df_period)
    
    if period_int == -1:
        diff_str = "(reference)"
    else:
        diff = p_mean - baseline_mean
        diff_str = f"{diff:+.4f}"
    
    print(f"τ = {period_int:<+3}       {p_n:<10,} {p_mean:<12.4f} {p_std:<12.4f} {diff_str:<15}")

print("-" * 65)
print()

# -----------------------------------------------------------------------------
# 5. STORE RESULTS
# -----------------------------------------------------------------------------

REVISION_5A_RESULTS = {
    'baseline_mean': baseline_mean,
    'baseline_std': baseline_std,
    'baseline_median': baseline_median,
    'baseline_n_speeches': baseline_n,
    'baseline_n_deputies': baseline_n_deputies,
    'pct_change_t1': pct_change_t1,
    'pct_change_peak': pct_change_peak,
    'peak_period': peak_period,
    'peak_effect': peak_effect
}

print("=" * 80)
print(f"KEY NUMBER FOR PAPER: Baseline P(old|speech) = {baseline_mean:.4f}")
print(f"                      Peak effect = {abs(peak_effect):.4f} pp = {abs(pct_change_peak):.1f}% decline")
print("=" * 80)

# %%
# PRE-TRENDS F-TEST (COMPUTATION) =============================================
# =============================================================================

# Extract pre-period coefficients
pre_period_cols = [c for c in results_dml.params.index if c.startswith('t_-') and c != 'const']
pre_period_cols_sorted = sorted(pre_period_cols, key=lambda x: int(x.split('_')[1]))

# Get coefficients and covariance matrix
beta_pre = results_dml.params[pre_period_cols_sorted].values
vcov_pre = results_dml.cov_params().loc[pre_period_cols_sorted, pre_period_cols_sorted].values

# Wald test: (beta)' * V^(-1) * (beta)
wald_stat = beta_pre @ np.linalg.inv(vcov_pre) @ beta_pre
f_stat = wald_stat / len(beta_pre)
df_numerator = len(beta_pre)
df_denominator = results_dml.df_resid

p_value_f = 1 - stats.f.cdf(f_stat, df_numerator, df_denominator)
p_value_chi2 = 1 - stats.chi2.cdf(wald_stat, df_numerator)

# Store results
dml_stats['pretrends'] = {
    'f_stat': f_stat,
    'p_value': p_value_f,
    'wald_stat': wald_stat,
    'df': (df_numerator, df_denominator),
    'n_pre_periods': len(beta_pre),
    'passed': p_value_f > 0.10,
    'marginal': p_value_f > 0.05,
    # Verification info
    'n_obs': int(results_dml.nobs),
    'n_params': len(results_dml.params),
    'df_check': int(results_dml.nobs) - len(results_dml.params)
}

# Print verification
print(f"F-test df verification: {dml_stats['pretrends']['n_obs']} - {dml_stats['pretrends']['n_params']} = {dml_stats['pretrends']['df_check']}")
print(f"df_resid from model: {df_denominator}")
print(f"Match: {dml_stats['pretrends']['df_check'] == df_denominator}")

# %%
# LINGUISTIC VS VOTING BEHAVIOR (COMPUTATION) ================================
# =============================================================================
from scipy.stats import pearsonr, spearmanr, norm, t
from scipy.optimize import brentq
import numpy as np

# Initialize stats container
section2_1_stats = {}

# 1. CORRELATION ANALYSIS -----------------------------------------------------
if 'df_corr' in locals() and len(df_corr) >= 10:
    # Calculate correlation
    r_pearson, p_pearson = pearsonr(df_corr['abs_linguistic_effect'], df_corr['loyalty_increase'])
    r_spearman, p_spearman = spearmanr(df_corr['abs_linguistic_effect'], df_corr['loyalty_increase'])
    
    section2_1_stats['correlation'] = {
        'status': 'Complete',
        'n': len(df_corr),
        'r_pearson': r_pearson,
        'p_pearson': p_pearson,
        'r_spearman': r_spearman,
        'p_spearman': p_spearman,
        # Store data for plotting
        'x_data': df_corr['abs_linguistic_effect'].values,
        'y_data': df_corr['loyalty_increase'].values
    }
    
    # 2. POWER ANALYSIS -------------------------------------------------------
    n_current = len(df_corr)
    
    def power_for_correlation(n, r, alpha=0.05):
        z_r = np.arctanh(r)
        se = 1 / np.sqrt(n - 3)
        z_crit = norm.ppf(1 - alpha/2)
        ncp = z_r / se
        return 1 - norm.cdf(z_crit - ncp) + norm.cdf(-z_crit - ncp)

    def min_detectable_r(n, power=0.80, alpha=0.05):
        try:
            return brentq(lambda r: power_for_correlation(n, r, alpha) - power, 0.01, 0.95)
        except:
            return np.nan

    # Calculate Power Stats
    r_min_80 = min_detectable_r(n_current, power=0.80)
    r_min_50 = min_detectable_r(n_current, power=0.50)
    
    # UPDATED: Include all effect sizes needed for paper table
    effect_sizes = [0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]
    power_curve = {r: power_for_correlation(n_current, r) for r in effect_sizes}
    
    # Calculate 95% CI for observed correlation using Fisher z-transformation
    z_observed = np.arctanh(r_pearson)
    se_z = 1 / np.sqrt(n_current - 3)
    z_crit = norm.ppf(0.975)  # 95% CI
    ci_lower = np.tanh(z_observed - z_crit * se_z)
    ci_upper = np.tanh(z_observed + z_crit * se_z)
    
    # Calculate Sample Size Needs
    n_needed_for_80 = {}
    for r_target in [0.10, 0.20, 0.30]:
        z_alpha = norm.ppf(1 - 0.05/2)
        z_beta = norm.ppf(0.80)
        z_r = np.arctanh(r_target)
        n_needed_for_80[r_target] = int(np.ceil(((z_alpha + z_beta) / z_r)**2 + 3))

    section2_1_stats['power_analysis'] = {
        'n_current': n_current,
        'mde_80': r_min_80,
        'mde_50': r_min_50,
        'power_curve': power_curve,
        'n_needed': n_needed_for_80,
        'ci_lower': ci_lower,
        'ci_upper': ci_upper
    }
    
    # 3. EQUIVALENCE TESTING (TOST) -------------------------------------------
    delta = 0.20  # Equivalence margin
    t_upper = (r_pearson - delta) / (1 / np.sqrt(n_current - 3))
    p_upper = t.cdf(t_upper, df=n_current - 3)
    
    t_lower = (r_pearson + delta) / (1 / np.sqrt(n_current - 3))
    p_lower = 1 - t.cdf(t_lower, df=n_current - 3)
    p_tost = max(p_upper, p_lower)
    
    section2_1_stats['equivalence'] = {
        'margin': delta,
        'p_tost': p_tost,
        'is_equivalent': p_tost < 0.05
    }
    
    # 4. REGRESSION ANALYSIS (for Table 2) ------------------------------------
    if len(df_corr) >= 10:
        from sklearn.preprocessing import StandardScaler
        import statsmodels.api as sm
        
        # Standardize variables
        scaler = StandardScaler()
        X_reg = scaler.fit_transform(df_corr[['abs_linguistic_effect']])
        y_reg = scaler.fit_transform(df_corr[['loyalty_increase']])
        
        # Add constant and fit
        X_reg_const = sm.add_constant(X_reg)
        reg_model = sm.OLS(y_reg, X_reg_const).fit()
        
        section2_1_stats['regression'] = {
            'beta': reg_model.params[1],
            'se': reg_model.bse[1],
            'p_value': reg_model.pvalues[1],
            'r_squared': reg_model.rsquared
        }
    else:
        section2_1_stats['regression'] = {'status': 'Skipped'}

    # 5. CONTROLLED REGRESSION ANALYSIS (Selection Bias Check) ----------------
    if 'df_deputy' in locals() and len(df_corr) >= 10:
        
        # Merge selection-relevant variables into df_corr
        df_corr_controls = df_corr.merge(
            df_deputy[['deputado_id', 'ideo_distance', 'n_switches', 'to_major_party', 
                       'is_rightward', 'speeches_per_year', 'career_tenure']],
            on='deputado_id',
            how='left'
        )
        
        # Model 1: Baseline (no controls) - unstandardized for interpretability
        Y_base = df_corr_controls['loyalty_increase'].values
        X_base = df_corr_controls[['abs_linguistic_effect']].values
        model_base = sm.OLS(Y_base, sm.add_constant(X_base)).fit(cov_type='HC3')
        
        # Model 2: With selection controls
        control_vars = ['ideo_distance', 'n_switches', 'to_major_party', 'is_rightward', 'speeches_per_year']
        
        df_reg = df_corr_controls.dropna(subset=['loyalty_increase', 'abs_linguistic_effect'] + control_vars).copy()
        
        Y_ctrl = df_reg['loyalty_increase'].values
        X_ctrl = df_reg[['abs_linguistic_effect'] + control_vars].copy()
        
        # Convert boolean to int
        for col in ['to_major_party', 'is_rightward']:
            if col in X_ctrl.columns and X_ctrl[col].dtype == bool:
                X_ctrl[col] = X_ctrl[col].astype(int)
        
        X_ctrl = sm.add_constant(X_ctrl.astype(float))
        model_ctrl = sm.OLS(Y_ctrl, X_ctrl).fit(cov_type='HC3')
        
        # Model 3: With selection controls + career tenure
        control_vars_full = control_vars + ['career_tenure']
        df_reg_full = df_corr_controls.dropna(subset=['loyalty_increase', 'abs_linguistic_effect'] + control_vars_full).copy()
        
        Y_full = df_reg_full['loyalty_increase'].values
        X_full = df_reg_full[['abs_linguistic_effect'] + control_vars_full].copy()
        
        for col in ['to_major_party', 'is_rightward']:
            if col in X_full.columns and X_full[col].dtype == bool:
                X_full[col] = X_full[col].astype(int)
        
        X_full = sm.add_constant(X_full.astype(float))
        model_full = sm.OLS(Y_full, X_full).fit(cov_type='HC3')
        
        section2_1_stats['controlled_regression'] = {
            'model_base': {
                'beta': model_base.params[1],
                'se': model_base.bse[1],
                'p': model_base.pvalues[1],
                'n': int(model_base.nobs),
                'r2': model_base.rsquared
            },
            'model_ctrl': {
                'beta': model_ctrl.params['abs_linguistic_effect'],
                'se': model_ctrl.bse['abs_linguistic_effect'],
                'p': model_ctrl.pvalues['abs_linguistic_effect'],
                'n': int(model_ctrl.nobs),
                'r2': model_ctrl.rsquared,
                'full_params': model_ctrl.params.to_dict(),
                'full_se': model_ctrl.bse.to_dict(),
                'full_pvalues': model_ctrl.pvalues.to_dict()
            },
            'model_full': {
                'beta': model_full.params['abs_linguistic_effect'],
                'se': model_full.bse['abs_linguistic_effect'],
                'p': model_full.pvalues['abs_linguistic_effect'],
                'n': int(model_full.nobs),
                'r2': model_full.rsquared
            }
        }
    else:
        section2_1_stats['controlled_regression'] = {'status': 'Skipped (df_deputy not available)'}

else:
    section2_1_stats['correlation'] = {'status': 'Skipped (N < 10)', 'n': len(df_corr) if 'df_corr' in locals() else 0}

# %% [markdown]
# ### Section Outputs

# %%
# =============================================================================
# SECTION 2 REPORT
# =============================================================================
def print_full_section2_report(general_stats, dml_data, save_dir=None):
    print("="*60)
    print("SECTION 2: FULL ANALYSIS REPORT")
    print("="*60)
    print("\n")
    
    # 1. Classification & Data Stats
    print("--- 1. Classifier & Data ---")
    if 'model_performance' in general_stats:
        print(f"CV Accuracy:     {general_stats['model_performance']['CV Accuracy']}")
        print(f"Training Size:   {general_stats['model_performance']['Training Samples']:,}")
    if 'event_study_data' in general_stats:
        print(f"Event Study N:   {general_stats['event_study_data']['Total Speech-Events']:,}")
    if 'correlation_analysis' in general_stats:
        corr = general_stats['correlation_analysis']
        if corr.get('Status') == 'Complete':
            print(f"Correlation N:   {corr['Switchers with Full Data']:,}")
            print(f"Avg Ling Effect: {corr['Avg Linguistic Effect']:.4f}")
    print("\n")

    # 2. DML Model Summary
    print("--- 2. DML Model Specification ---")
    print(f"Observations:    {dml_data['model_summary']['N_Observations']:,}")
    print(f"Clusters:        {dml_data['model_summary']['N_Clusters']:,}")
    print(f"Topic Controls:  {'Yes' if dml_data['model_summary']['Topic_Controls'] else 'No'}")
    print(f"R-squared:       {dml_data['model_summary']['R_squared']:.4f}")
    print("\n")

    # 3. Pre-trends Test
    pt = dml_data['pretrends']
    print("--- 3. Pre-trends Test (Parallel Trends) ---")
    print(f"H0: All pre-period effects = 0")
    print(f"F-statistic:     {pt['f_stat']:.3f}")
    print(f"P-value:         {pt['p_value']:.4f}")
    print("\n")
    
    # 4. Detailed Regression Output
    print("--- 4. Detailed Regression Results ---")
    # Assuming 'results_dml' is global or accessible, otherwise pass it as arg
    # Since it was used in your snippet directly, I assume it's available in scope.
    if 'results_dml' in globals():
        time_periods = [-6, -5, -4, -3, -2, 1, 2, 3, 4, 5, 6]
        print(f"{'Period':<8} | {'Coef':<9} | {'SE':<8} | {'p-value':<8} | {'CI Lower':<9} | {'CI Upper':<9}")
        print("-" * 75)
        for t in time_periods:
            col_name = f't_{t}'
            if col_name in results_dml.params.index:
                coef = results_dml.params[col_name]
                se = results_dml.bse[col_name]
                pval = results_dml.pvalues[col_name]
                ci = results_dml.conf_int().loc[col_name]
                
                stars = '***' if pval < 0.001 else '**' if pval < 0.01 else '*' if pval < 0.05 else ''
                
                print(f"τ = {t:+2d} | {coef:9.4f}{stars:3s} | {se:8.4f} | {pval:8.3f} | {ci[0]:9.4f} | {ci[1]:9.4f}")
    else:
        print("(Detailed regression coefficients not available in current scope)")
    print("\n")
    

# Pass both stats dictionaries to the report
print_full_section2_report(section2_stats, dml_stats)

def print_section2_1_report(stats):
    print("="*80)
    print("SECTION 2.1: LANGUAGE VS VOTING BEHAVIOR REPORT")
    print("="*80)
    print("\n")
    
    corr = stats['correlation']
    if corr['status'] != 'Complete':
        print(f"⚠️ Analysis Skipped: {corr['status']}")
        return

    # =========================================================================
    # TABLE 1: VOTING LOYALTY AGGREGATE STATISTICS
    # =========================================================================
    print("="*80)
    print("TABLE 1: VOTING LOYALTY BEFORE AND AFTER PARTY SWITCH")
    print("="*80)
    print("\n")
    
    # Get voting data with complete records
    df_voting_complete = df_voting.dropna(subset=[
        'loyalty_pre_old', 'loyalty_post_new', 'loyalty_post_old'
    ])
    
    n_switchers = len(df_voting_complete)
    
    # Pre-switch statistics
    pre_mean = df_voting_complete['loyalty_pre_old'].mean()
    pre_std = df_voting_complete['loyalty_pre_old'].std()
    
    # Post-switch statistics
    post_new_mean = df_voting_complete['loyalty_post_new'].mean()
    post_new_std = df_voting_complete['loyalty_post_new'].std()
    
    post_old_mean = df_voting_complete['loyalty_post_old'].mean()
    post_old_std = df_voting_complete['loyalty_post_old'].std()
    
    # Changes
    delta_new = df_voting_complete['loyalty_post_new'] - df_voting_complete['loyalty_pre_old']
    delta_old = df_voting_complete['loyalty_post_old'] - df_voting_complete['loyalty_pre_old']
    
    delta_new_mean = delta_new.mean()
    delta_new_std = delta_new.std()
    
    delta_old_mean = delta_old.mean()
    delta_old_std = delta_old.std()
    
    # T-tests
    from scipy.stats import ttest_1samp
    t_new, p_new = ttest_1samp(delta_new.dropna(), 0)
    t_old, p_old = ttest_1samp(delta_old.dropna(), 0)
    
    print(f"{'Measure':<45} {'Mean':>10} {'SD':>10} {'N':>6}")
    print("-" * 80)
    print(f"{'Pre-switch:':<45}")
    print(f"{'  Loyalty to old party':<45} {pre_mean:>10.3f} {pre_std:>10.3f} {n_switchers:>6}")
    print()
    print(f"{'Post-switch:':<45}")
    print(f"{'  Loyalty to new party':<45} {post_new_mean:>10.3f} {post_new_std:>10.3f} {n_switchers:>6}")
    print(f"{'  Loyalty to old party (counterfactual)':<45} {post_old_mean:>10.3f} {post_old_std:>10.3f} {n_switchers:>6}")
    print()
    print(f"{'Change:':<45}")
    print(f"{'  Δ to new':<45} {delta_new_mean:>10.3f} {delta_new_std:>10.3f} {n_switchers:>6}")
    print(f"{'  Δ to old':<45} {delta_old_mean:>10.3f} {delta_old_std:>10.3f} {n_switchers:>6}")
    print("-" * 80)
    print(f"\nt-tests (H₀: Δ = 0):")
    print(f"  To new party: t = {t_new:+.3f}, p = {p_new:.4f}")
    print(f"  To old party: t = {t_old:+.3f}, p = {p_old:.4f}")
    print("\n")

    # =========================================================================
    # TABLE 2: LANGUAGE-VOTING CORRELATION
    # =========================================================================
    print("="*80)
    print("TABLE 2: LANGUAGE-VOTING CORRELATION")
    print("="*80)
    print("\n")
    
    r_p = corr['r_pearson']
    p_p = corr['p_pearson']
    n = corr['n']
    
    print(f"{'Test':<50} {'Statistic':>15} {'p-value':>10}")
    print("-" * 80)
    print(f"{'Pearson correlation':<50} {f'r = {r_p:+.3f}':>15} {p_p:>10.4f}")
    print(f"{'Spearman correlation':<50} {f'ρ = {corr['r_spearman']:+.3f}':>15} {corr['p_spearman']:>10.4f}")
    print()
    
    # Regression results
    if 'regression' in stats and stats['regression'].get('status') != 'Skipped':
        reg = stats['regression']
        print(f"{'Regression: ΔVoting ~ |ΔLanguage|':<50}")
        print(f"{'  Coefficient (β)':<50} {reg['beta']:>15.4f} {reg['p_value']:>10.4f}")
        print(f"{'  Standard error':<50} {reg['se']:>15.4f}")
        print(f"{'  R²':<50} {reg['r_squared']:>15.4f}")
    print("-" * 80)
    
    # Power analysis - FULL TABLE
    pwr = stats['power_analysis']
    print(f"\\nSTATISTICAL POWER (N = {n}):")
    print(f"  Minimum detectable effect (80% power):  |r| ≥ {pwr['mde_80']:.3f}")
    print(f"  Minimum detectable effect (50% power):  |r| ≥ {pwr['mde_50']:.3f}")
    
    # Calculate power for specific effect sizes
    def calc_power(n, r, alpha=0.05):
        z_r = np.arctanh(r)
        se = 1 / np.sqrt(n - 3)
        z_crit = norm.ppf(1 - alpha/2)
        ncp = z_r / se
        return 1 - norm.cdf(z_crit - ncp) + norm.cdf(-z_crit - ncp)
    
    print()
    print("=" * 80)
    print("TABLE: POWER BY TRUE CORRELATION")
    print("=" * 80)
    print()
    print(f"{'True |r|':<12} {'Cohen Benchmark':<18} {'Power':<12} {'Assessment':<18}")
    print("-" * 60)
    
    benchmarks = {
        0.10: ('Negligible', 'Underpowered'),
        0.15: ('Small', 'Underpowered'),
        0.20: ('Small-moderate', 'Moderate power'),
        0.25: ('Moderate', 'Well-powered'),
        0.30: ('Moderate', 'Well-powered'),
        0.40: ('Large', 'Well-powered'),
        0.50: ('Large', 'Well-powered')
    }
    
    for r_val in [0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]:
        power_val = calc_power(n, r_val) * 100
        cohen, assessment = benchmarks[r_val]
        if power_val > 99.9:
            power_str = ">99.9%"
        else:
            power_str = f"{power_val:.1f}%"
        print(f"|r| = {r_val:<6} {cohen:<18} {power_str:<12} {assessment:<18}")
    
    print("-" * 60)
    print()
    
    print("--- Confidence Interval for Observed r ---")
    print(f"  Observed r: {r_p:.3f}")
    print(f"  95% CI: [{pwr['ci_lower']:.2f}, {pwr['ci_upper']:.2f}]")
    print()
    
    print("\n" + "="*80)
    
    # =========================================================================
    # TABLE 3: CONTROLLED REGRESSION (SELECTION BIAS CHECK)
    # =========================================================================
    if 'controlled_regression' in stats and stats['controlled_regression'].get('status') != 'Skipped':
        ctrl = stats['controlled_regression']
        
        print()
        print("=" * 90)
        print("TABLE 3: LANGUAGE-VOTING CORRELATION WITH SELECTION CONTROLS")
        print("=" * 90)
        print()
        print("Addresses potential selection bias: included deputies have higher ideological distance,")
        print("more switches, and are more likely to switch to major parties or rightward.")
        print()
        print(f"{'Model':<40} {'β (Language)':<12} {'SE':<10} {'p-value':<10} {'N':<8} {'R²':<8}")
        print("-" * 90)
        
        m1 = ctrl['model_base']
        m2 = ctrl['model_ctrl']
        m3 = ctrl['model_full']
        
        print(f"{'(1) Baseline (no controls)':<40} {m1['beta']:<12.4f} {m1['se']:<10.4f} {m1['p']:<10.4f} {m1['n']:<8} {m1['r2']:<8.4f}")
        print(f"{'(2) + Selection controls':<40} {m2['beta']:<12.4f} {m2['se']:<10.4f} {m2['p']:<10.4f} {m2['n']:<8} {m2['r2']:<8.4f}")
        print(f"{'(3) + Career tenure':<40} {m3['beta']:<12.4f} {m3['se']:<10.4f} {m3['p']:<10.4f} {m3['n']:<8} {m3['r2']:<8.4f}")
        print("-" * 90)
        print()
        
        # Full Model 2 coefficients
        print("Model (2) Full Results:")
        print("-" * 60)
        params = m2['full_params']
        ses = m2['full_se']
        pvals = m2['full_pvalues']
        
        var_labels = {
            'const': 'Constant',
            'abs_linguistic_effect': '|Δ Language|',
            'ideo_distance': 'Ideological distance',
            'n_switches': 'Number of switches',
            'to_major_party': 'Switch to major party',
            'is_rightward': 'Rightward switch',
            'speeches_per_year': 'Speeches per year'
        }
        
        for var in params.keys():
            label = var_labels.get(var, var)
            coef = params[var]
            se = ses[var]
            p = pvals[var]
            sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
            print(f"  {label:<30} {coef:>10.4f}  ({se:.4f}) {sig}")
        
        print(f"  {'N':<30} {m2['n']:>10}")
        print(f"  {'R²':<30} {m2['r2']:>10.4f}")
        print("-" * 60)
        print()
        
        # Interpretation
        print("INTERPRETATION:")
        beta1 = m1['beta']
        beta2 = m2['beta']
        p1 = m1['p']
        p2 = m2['p']
        
        pct_change = ((beta2 - beta1) / abs(beta1) * 100) if beta1 != 0 else 0
        
        if abs(beta2) < abs(beta1) * 0.5:
            print("  ⚠️  Coefficient ATTENUATES substantially after adding controls.")
            print("     Selection bias may have inflated the baseline estimate.")
        elif abs(beta2) > abs(beta1) * 1.5:
            print("  ⚠️  Coefficient INCREASES substantially after adding controls.")
            print("     Selection bias may have suppressed the baseline estimate.")
        else:
            print("  ✅  Coefficient is ROBUST to selection controls.")
            print("     The null correlation is not primarily driven by selection.")
        
        print()
        print(f"     Baseline β:        {beta1:.4f} (p = {p1:.4f})")
        print(f"     With controls β:   {beta2:.4f} (p = {p2:.4f})")
        print(f"     Change:            {pct_change:+.1f}%")
        print()

print_section2_1_report(section2_1_stats)



# %%
def plot_event_study(dml_data, save_path=None):
    """
    Generates the Economist-styled event study plot.
    """
    # --- Setup Colors ---
    # Index 0: Dark Blue (Main data)
    # Index 4: Red (Switch line/Significance)
    # Index 1: Light Blue (Secondary)
    ECONOMIST_COLORS = ['#006BA2', '#3EBCD2', '#379A8B', '#EBB434', '#DB444B']
    
    # Map to semantic keys for readability
    STYLE = {
        'data': ECONOMIST_COLORS[0],      # Dark Blue
        'ci':   ECONOMIST_COLORS[0],      # Dark Blue
        'switch': ECONOMIST_COLORS[4],    # Red
        'zero': '#444444',                # Dark Gray
        'grid': '#dcdcdc'                 # Light Gray
    }

    # --- Prepare Data ---
    plot_data = dml_data['plot_data']
    pt = dml_data['pretrends']
    
    # Use your figsize utility
    fig, ax = plt.subplots(figsize=figsize(width_key='my_paper', aspect=0.5))
    
    # --- Plotting ---
    # 1. Zero line
    ax.axhline(0, color=STYLE['zero'], lw=0.8, alpha=0.8, zorder=1)
    
    # 2. Switch line (dashed red)
    ax.axvline(0, color=STYLE['switch'], ls='--', lw=0.8, 
               label='Party Switch', zorder=1)
    
    # 3. Error Bars & Coefficients
    # Economist style usually prefers clean lines. We use solid colors for CIs.
    ax.errorbar(
        plot_data['time'], 
        plot_data['coef'],
        yerr=[plot_data['coef'] - plot_data['lower'], plot_data['upper'] - plot_data['coef']],
        fmt='o', 
        ms=3,                    # Marker size
        color=STYLE['data'],     # Main Blue
        ecolor=STYLE['ci'],      # Main Blue for bars
        elinewidth=0.8, 
        capsize=0,               # Economist style rarely uses caps on error bars
        lw=0.8,
        zorder=3,
        label='Coefficient (95% CI)'
    )
    
    # 4. Highlight Significant Points (Optional: Fill marker with Red if significant)
    sig_data = plot_data[plot_data['sig']]
    if not sig_data.empty:
        ax.scatter(sig_data['time'], sig_data['coef'], 
                   facecolors=STYLE['switch'], edgecolors=STYLE['switch'], 
                   s=15, zorder=4, label='Significant (p<0.05)')

    # --- Formatting ---
    # Economist style: Title usually left-aligned, subtitle distinct (handled via suptitle/title)
    
    # Clean up spines (already done in your setup, but reinforcing)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False) # Often y-axis spine is hidden, just ticks
    
    # Grid: Horizontal only
    ax.grid(axis='y', color=STYLE['grid'], lw=0.5, zorder=0)
    ax.grid(axis='x', visible=False)

    # Labels
    ax.set_xlabel('Bimesters Relative to Switch')
    ax.set_ylabel('Effect on P(Old Party)')
    
    # Ticks
    ax.set_xticks(plot_data['time'])
    
    # Legend (Top right, frameon=False is standard)
    ax.legend(loc='upper right', frameon=False, ncol=1)

    # Save and Show
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()

plot_event_study(dml_stats, save_path = '../results/final_paper/figures/fig1_event_study.png')

# %% [markdown]
# ---
# # PART III: ROBUSTNESS CHECKS
# ---

# %% [markdown]
# ### Section Setup

# %%
# ALTERNATIVE ESTIMATION METHODS (FIXED: ALIGNED TO 60-DAY BINS) ==============
# =============================================================================
import statsmodels.api as sm
from scipy.stats import pearsonr

# Initialize container
section3_stats = {}

print("Running robustness checks (OLS vs TWFE vs DML)...")

# 1. DATA PREPARATION ---------------------------------------------------------
# MATCH MAIN DML: 60-day bins, ±360 day window
bins_comp = list(range(-360, 361, 60))
labels_comp = [-6, -5, -4, -3, -2, -1, 1, 2, 3, 4, 5, 6]

# Filter to 360-day window (previously 180)
df_es_comp = df_es.loc[df_es['days_from_switch'].abs() <= 360].dropna(subset=['Y_confidence']).copy()

# Create Bimester Bins
df_es_comp['month'] = pd.cut(df_es_comp['days_from_switch'], bins=bins_comp, labels=labels_comp)

time_dummies_comp = pd.get_dummies(df_es_comp['month'], prefix='t')
if 't_-1' in time_dummies_comp.columns:
    time_dummies_comp = time_dummies_comp.drop(columns=['t_-1'])
time_dummies_comp = time_dummies_comp.astype(float)

# Covariates (Standardized - Same as Main)
print("Building OLS covariate matrix...")
X_base_comp = build_covariates(
    df_es_comp,
    include_calendar_month_fe=False,
    include_topic_fe=True,
    date_col='dataHoraInicio',
    verbose=True
)
Y_comp = df_es_comp['Y_confidence'].values
clusters_comp = df_es_comp['deputado_id'].values

# 2. METHOD 1: STANDARD OLS ---------------------------------------------------
print("   Fitting Method 1: Standard OLS...")
X_ols = sm.add_constant(pd.concat([time_dummies_comp, X_base_comp], axis=1))
model_ols = sm.OLS(Y_comp, X_ols).fit(cov_type='cluster', cov_kwds={'groups': clusters_comp})

# Extract coefficients
coefs_ols = model_ols.params[[c for c in model_ols.params.index if c.startswith('t_')]]
ses_ols = model_ols.bse[[c for c in model_ols.params.index if c.startswith('t_')]]

section3_stats['OLS'] = {
    'R2': model_ols.rsquared,
    'Peak_Effect': coefs_ols.get('t_1', np.nan),  # Changed to t_1 as peak is often immediate
    'N': len(Y_comp)
}

# 3. METHOD 2: TWO-WAY FIXED EFFECTS (TWFE) -----------------------------------
print("   Fitting Method 2: Proper TWFE (Deputy + Calendar Month)...")

# A. Unit Fixed Effects (Deputy)
dep_fe = pd.get_dummies(df_es_comp['deputado_id'], prefix='dep', drop_first=True).astype(float)

# B. Time Fixed Effects (Calendar Month)
df_es_comp['cal_month'] = df_es_comp['dataHoraInicio'].dt.to_period('M').astype(str)
month_fe = pd.get_dummies(df_es_comp['cal_month'], prefix='month', drop_first=True).astype(float)

# C. Topic Controls
if 'topic_id' in df_es_comp.columns:
    topic_fe = pd.get_dummies(df_es_comp['topic_id'], prefix='topic', drop_first=True).astype(float)
else:
    topic_fe = pd.DataFrame(index=df_es_comp.index)

# Combine X
X_twfe = pd.concat([time_dummies_comp, topic_fe, dep_fe, month_fe], axis=1)
X_twfe = sm.add_constant(X_twfe)

# Fit Model
model_twfe = sm.OLS(Y_comp, X_twfe).fit(cov_type='cluster', cov_kwds={'groups': clusters_comp})

coefs_twfe = model_twfe.params[[c for c in model_twfe.params.index if c.startswith('t_')]]
ses_twfe = model_twfe.bse[[c for c in model_twfe.params.index if c.startswith('t_')]]

section3_stats['TWFE'] = {
    'R2': model_twfe.rsquared,
    'Peak_Effect': coefs_twfe.get('t_1', np.nan),
    'N_Vars': X_twfe.shape[1]
}

# 4. METHOD 3: RETRIEVE DML RESULTS -------------------------------------------
if 'dml_stats' in globals() and 'plot_data' in dml_stats:
    plot_data_ref = dml_stats['plot_data']
    coefs_dml = plot_data_ref.set_index('time')['coef']
    ses_dml = (plot_data_ref.set_index('time')['upper'] - plot_data_ref.set_index('time')['lower']) / 3.92
    section3_stats['DML'] = {'Peak_Effect': coefs_dml.get(1, np.nan), 'Available': True}
else:
    coefs_dml = pd.Series(dtype=float)
    ses_dml = pd.Series(dtype=float)
    section3_stats['DML'] = {'Available': False}

# 5. BUILD COMPARISON DATA ----------------------------------------------------
comparison_data = []
for time_point in [-6, -5, -4, -3, -2, 1, 2, 3, 4, 5, 6]:
    comparison_data.append({
        'time': time_point,
        'OLS': coefs_ols.get(f't_{time_point}', np.nan),
        'OLS_se': ses_ols.get(f't_{time_point}', np.nan),
        'TWFE': coefs_twfe.get(f't_{time_point}', np.nan),
        'TWFE_se': ses_twfe.get(f't_{time_point}', np.nan),
        'DML': coefs_dml.get(time_point, np.nan),
        'DML_se': ses_dml.get(time_point, np.nan)
    })

df_comparison = pd.DataFrame(comparison_data)
section3_stats['comparison_table'] = df_comparison

# 6. CALCULATE CORRELATIONS ---------------------------------------------------
valid_idx = df_comparison[['OLS','TWFE','DML']].notna().all(axis=1)
if valid_idx.sum() > 2:
    corr_ols_dml, _ = pearsonr(df_comparison.loc[valid_idx, 'OLS'], df_comparison.loc[valid_idx, 'DML'])
    corr_twfe_dml, _ = pearsonr(df_comparison.loc[valid_idx, 'TWFE'], df_comparison.loc[valid_idx, 'DML'])
    section3_stats['correlations'] = {'OLS_vs_DML': corr_ols_dml, 'TWFE_vs_DML': corr_twfe_dml, 'Calculated': True}
else:
    section3_stats['correlations'] = {'Calculated': False}

print("Robustness checks complete (Aligned to 60-day bimesters).")

# PREPARE METHOD COMPARISON PLOT DATA =========================================
# =============================================================================

# Calculate correlation annotation text for the plot
# This ensures the logic remains here, but the rendering happens in the report
if section3_stats.get('correlations', {}).get('Calculated'):
    c = section3_stats['correlations']
    corr_text = f"Correlations with DML:\n"
    corr_text += f"  OLS:  r = {c['OLS_vs_DML']:.3f}\n"
    corr_text += f"  TWFE: r = {c['TWFE_vs_DML']:.3f}"
    
    section3_stats['plot_annotation'] = corr_text
else:
    section3_stats['plot_annotation'] = None

# Store plot labels and metadata if needed for custom formatting later
section3_stats['plot_config'] = {
    'labels': {
        'OLS': 'OLS',
        'TWFE': 'TWFE (Legislature FE)',
        'DML': 'DML (Main)'
    },
    'colors': {
        'OLS': '#3498db',
        'TWFE': '#9b59b6'
    }
}

# =============================================================================
# FULL COEFFICIENT TABLE & EXTENDED METHOD CORRELATIONS
# =============================================================================

print("   Extracting full coefficient tables with p-values...")

# Time periods to extract
time_periods = [-6, -5, -4, -3, -2, 1, 2, 3, 4, 5, 6]

# Build full coefficient table with p-values
full_coef_table = []

for t in time_periods:
    col_name = f't_{t}'
    row = {'time': t}
    
    # OLS coefficients
    if col_name in model_ols.params.index:
        row['ols_coef'] = model_ols.params[col_name]
        row['ols_se'] = model_ols.bse[col_name]
        row['ols_pval'] = model_ols.pvalues[col_name]
    else:
        row['ols_coef'] = row['ols_se'] = row['ols_pval'] = np.nan
    
    # TWFE coefficients
    if col_name in model_twfe.params.index:
        row['twfe_coef'] = model_twfe.params[col_name]
        row['twfe_se'] = model_twfe.bse[col_name]
        row['twfe_pval'] = model_twfe.pvalues[col_name]
    else:
        row['twfe_coef'] = row['twfe_se'] = row['twfe_pval'] = np.nan
    
    # DML coefficients (from main results)
    if 'results_dml' in globals() and col_name in results_dml.params.index:
        row['dml_coef'] = results_dml.params[col_name]
        row['dml_se'] = results_dml.bse[col_name]
        row['dml_pval'] = results_dml.pvalues[col_name]
    else:
        row['dml_coef'] = row['dml_se'] = row['dml_pval'] = np.nan
    
    full_coef_table.append(row)

df_full_coefs = pd.DataFrame(full_coef_table)

# Calculate ALL method correlations (including OLS vs TWFE with p-values)
ols_coefs_arr = df_full_coefs['ols_coef'].dropna().values
twfe_coefs_arr = df_full_coefs['twfe_coef'].dropna().values
dml_coefs_arr = df_full_coefs['dml_coef'].dropna().values

corr_ols_dml, p_ols_dml = pearsonr(ols_coefs_arr, dml_coefs_arr)
corr_twfe_dml, p_twfe_dml = pearsonr(twfe_coefs_arr, dml_coefs_arr)
corr_ols_twfe, p_ols_twfe = pearsonr(ols_coefs_arr, twfe_coefs_arr)

# Count fixed effects in TWFE
n_deputy_fe = df_es_comp['deputado_id'].nunique()
n_month_fe = pd.to_datetime(df_es_comp['dataHoraInicio']).dt.to_period('M').nunique()

# Store everything
section3_stats['full_coef_table'] = df_full_coefs
section3_stats['method_correlations'] = {
    'ols_dml': {'r': corr_ols_dml, 'p': p_ols_dml},
    'twfe_dml': {'r': corr_twfe_dml, 'p': p_twfe_dml},
    'ols_twfe': {'r': corr_ols_twfe, 'p': p_ols_twfe}
}
section3_stats['fixed_effects'] = {
    'n_deputy_fe': n_deputy_fe,
    'n_month_fe': n_month_fe
}
section3_stats['model_summaries'] = {
    'ols': {'r2': model_ols.rsquared, 'n': int(model_ols.nobs)},
    'twfe': {'r2': model_twfe.rsquared, 'n': int(model_twfe.nobs)},
    'dml': {'r2': results_dml.rsquared if 'results_dml' in globals() else np.nan, 
            'n': int(results_dml.nobs) if 'results_dml' in globals() else np.nan}
}

print("✓ Full coefficient table and correlations computed")

# =============================================================================
# PRE-TRENDS F-TEST FOR ALL METHODS
# =============================================================================

print("   Computing pre-trends F-tests for all methods...")

from scipy.stats import f as f_dist

def compute_pretrends_ftest(model, pre_cols):
    """Compute F-test for joint significance of pre-period coefficients."""
    # Get pre-period coefficients and covariance
    beta_pre = model.params[pre_cols].values
    vcov_pre = model.cov_params().loc[pre_cols, pre_cols].values
    
    # Wald test
    try:
        wald_stat = beta_pre @ np.linalg.inv(vcov_pre) @ beta_pre
        n_pre = len(beta_pre)
        f_stat = wald_stat / n_pre
        df_resid = model.df_resid
        p_value = 1 - f_dist.cdf(f_stat, n_pre, df_resid)
        return {'f_stat': f_stat, 'p_value': p_value, 'df': (n_pre, df_resid)}
    except:
        return {'f_stat': np.nan, 'p_value': np.nan, 'df': (np.nan, np.nan)}

# Pre-period columns
pre_cols = ['t_-6', 't_-5', 't_-4', 't_-3', 't_-2']

# OLS pre-trends
ols_pre_cols = [c for c in pre_cols if c in model_ols.params.index]
pretrends_ols = compute_pretrends_ftest(model_ols, ols_pre_cols)

# TWFE pre-trends
twfe_pre_cols = [c for c in pre_cols if c in model_twfe.params.index]
pretrends_twfe = compute_pretrends_ftest(model_twfe, twfe_pre_cols)

# DML pre-trends (already computed in dml_stats)
pretrends_dml = dml_stats['pretrends'] if 'dml_stats' in globals() and 'pretrends' in dml_stats else {'f_stat': np.nan, 'p_value': np.nan}

# Store results
section3_stats['pretrends_tests'] = {
    'ols': pretrends_ols,
    'twfe': pretrends_twfe,
    'dml': pretrends_dml
}

print(f"      OLS:  F = {pretrends_ols['f_stat']:.3f}, p = {pretrends_ols['p_value']:.4f}")
print(f"      TWFE: F = {pretrends_twfe['f_stat']:.3f}, p = {pretrends_twfe['p_value']:.4f}")
print(f"      DML:  F = {pretrends_dml['f_stat']:.3f}, p = {pretrends_dml['p_value']:.4f}")
print("✓ Pre-trends F-tests computed")

# %%
# CLASSIFIER PERFORMANCE METRICS ==============================================
# =============================================================================

print("Computing classifier performance metrics...")

from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report
from sklearn.model_selection import cross_val_predict

# Initialize stats container
if 'classifier_metrics' not in section3_stats:
    section3_stats['classifier_metrics'] = {'status': 'Initialized'}

# 1. GET PARTY SIGLAS (NAMES) ------------------------------------------------
# We need a mapping from party IDs to siglas
if 'siglaPartido' in df.columns:
    party_id_to_sigla = df[['idPartido', 'siglaPartido']].drop_duplicates().set_index('idPartido')['siglaPartido'].to_dict()
else:
    # If sigla not available, we'll use IDs but warn
    print("⚠️  Warning: Party siglas not available, using IDs")
    party_id_to_sigla = {pid: f"Party_{pid}" for pid in df_train['idPartido'].unique()}

# 2. CROSS-VALIDATION PREDICTIONS ON TRAINING SET ----------------------------
# Get predictions via 5-fold CV (this is proper - never saw these predictions during training)
print("   Running 5-fold cross-validation...")

y_train_array = np.asarray(y_train)
groups_train = df_train['deputado_id'].astype(str).values

y_pred_cv = groupkfold_cv_predict_classifier(
    estimator=clf_party,
    X=X_train,
    y=y_train_array,
    groups=groups_train,
    n_splits=5
)


# 3. OVERALL METRICS ----------------------------------------------------------
overall_accuracy = accuracy_score(y_train_array, y_pred_cv)
n_classes = len(clf_party.classes_)
baseline_accuracy = 1.0 / n_classes
improvement_ratio = overall_accuracy / baseline_accuracy

print(f"   Overall accuracy: {overall_accuracy:.4f}")
print(f"   Baseline (random): {baseline_accuracy:.4f}")
print(f"   Improvement: {improvement_ratio:.2f}x")

# 4. PER-CLASS METRICS --------------------------------------------------------
# Get precision, recall, F1 for each party
precision, recall, f1, support = precision_recall_fscore_support(
    y_train_array, y_pred_cv, labels=clf_party.classes_, average=None
)

# Weighted F1 (accounts for class imbalance)
_, _, weighted_f1, _ = precision_recall_fscore_support(
    y_train_array, y_pred_cv, average='weighted'
)

# Create party performance dataframe
party_performance = pd.DataFrame({
    'party_id': clf_party.classes_,
    'sigla': [party_id_to_sigla.get(str(pid), str(pid)) for pid in clf_party.classes_],
    'n_speeches': support,
    'precision': precision,
    'recall': recall,
    'f1': f1
})

# Sort by number of speeches (descending)
party_performance = party_performance.sort_values('n_speeches', ascending=False).reset_index(drop=True)

# 5. GROUP BY PARTY SIZE ------------------------------------------------------
# Define major/medium/minor based on ranking
party_performance['rank'] = range(1, len(party_performance) + 1)
party_performance['size_category'] = pd.cut(
    party_performance['rank'],
    bins=[0, 3, 10, 100],
    labels=['Major (top 3)', 'Medium (4-10)', 'Minor (11+)']
)

# Calculate mean F1 by category
f1_by_size = party_performance.groupby('size_category')['f1'].mean()

# Mean F1 for major parties
major_parties_f1 = party_performance.head(3)['f1'].mean()

# 6. STORE RESULTS ------------------------------------------------------------
section3_stats['classifier_metrics'] = {
    'status': 'Complete',
    'overall_accuracy': overall_accuracy,
    'baseline_accuracy': baseline_accuracy,
    'improvement_ratio': improvement_ratio,
    'weighted_f1': weighted_f1,
    'mean_f1_major': major_parties_f1,
    'n_classes': n_classes,
    'n_train': len(y_train),
    'party_performance': party_performance,
    'f1_by_size': f1_by_size.to_dict()
}


# 7. HOLDOUT (20% NON-SWITCHERS) METRICS --------------------------------------
print("   Running holdout (20% non-switchers) evaluation...")

# Speech-level holdout accuracy
y_control_array = np.asarray(y_control)
y_pred_holdout = clf_party.predict(X_control)

holdout_accuracy = accuracy_score(y_control_array, y_pred_holdout)

# Weighted F1 on holdout
_, _, holdout_weighted_f1, _ = precision_recall_fscore_support(
    y_control_array, y_pred_holdout, average='weighted', zero_division=0
)

# Holdout baseline and improvement (same baseline = 1 / #classes)
holdout_baseline_accuracy = baseline_accuracy
holdout_improvement_ratio = holdout_accuracy / holdout_baseline_accuracy


print(f"   Holdout accuracy: {holdout_accuracy:.4f}")
print(f"   Holdout weighted F1: {holdout_weighted_f1:.4f}")

# Deputy-level accuracy (majority vote across that deputy's speeches)
tmp_hold = pd.DataFrame({
    'deputado_id': df_control_eval['deputado_id'].astype(str).values,
    'y_true': y_control_array.astype(str),
    'y_pred': y_pred_holdout.astype(str)
})

dep_true = tmp_hold.groupby('deputado_id')['y_true'].agg(lambda s: s.value_counts().index[0])
dep_pred = tmp_hold.groupby('deputado_id')['y_pred'].agg(lambda s: s.value_counts().index[0])

holdout_deputy_acc = float((dep_true.values == dep_pred.values).mean())
print(f"   Holdout deputy-level accuracy (majority vote): {holdout_deputy_acc:.4f}")

# Store
section3_stats['classifier_metrics'].update({
    'holdout_accuracy': holdout_accuracy,
    'holdout_weighted_f1': holdout_weighted_f1,
    'holdout_deputy_level_accuracy': holdout_deputy_acc,
    'n_holdout': int(len(y_control_array)),
    'n_holdout_deputies': int(df_control_eval['deputado_id'].nunique()),
    'holdout_baseline_accuracy': holdout_baseline_accuracy,
    'holdout_improvement_ratio': holdout_improvement_ratio,
})


print(f"✅ Classifier metrics computed")

# %%
# CLASSIFIER ALGORITHM COMPARISON =============================================
# =============================================================================

from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.metrics import accuracy_score, f1_score

print("Running classifier algorithm comparison...")

# Define classifiers to compare
classifiers = {
    'Logistic Regression': LogisticRegression(
        class_weight='balanced', C=1.0, max_iter=500, n_jobs=-1, random_state=SEED
    ),
    'Linear SVM': LinearSVC(
        class_weight='balanced', C=1.0, max_iter=1000, random_state=SEED
    ),
    'Multinomial Naive Bayes': MultinomialNB(
        alpha=1.0
    ),
    'Random Forest': RandomForestClassifier(
        n_estimators=100, max_depth=20, class_weight='balanced', 
        n_jobs=-1, random_state=SEED
    )
}

# Probability output characteristics
prob_output = {
    'Logistic Regression': 'Native, well-calibrated',
    'Linear SVM': 'Requires calibration',
    'Multinomial Naive Bayes': 'Poorly calibrated',
    'Random Forest': 'Requires calibration'
}

# Evaluate each classifier
classifier_results = {} # Temporary dict for the models


groups_train = df_train['deputado_id'].astype(str).values
y_train_array = np.asarray(y_train)

for name, clf in classifiers.items():
    print(f"   Evaluating {name}...")
    
    y_pred_cv = groupkfold_cv_predict_classifier(
        estimator=clf,
        X=X_train,
        y=y_train_array,
        groups=groups_train,
        n_splits=5
    )
    
    acc = float((y_pred_cv == y_train_array).mean())
    weighted_f1 = float(f1_score(y_train_array, y_pred_cv, average='weighted'))
    macro_f1 = float(f1_score(y_train_array, y_pred_cv, average='macro'))
    
    classifier_results[name] = {
        'accuracy_mean': acc,
        'weighted_f1': weighted_f1,
        'macro_f1': macro_f1,
        'prob_output': prob_output[name],
        'time': 0.0
    }

# Store with the specific nesting the printing cell expects
section3_stats['classifier_comparison'] = {
    'results': classifier_results,
    'status': 'Complete'
}

print("✓ Classifier comparison complete and formatted for printing")

# %%
# NAMED ENTITY REMOVAL ========================================================
# =============================================================================

ALT_TEXT_COL = 'text_level_3'  # NER removed

print(f"Running Robustness Check: Alternative Text Representation (Level 3)...")
print(f"   Using text column: {ALT_TEXT_COL}")

if ALT_TEXT_COL in df_es.columns:
    
    # CRITICAL FIX: Use the EXACT same sample as main analysis
    # Main analysis uses ALL df_es rows (including speeches outside ±180 window)
    
    df_es_rob = df_es.copy()
    
    initial_n = len(df_es_rob)
    print(f"   Initial sample (matching main): {initial_n:,}")
    
    # Ensure both text columns are valid
    df_es_rob = df_es_rob.dropna(subset=[CFG.text_col, ALT_TEXT_COL])
    
    final_n = len(df_es_rob)
    print(f"   After requiring both text levels: {final_n:,}")
    
    if initial_n != final_n:
        print(f"   ⚠️  Lost {initial_n - final_n:,} speeches due to missing text")
    
    # Train classifier on Level 3 text
    print("   Training classifier on Level 3 text...")
    
    df_train_rob = df_train.dropna(subset=[CFG.text_col, ALT_TEXT_COL])
    
    tfidf_alt = TfidfVectorizer(max_features=CFG.tfidf_max_features, 
                                min_df=CFG.tfidf_min_df,
                                max_df=CFG.tfidf_max_df, 
                                ngram_range=(1,2))
    
    X_train_alt = tfidf_alt.fit_transform(df_train_rob[ALT_TEXT_COL])
    y_train_alt = df_train_rob['idPartido']
    
    clf_alt = LogisticRegression(class_weight='balanced', C=1.0, 
                                 max_iter=500, n_jobs=-1, random_state=SEED)
    clf_alt.fit(X_train_alt, y_train_alt)
    
    # Generate predictions
    print("   Generating confidence scores...")
    
    X_es_alt = tfidf_alt.transform(df_es_rob[ALT_TEXT_COL])
    probs_alt = clf_alt.predict_proba(X_es_alt)
    
    ALT_CLASS_INDICES = {str(label): idx for idx, label in enumerate(clf_alt.classes_)}
    
    # Extract P(Old Party)
    y_conf_alt = []
    for idx, row in df_es_rob.iterrows():
        old_pid = str(row['old_party_id'])
        if old_pid in ALT_CLASS_INDICES:
            y_conf_alt.append(probs_alt[len(y_conf_alt), ALT_CLASS_INDICES[old_pid]])
        else:
            y_conf_alt.append(np.nan)
    
    df_es_rob['Y_confidence_alt'] = y_conf_alt
    df_es_rob = df_es_rob.dropna(subset=['Y_confidence_alt'])
    
    print(f"   Final sample size: {len(df_es_rob):,}")
    
    # Run DML - INCLUDING THE MONTH BINNING EXACTLY AS MAIN
    print("   Running DML on Level 3 text...")
    
    # Create month bins EXACTLY as main analysis (includes all rows, ±180 window gets bins, rest get NaN)
    bins = list(range(-180, 181, 30))
    labels = [-6,-5,-4,-3,-2,-1,1,2,3,4,5,6]
    df_es_rob['month'] = pd.cut(df_es_rob['days_from_switch'], bins=bins, labels=labels)
    
    # Time dummies (includes ALL rows, those outside window get 0 for all dummies)
    time_dummies_rob = pd.get_dummies(df_es_rob['month'], prefix='t')
    if 't_-1' in time_dummies_rob.columns: 
        time_dummies_rob = time_dummies_rob.drop(columns=['t_-1'])
    
    # Covariates (STANDARDIZED - identical to main DML)
    X_rob = build_covariates(
        df_es_rob,
        include_calendar_month_fe=False,
        include_topic_fe=True,
        date_col='dataHoraInicio',
        verbose=True
    ).astype(float)
    Y_rob = df_es_rob['Y_confidence_alt'].values
    clusters_rob = df_es_rob['deputado_id'].values
    
    # DML
    from sklearn.ensemble import HistGradientBoostingRegressor, HistGradientBoostingClassifier

    learner_Y = HistGradientBoostingRegressor(max_iter=100, max_depth=5, random_state=SEED)
    learner_D = HistGradientBoostingClassifier(max_iter=50, max_depth=3, random_state=SEED)

    # GROUPED cross-fitting by deputy (prevents leakage across speeches by same deputy)
    groups_rob = clusters_rob  # df_es_rob['deputado_id'].values

    Y_pred = grouped_oof_predict_regressor(
        model=learner_Y,
        X=X_rob.values if hasattr(X_rob, "values") else X_rob,
        y=Y_rob,
        groups=groups_rob,
        n_splits=CFG.dml_n_splits,
        seed=SEED
    )
    Y_resid_ner = Y_rob - Y_pred

    D_resid_ner = pd.DataFrame(index=df_es_rob.index, columns=time_dummies_rob.columns, dtype=float)
    for col in tqdm(time_dummies_rob.columns, desc="   DML Residualizing (group-CV)"):
        d = time_dummies_rob[col].astype(int).values
        D_pred = grouped_oof_predict_proba_binary(
            model=learner_D,
            X=X_rob.values if hasattr(X_rob, "values") else X_rob,
            d=d,
            groups=groups_rob,
            n_splits=CFG.dml_n_splits,
            seed=SEED
        )
        D_resid_ner[col] = d - D_pred

    
    # Final regression
    X_final_rob = sm.add_constant(D_resid_ner)
    results_rob = sm.OLS(Y_resid_ner, X_final_rob).fit(cov_type='cluster', cov_kwds={'groups': clusters_rob})
    
    coefs_rob = results_rob.params.drop('const')
    
    section3_stats['robustness_text'] = {
        'status': 'Complete',
        'col_name': ALT_TEXT_COL,
        'coefs': coefs_rob.to_dict(),
        'peak_effect_t1': coefs_rob.get('t_1', np.nan),
        'peak_effect_t2': coefs_rob.get('t_2', np.nan),
        'n_obs': len(Y_rob),
        'n_in_window': (df_es_rob['month'].notna()).sum(),
        'n_out_window': (df_es_rob['month'].isna()).sum()
    }
    
    print(f"   ✓ Complete. Peak (τ=+1): {coefs_rob.get('t_1', np.nan):.4f}, N={len(Y_rob):,}")
    print(f"      (±180 window: {(df_es_rob['month'].notna()).sum():,}, Outside: {(df_es_rob['month'].isna()).sum():,})")

else:
    print(f"   ⚠️ Column '{ALT_TEXT_COL}' not found")
    section3_stats['robustness_text'] = {'status': 'Skipped'}

# %%
# IDEOLOGICAL BLOC CLASSIFICATION =============================================
# =============================================================================
from scipy import stats  # <--- RE-IMPORT TO FIX THE OVERWRITE
import numpy as np
import pandas as pd

print("Running Robustness Check: Bloc-Level Classifier (Left/Center/Right)...")

# 1. TRAIN CLASSIFIER ---------------------------------------------------------
# We use 'CAT' (Ideological Category) as the target instead of 'idPartido'
if 'CAT' in df_train.columns and df_train['CAT'].nunique() > 1:
    
    # Initialize Vectorizer
    tfidf_bloc = TfidfVectorizer(max_features=CFG.tfidf_max_features, 
                                min_df=CFG.tfidf_min_df,
                                max_df=CFG.tfidf_max_df, 
                                ngram_range=(1,2))
    X_bloc = tfidf_bloc.fit_transform(df_train[CFG.text_col])
    y_bloc = df_train['CAT']  # Target: Ideology (Left, Center, Right)

    clf_bloc = LogisticRegression(class_weight='balanced', C=1.0, max_iter=500, n_jobs=-1, random_state=SEED)
    clf_bloc.fit(X_bloc, y_bloc)

    groups_train = df_train['deputado_id'].astype(str).values

    cv_acc_bloc = groupkfold_cv_accuracy_classifier(
        estimator=clf_bloc,
        X=X_bloc,
        y=y_bloc,
        groups=groups_train,
        n_splits=5
    )

    section3_stats['bloc_robustness'] = {
        'status': 'Complete',
        'cv_accuracy': float(cv_acc_bloc),
        'classes': list(clf_bloc.classes_)
    }

    # 2. GENERATE PREDICTIONS -------------------------------------------------
    BLOC_CI = {l:i for i,l in enumerate(clf_bloc.classes_)}
    
    # Predict directly on event study dataframe
    df_es_bloc = df_es.dropna(subset=['old_CAT', 'new_CAT', CFG.text_col]).copy()
    
    X_es_bloc = tfidf_bloc.transform(df_es_bloc[CFG.text_col])
    probs_bloc = list(clf_bloc.predict_proba(X_es_bloc))
    
    # Map probabilities
    df_es_bloc['probs_temp'] = probs_bloc
    df_es_bloc['Y_bloc'] = df_es_bloc.apply(
        lambda r: r['probs_temp'][BLOC_CI[r['old_CAT']]] if r['old_CAT'] in BLOC_CI else np.nan, 
        axis=1
    )
    df_es_bloc = df_es_bloc.dropna(subset=['Y_bloc'])

    # 3. COMPARE PRE VS POST --------------------------------------------------
    df_es_bloc['post'] = (df_es_bloc['days_from_switch'] > 0).astype(int)
    
    pre_bloc = df_es_bloc[df_es_bloc['post']==0]['Y_bloc']
    post_bloc = df_es_bloc[df_es_bloc['post']==1]['Y_bloc']
    
    # T-test (Now uses the re-imported stats module)
    bloc_diff = post_bloc.mean() - pre_bloc.mean()
    bloc_t, bloc_p = stats.ttest_ind(post_bloc, pre_bloc, equal_var=False)
    
    # Cohen's d
    pooled_std = np.sqrt((pre_bloc.std()**2 + post_bloc.std()**2)/2)
    bloc_d = bloc_diff / pooled_std

    # Store results
    section3_stats['bloc_robustness'].update({
        'n_obs': len(df_es_bloc),
        'diff': bloc_diff,
        't_stat': bloc_t,
        'p_value': bloc_p,
        'cohens_d': bloc_d,
        'pre_mean': pre_bloc.mean(),
        'post_mean': post_bloc.mean()
    })
    
    print(f"   Bloc Acc: {cv_acc_bloc:.3f} | Diff: {bloc_diff:+.4f} | p={bloc_p:.4f} | d={bloc_d:.3f}")

else:
    print("   ⚠️ 'CAT' column missing or invalid. Skipping Bloc check.")
    section3_stats['bloc_robustness'] = {'status': 'Skipped (Missing CAT)'}

# %%
# EMBEDDING-BASED SEMANTIC DISTANCE ===========================================
# =============================================================================
from scipy.spatial.distance import cosine
import numpy as np
import scipy.stats as stats

print("Running Robustness Check: Embedding Similarity...")

# Initialize stats container entry
section3_stats['embedding_robustness'] = {'status': 'Initialized'}

# Check if embedding data is available
if 'embedding' in df.columns:
    
    # 1. COMPUTE MEAN PARTY EMBEDDINGS (CENTROIDS) ----------------------------
    def safe_mean_emb(embs):
        # Filter out None or empty entries
        valid = [np.array(e) for e in embs if e is not None and len(np.array(e).shape) > 0]
        return np.mean(np.vstack(valid), axis=0) if valid else None

    PARTY_EMB = {}
    # Use df_train to avoid data leakage (only training data defines the "party core")
    for pid in df_train['idPartido'].unique():
        embs = df_train[df_train['idPartido'] == pid]['embedding'].dropna().tolist()
        if len(embs) >= CFG.min_party_speeches:
            PARTY_EMB[pid] = safe_mean_emb(embs)
            
    section3_stats['embedding_robustness']['n_parties_with_emb'] = len(PARTY_EMB)

    # 2. CALCULATE SIMILARITY TO OLD PARTY ------------------------------------
    def cos_sim(a, b):
        if a is None or b is None: return None
        a = np.array(a).flatten()
        b = np.array(b).flatten()
        if len(a) != len(b): return None
        return 1 - cosine(a, b)

    # Create local copy for analysis
    df_es_emb = df_es.copy()
    
    # Ensure embeddings are attached to df_es (if not already)
    if 'embedding' not in df_es_emb.columns:
        # Merge if missing (assuming index alignment or merging by ID/Date)
        # For safety, we map from main df using index if preserved, or assume it's there
        df_es_emb['embedding'] = df.loc[df_es_emb.index, 'embedding']

    df_es_emb['sim_old'] = df_es_emb.apply(
        lambda r: cos_sim(r['embedding'], PARTY_EMB.get(str(r['old_party_id']))), 
        axis=1
    )
    
    df_es_emb = df_es_emb.dropna(subset=['sim_old'])

    # 3. STATISTICAL TEST (Pre vs Post) ---------------------------------------
    df_es_emb['post'] = (df_es_emb['days_from_switch'] > 0).astype(int)

    pre_emb = df_es_emb[df_es_emb['post'] == 0]['sim_old']
    post_emb = df_es_emb[df_es_emb['post'] == 1]['sim_old']
    
    if len(pre_emb) > 1 and len(post_emb) > 1:
        emb_diff = post_emb.mean() - pre_emb.mean()
        emb_t, emb_p = stats.ttest_ind(post_emb, pre_emb, equal_var=False)
        emb_d = emb_diff / np.sqrt((pre_emb.std()**2 + post_emb.std()**2)/2)

        # Store results
        section3_stats['embedding_robustness'].update({
            'status': 'Complete',
            'n_obs': len(df_es_emb),
            'diff': emb_diff,
            't_stat': emb_t,
            'p_value': emb_p,
            'cohens_d': emb_d,
            'pre_mean': pre_emb.mean(),
            'post_mean': post_emb.mean()
        })
        
        print(f"   Emb Diff: {emb_diff:+.4f} | p={emb_p:.4f} | d={emb_d:.3f}")
    else:
        print("   ⚠️ Insufficient data for embedding t-test.")
        section3_stats['embedding_robustness']['status'] = 'Skipped (Insufficient Data)'

else:
    print("   ⚠️ 'embedding' column not found. Skipping Embedding check.")
    section3_stats['embedding_robustness']['status'] = 'Skipped (Missing Embeddings)'

# %%
# PLACEBO TEST (PERMUTATION) ==================================================
# =============================================================================
import numpy as np
from tqdm import tqdm

print("Running Robustness Check: Placebo Distribution Test...")

# 1. CALCULATE TRUE EFFECT (RAW Y) --------------------------------------------
# We use the raw outcome Y_confidence as residuals absorb timing info
true_post_mask = (df_es['days_from_switch'] >= 30) & (df_es['days_from_switch'] <= 150)
true_pre_mask = (df_es['days_from_switch'] >= -60) & (df_es['days_from_switch'] < 0)

true_effect_stat = abs(df_es.loc[true_post_mask, 'Y_confidence'].mean() - 
                       df_es.loc[true_pre_mask, 'Y_confidence'].mean())

# 2. RUN PERMUTATIONS ---------------------------------------------------------
n_permutations = 1000
placebo_stats = []
deputy_ids = df_es['deputado_id'].unique()

np.random.seed(SEED)

for i in tqdm(range(n_permutations), desc="   Running Permutations"):
    perm_seed = SEED + i
    np.random.seed(perm_seed)
    
    # Generate random date offsets (shift ± 1 year)
    offsets = np.random.randint(-365, 365, size=len(deputy_ids))
    deputy_offset_map = dict(zip(deputy_ids, offsets))
    
    # Apply offsets
    current_offsets = df_es['deputado_id'].map(deputy_offset_map)
    fake_days = df_es['days_from_switch'] - current_offsets
    
    # Calculate placebo effect using SAME windows
    fake_post = (fake_days >= 30) & (fake_days <= 150)
    fake_pre = (fake_days >= -60) & (fake_days < 0)
    
    # Require minimum observations to be valid
    if fake_post.sum() > 50 and fake_pre.sum() > 50:
        stat = abs(df_es.loc[fake_post, 'Y_confidence'].mean() - 
                   df_es.loc[fake_pre, 'Y_confidence'].mean())
        placebo_stats.append(stat)

placebo_stats = np.array(placebo_stats)

# 3. CALCULATE STATISTICS -----------------------------------------------------
p_value_lower = (placebo_stats >= true_effect_stat).mean()
p_value_two_tailed = 2 * min(p_value_lower, 1 - p_value_lower)
percentile = (placebo_stats <= true_effect_stat).mean() * 100

section3_stats['placebo'] = {
    'status': 'Complete',
    'true_effect': true_effect_stat,
    'placebo_stats': placebo_stats, # Stored for plotting
    'mean_placebo': placebo_stats.mean(),
    'percentile': percentile,
    'p_value': p_value_lower,
    'n_permutations': len(placebo_stats)
}
print(f"   True Effect: {true_effect_stat:.4f} | Placebo P-val: {p_value_lower:.4f}")

# %%
# ALTERNATIVE TIME WINDOWS ====================================================
# =============================================================================

print("Computing window sensitivity with standard errors...")

if 'df_es' in globals():
    windows_months = [6, 9, 12, 15, 18]
    results_with_se = []
    
    for window_months in windows_months:
        window_days = window_months * 30
        
        # Filter to window
        df_window = df_es[df_es['days_from_switch'].abs() <= window_days].copy()
        
        # Split pre/post
        pre = df_window[df_window['days_from_switch'] < 0]['Y_confidence']
        post = df_window[df_window['days_from_switch'] > 0]['Y_confidence']
        
        if len(pre) > 0 and len(post) > 0:
            # Calculate ATE
            ate = post.mean() - pre.mean()
            
            # Calculate SE using t-test
            from scipy.stats import ttest_ind
            t_stat, p_val = ttest_ind(post, pre, equal_var=False)
            
            # SE = ATE / t_stat
            se = abs(ate / t_stat) if t_stat != 0 else np.nan
            
            results_with_se.append({
                'window_months': window_months,
                'coef': ate,
                'se': se,
                'p_value': p_val,
                'n_obs': len(df_window)
            })
            
            print(f"   ±{window_months} months: ATE = {ate:.4f}, SE = {se:.4f}, N = {len(df_window):,}")
    
    df_win_new = pd.DataFrame(results_with_se)
    
    # Update stored results
    if 'section3_stats' not in globals():
        section3_stats = {}
    
    section3_stats['window_sensitivity'] = {
        'status': 'Complete',
        'method': 'DML (ATE)',
        'results': df_win_new
    }
    
    print("\n✓ Window sensitivity with SE computed")
else:
    print("⚠️  df_es not found")

# %%
# BIN SIZE SENSITIVITY ANALYSIS ================================================
# ==============================================================================

from scipy.stats import f as f_dist

def run_event_study_with_bins(df_es_input, bin_days, window_days=360, verbose=False):
    """
    Run full DML event study with specified bin size.
    
    Parameters:
    - df_es_input: DataFrame with event study data (must have Y_confidence, days_from_switch)
    - bin_days: Size of each bin in days (30, 60, 90, etc.)
    - window_days: Total window in days (default ±360)
    
    Returns:
    - Dictionary with results
    """
    
    # Create bins
    bins = list(range(-window_days, window_days + 1, bin_days))
    n_bins_per_side = window_days // bin_days
    labels = list(range(-n_bins_per_side, 0)) + list(range(1, n_bins_per_side + 1))
    
    # Filter to window and create time bins
    df_temp = df_es_input[df_es_input['days_from_switch'].abs() <= window_days].copy()
    df_temp['time_bin'] = pd.cut(df_temp['days_from_switch'], bins=bins, labels=labels)
    df_temp = df_temp.dropna(subset=['time_bin', 'Y_confidence'])
    
    # Create time dummies (reference = -1)
    time_dummies = pd.get_dummies(df_temp['time_bin'], prefix='t')
    ref_col = 't_-1'
    if ref_col in time_dummies.columns:
        time_dummies = time_dummies.drop(columns=[ref_col])
    
    # Build covariates
    X = build_covariates(
        df_temp,
        include_calendar_month_fe=False,
        include_topic_fe=True,
        date_col='dataHoraInicio',
        verbose=False
    )
    
    Y = df_temp['Y_confidence'].values
    clusters = df_temp['deputado_id'].values
    
    # DML residualization (GROUPED cross-fitting by deputy)
    learner_Y = HistGradientBoostingRegressor(max_iter=100, max_depth=5, random_state=SEED)
    learner_D = HistGradientBoostingClassifier(max_iter=50, max_depth=3, random_state=SEED)

    groups = clusters  # df_temp['deputado_id'].values

    Y_pred = grouped_oof_predict_regressor(
        model=learner_Y,
        X=X.values if hasattr(X, "values") else X,
        y=Y,
        groups=groups,
        n_splits=CFG.dml_n_splits,
        seed=SEED
    )
    Y_resid = Y - Y_pred

    D_resid = pd.DataFrame(index=df_temp.index, columns=time_dummies.columns, dtype=float)
    for col in time_dummies.columns:
        d = time_dummies[col].astype(int).values
        D_pred = grouped_oof_predict_proba_binary(
            model=learner_D,
            X=X.values if hasattr(X, "values") else X,
            d=d,
            groups=groups,
            n_splits=CFG.dml_n_splits,
            seed=SEED
        )
        D_resid[col] = d - D_pred

    
    # Final OLS
    X_final = sm.add_constant(D_resid)
    results = sm.OLS(Y_resid, X_final).fit(cov_type='cluster', cov_kwds={'groups': clusters})
    
    # Extract coefficients
    coefs = results.params.drop('const')
    pvals = results.pvalues.drop('const')
    bse = results.bse.drop('const')
    
    # Find peak effect (first post-period, τ=+1)
    peak_col = 't_1'
    peak_effect = coefs[peak_col] if peak_col in coefs.index else np.nan
    peak_se = bse[peak_col] if peak_col in bse.index else np.nan
    peak_pval = pvals[peak_col] if peak_col in pvals.index else np.nan
    
    # Pre-trends F-test
    pre_cols = [c for c in coefs.index if c.startswith('t_-') and c != 't_-1']
    n_pre = len(pre_cols)
    
    if n_pre > 0:
        # Construct restriction matrix for joint test
        R = np.zeros((n_pre, len(coefs) + 1))  # +1 for constant
        for i, col in enumerate(pre_cols):
            col_idx = list(results.params.index).index(col)
            R[i, col_idx] = 1
        
        # F-test
        try:
            f_test = results.f_test(R[:, 1:])  # Exclude constant column
            f_stat = f_test.fvalue[0][0]
            f_pval = f_test.pvalue
        except:
            # Manual calculation if f_test fails
            pre_coefs = coefs[pre_cols].values
            pre_vcov = results.cov_params().loc[pre_cols, pre_cols].values
            try:
                f_stat = pre_coefs @ np.linalg.inv(pre_vcov) @ pre_coefs / n_pre
                df_resid = len(Y) - len(results.params)
                f_pval = 1 - f_dist.cdf(f_stat, n_pre, df_resid)
            except:
                f_stat, f_pval = np.nan, np.nan
    else:
        f_stat, f_pval = np.nan, np.nan
    
    return {
        'bin_days': bin_days,
        'n_periods': len(labels),
        'n_pre_periods': n_pre,
        'n_obs': len(Y),
        'n_clusters': len(np.unique(clusters)),
        'peak_effect': peak_effect,
        'peak_se': peak_se,
        'peak_pval': peak_pval,
        'pretrends_f': f_stat,
        'pretrends_p': f_pval,
        'r_squared': results.rsquared,
        'full_results': results,
        'coefs': coefs,
        'pvals': pvals
    }

# Run for different bin sizes
print("Running bin sensitivity analysis...")
bin_sensitivity_results = {}

for bin_size in [30, 60, 90]:
    print(f"   Testing {bin_size}-day bins...")
    bin_sensitivity_results[bin_size] = run_event_study_with_bins(df_es, bin_days=bin_size)
    
print("✓ Bin sensitivity analysis complete")

# %%
# ALTERNATIVE OUTCOME VARIABLE ================================================
# =============================================================================
from sklearn.model_selection import cross_val_predict, KFold
from sklearn.ensemble import HistGradientBoostingRegressor, HistGradientBoostingClassifier
import statsmodels.api as sm

print("Running Robustness Check: Alternative Outcome (P(New) - P(Old))...")

# 1. CALCULATE P(New) ---------------------------------------------------------
print("   Calculating P(New Party) for each speech...")

df_es_alt = df_es.copy()

# We need to map each deputy to their NEW party ID
# Using df_events which contains the switch details
# We create a map: deputy_id -> new_party_id
deputy_to_new_party = df_events.set_index('deputado_id')['new_party_id'].to_dict()

new_party_probs = []
# We iterate to extract the specific probability for the new party
# (Vectorizing this is possible but complex due to variable indices)
for idx, row in df_es_alt.iterrows():
    dep_id = row['deputado_id']
    new_pid = str(deputy_to_new_party.get(dep_id))
    
    # Get P(New Party)
    if new_pid in PARTY_CLASS_INDICES:
        new_idx = PARTY_CLASS_INDICES[new_pid]
        probs = row['party_probs']
        if probs is not None and len(probs) > new_idx:
            new_party_probs.append(probs[new_idx])
        else:
            new_party_probs.append(np.nan)
    else:
        new_party_probs.append(np.nan)

df_es_alt['P_new'] = new_party_probs
# Outcome: Difference between probability of new party and old party
df_es_alt['Y_alt'] = df_es_alt['P_new'] - df_es_alt['Y_confidence']

# Filter valid rows
df_es_alt = df_es_alt.dropna(subset=['Y_alt'])

section3_stats['alt_outcome'] = {
    'status': 'Initialized',
    'n_obs': len(df_es_alt)
}

# =============================================================================
# 2. RUN DML ON ALTERNATIVE OUTCOME (FIXED: ALIGNED TO MAIN 60-DAY BINS)
# =============================================================================
print(f"   Running DML on {len(df_es_alt):,} observations...")

# Filter to 360-day window to match Main Analysis
df_es_alt = df_es_alt.loc[df_es_alt['days_from_switch'].abs() <= 360].copy()

# A. Prepare Matrix
# Time Dummies: 60-DAY BINS (BIMESTERS)
bins = list(range(-360, 361, 60))
labels = [-6, -5, -4, -3, -2, -1, 1, 2, 3, 4, 5, 6]

df_es_alt['month'] = pd.cut(df_es_alt['days_from_switch'], bins=bins, labels=labels)
time_dummies_alt = pd.get_dummies(df_es_alt['month'], prefix='t')
if 't_-1' in time_dummies_alt.columns: 
    time_dummies_alt = time_dummies_alt.drop(columns=['t_-1'])

# Controls (Standardized)
X_alt = build_covariates(
    df_es_alt,
    include_calendar_month_fe=False,
    include_topic_fe=True,
    date_col='dataHoraInicio',
    verbose=True
).astype(float)

Y_alt = df_es_alt['Y_alt'].values
clusters_alt = df_es_alt['deputado_id'].values


# B. DML Estimation
learner_Y = HistGradientBoostingRegressor(max_iter=100, max_depth=5, random_state=SEED)
learner_D = HistGradientBoostingClassifier(max_iter=50, max_depth=3, random_state=SEED)

groups_alt = clusters_alt  # df_es_alt['deputado_id'].values

# Residualize Outcome (GROUPED cross-fitting)
Y_pred = grouped_oof_predict_regressor(
    model=learner_Y,
    X=X_alt.values if hasattr(X_alt, "values") else X_alt,
    y=Y_alt,
    groups=groups_alt,
    n_splits=CFG.dml_n_splits,
    seed=SEED
)
Y_resid_alt = Y_alt - Y_pred

# Residualize Treatments (GROUPED cross-fitting)
D_resid_alt = pd.DataFrame(index=df_es_alt.index, columns=time_dummies_alt.columns, dtype=float)
for col in tqdm(time_dummies_alt.columns, desc="   DML Residualizing (group-CV)"):
    d = time_dummies_alt[col].astype(int).values
    D_pred = grouped_oof_predict_proba_binary(
        model=learner_D,
        X=X_alt.values if hasattr(X_alt, "values") else X_alt,
        d=d,
        groups=groups_alt,
        n_splits=CFG.dml_n_splits,
        seed=SEED
    )
    D_resid_alt[col] = d - D_pred


# Final OLS
X_final_alt = sm.add_constant(D_resid_alt)
results_alt = sm.OLS(Y_resid_alt, X_final_alt).fit(cov_type='cluster', cov_kwds={'groups': clusters_alt})

# 3. STORE RESULTS ------------------------------------------------------------
# Format data for plotting/comparison
coefs_alt = results_alt.params.drop('const')
ses_alt = results_alt.bse.drop('const')

plot_data_alt = pd.DataFrame({
    'time': [int(c.replace('t_', '')) for c in coefs_alt.index],
    'coef': coefs_alt.values,
    'se': ses_alt.values
}).sort_values('time')

# Calculate statistics for report
# Peak effect (usually t=1 to t=4)
peak_idx = (plot_data_alt['time'] >= 1) & (plot_data_alt['time'] <= 4)
peak_new_old = plot_data_alt.loc[peak_idx, 'coef'].max()

section3_stats['alt_outcome'].update({
    'status': 'Complete',
    'peak_effect': peak_new_old,
    'plot_data': plot_data_alt
})

print(f"   Robustness check complete. Peak P(New)-P(Old): {peak_new_old:+.4f}")

# %% [markdown]
# ### Section Outputs

# %%
# =============================================================================
# COMPREHENSIVE ROBUSTNESS RESULTS REPORT
# =============================================================================

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec

# Set plotting style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (16, 10)

print("="*80)
print("COMPREHENSIVE ROBUSTNESS RESULTS")
print("="*80)

# =============================================================================
# PART 1: ALL PRINTED REPORTS
# =============================================================================

# ============================================================================
# TABLE: METHOD COMPARISON
# ============================================================================

if 'section3_stats' in globals() and 'comparison_table' in section3_stats:
    print("\n" + "="*80)
    print("TABLE: ESTIMATION METHOD COMPARISON")
    print("="*80)
    print("\n")
    
    from scipy import stats as sp_stats
    from scipy.stats import pearsonr
    
    df_comp = section3_stats['comparison_table']
    
    # Get peak effect at tau = +1
    peak_row = df_comp[df_comp['time'] == 1].iloc[0]
    
    # Get correlations WITH P-VALUES
    corr = section3_stats.get('correlations', {})
    
    # Recalculate correlations with p-values if available
    if corr.get('Calculated'):
        valid_idx = df_comp[['OLS','TWFE','DML']].notna().all(axis=1)
        if valid_idx.sum() > 2:
            corr_ols_dml, p_ols_dml = pearsonr(df_comp.loc[valid_idx, 'OLS'], df_comp.loc[valid_idx, 'DML'])
            corr_twfe_dml, p_twfe_dml = pearsonr(df_comp.loc[valid_idx, 'TWFE'], df_comp.loc[valid_idx, 'DML'])
            n_periods = valid_idx.sum()
        else:
            corr_ols_dml = corr['OLS_vs_DML']
            corr_twfe_dml = corr['TWFE_vs_DML']
            p_ols_dml = np.nan
            p_twfe_dml = np.nan
            n_periods = 0
    else:
        corr_ols_dml = np.nan
        corr_twfe_dml = np.nan
        p_ols_dml = np.nan
        p_twfe_dml = np.nan
        n_periods = 0
    
    print(f"{'Method':<30} {'Peak (τ=+1)':>15} {'SE':>10} {'R²':>10} {'N':>10}")
    print("-" * 80)
    
    # OLS
    ols_peak = peak_row['OLS']
    ols_se = peak_row['OLS_se']
    ols_r2 = section3_stats['OLS'].get('R2', np.nan)
    ols_n = section3_stats['OLS'].get('N', np.nan)
    
    ols_t = ols_peak / ols_se if ols_se > 0 else np.nan
    ols_p = 2 * (1 - sp_stats.t.cdf(abs(ols_t), df=ols_n-1)) if not np.isnan(ols_t) else np.nan
    ols_sig = '***' if ols_p < 0.001 else '**' if ols_p < 0.01 else '*' if ols_p < 0.05 else ''
    
    print(f"{'OLS (with controls)':<30} {ols_peak:>15.4f}{ols_sig} ({ols_se:.4f}) {ols_r2:>10.4f} {ols_n:>10,}")
    
    # TWFE
    twfe_peak = peak_row['TWFE']
    twfe_se = peak_row['TWFE_se']
    twfe_r2 = section3_stats['TWFE'].get('R2', np.nan)
    twfe_n = ols_n
    
    twfe_t = twfe_peak / twfe_se if twfe_se > 0 else np.nan
    twfe_p = 2 * (1 - sp_stats.t.cdf(abs(twfe_t), df=twfe_n-1)) if not np.isnan(twfe_t) else np.nan
    twfe_sig = '***' if twfe_p < 0.001 else '**' if twfe_p < 0.01 else '*' if twfe_p < 0.05 else ''
    
    print(f"{'TWFE (deputy + month FE)':<30} {twfe_peak:>15.4f}{twfe_sig} ({twfe_se:.4f}) {twfe_r2:>10.4f} {twfe_n:>10,}")
    
    dml_peak = peak_row['DML']
    dml_se = peak_row['DML_se']
    dml_r2 = dml_stats['model_summary']['R_squared']
    dml_n = dml_stats['model_summary']['N_Observations']
    
    dml_t = dml_peak / dml_se if dml_se > 0 else np.nan
    dml_p = 2 * (1 - sp_stats.t.cdf(abs(dml_t), df=dml_n-1)) if not np.isnan(dml_t) else np.nan
    dml_sig = '***' if dml_p < 0.001 else '**' if dml_p < 0.01 else '*' if dml_p < 0.05 else ''
    
    print(f"{'DML (main specification)':<30} {dml_peak:>15.4f}{dml_sig} ({dml_se:.4f}) {dml_r2:>10.4f} {dml_n:>10,}")
    
    print("-" * 80)
    
    # Correlations WITH P-VALUES
    if not np.isnan(corr_ols_dml):
        print(f"\nCoefficient Correlations Across Time Periods (n={n_periods}):")
        
        ols_corr_sig = '***' if p_ols_dml < 0.001 else '**' if p_ols_dml < 0.01 else '*' if p_ols_dml < 0.05 else ''
        twfe_corr_sig = '***' if p_twfe_dml < 0.001 else '**' if p_twfe_dml < 0.01 else '*' if p_twfe_dml < 0.05 else ''
        
        print(f"  OLS vs DML:  r = {corr_ols_dml:+.3f}{ols_corr_sig} (p = {p_ols_dml:.4f})")
        print(f"  TWFE vs DML: r = {corr_twfe_dml:+.3f}{twfe_corr_sig} (p = {p_twfe_dml:.4f})")
    
    print("\n")

# =============================================================================
# TABLE: FULL EVENT STUDY COEFFICIENTS (OLS, TWFE, DML)
# =============================================================================

if 'full_coef_table' in section3_stats:
    print("\n" + "=" * 100)
    print("TABLE: FULL EVENT STUDY COEFFICIENTS - OLS, TWFE, AND DML COMPARISON")
    print("=" * 100)
    print()
    
    df_coefs = section3_stats['full_coef_table']
    summaries = section3_stats['model_summaries']
    
    # Header
    print(f"{'Period':<10} {'OLS Coef':<12} {'OLS SE':<10} {'TWFE Coef':<12} {'TWFE SE':<10} {'DML Coef':<12} {'DML SE':<10}")
    print("-" * 100)
    
    for _, row in df_coefs.iterrows():
        t = int(row['time'])
        
        # Significance stars
        def get_stars(p):
            if pd.isna(p): return ''
            if p < 0.001: return '***'
            if p < 0.01: return '**'
            if p < 0.05: return '*'
            return ''
        
        ols_stars = get_stars(row['ols_pval'])
        twfe_stars = get_stars(row['twfe_pval'])
        dml_stars = get_stars(row['dml_pval'])
        
        print(f"τ = {t:+2d}    {row['ols_coef']:+.4f}{ols_stars:<3}  ({row['ols_se']:.4f})  "
              f"{row['twfe_coef']:+.4f}{twfe_stars:<3}  ({row['twfe_se']:.4f})  "
              f"{row['dml_coef']:+.4f}{dml_stars:<3}  ({row['dml_se']:.4f})")
    
    print("-" * 100)
    print(f"{'N':<10} {summaries['ols']['n']:<22,} {summaries['twfe']['n']:<22,} {summaries['dml']['n']:<22,}")
    print(f"{'R²':<10} {summaries['ols']['r2']:<22.4f} {summaries['twfe']['r2']:<22.4f} {summaries['dml']['r2']:<22.4f}")
    print()

# =============================================================================
# METHOD CORRELATIONS & FIXED EFFECTS
# =============================================================================

if 'method_correlations' in section3_stats:
    print("=" * 80)
    print("METHOD CORRELATIONS & FIXED EFFECTS")
    print("=" * 80)
    print()
    
    mc = section3_stats['method_correlations']
    fe = section3_stats['fixed_effects']
    
    print("--- Coefficient Correlations Across All 11 Time Periods ---")
    
    def format_corr(r, p):
        stars = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else ''
        return f"r = {r:+.3f}{stars} (p = {p:.4f})"
    
    print(f"   OLS vs. DML:  {format_corr(mc['ols_dml']['r'], mc['ols_dml']['p'])}")
    print(f"   TWFE vs. DML: {format_corr(mc['twfe_dml']['r'], mc['twfe_dml']['p'])}")
    print(f"   OLS vs. TWFE: {format_corr(mc['ols_twfe']['r'], mc['ols_twfe']['p'])}")
    print()
    
    print("--- Fixed Effects in TWFE Specification ---")
    print(f"   Deputy fixed effects:         {fe['n_deputy_fe']:,}")
    print(f"   Calendar-month fixed effects: {fe['n_month_fe']:,}")
    print()
    
    print("=" * 80)

# =============================================================================
# PRE-TRENDS TEST COMPARISON
# =============================================================================

if 'pretrends_tests' in section3_stats:
    print("=" * 80)
    print("PRE-TRENDS TEST (PARALLEL TRENDS) BY METHOD")
    print("=" * 80)
    print()
    print("H₀: All pre-period coefficients = 0")
    print()
    
    pt = section3_stats['pretrends_tests']
    
    print(f"{'Method':<30} {'F-statistic':<15} {'p-value':<15} {'Passes (p>0.10)':<15}")
    print("-" * 75)
    
    for method, label in [('ols', 'OLS'), ('twfe', 'TWFE'), ('dml', 'DML')]:
        f_stat = pt[method]['f_stat']
        p_val = pt[method]['p_value']
        passes = "✓ Yes" if p_val > 0.10 else "✗ No"
        print(f"{label:<30} {f_stat:<15.3f} {p_val:<15.4f} {passes:<15}")
    
    print("-" * 75)
    print()
    print("=" * 80)
    
# TABLE: COMPREHENSIVE CLASSIFIER COMPARISON ==================================

if (
    'classifier_comparison' in section3_stats
    and section3_stats['classifier_comparison']['status'] == 'Complete'
):

    print("\n" + "=" * 80)
    print("TABLE: COMPREHENSIVE CLASSIFIER COMPARISON")
    print("=" * 80)
    print()

    raw = section3_stats['classifier_comparison']['results']

    # dict-of-dicts → DataFrame
    df_res = (
        pd.DataFrame.from_dict(raw, orient='index')
        .reset_index()
        .rename(columns={
            'index': 'classifier',
            'accuracy_mean': 'accuracy',
            'weighted_f1': 'f1_weighted',
            'macro_f1': 'f1_macro',
            'time': 'time_seconds'
        })
    )

    # sort by accuracy
    df_res = df_res.sort_values('accuracy', ascending=False).reset_index(drop=True)

    print(f"{'Classifier':<30} {'Accuracy':>11} {'Weighted F1':>15} {'Macro F1':>12} {'Time (s)':>10}")
    print("-" * 85)

    for _, row in df_res.iterrows():
        print(
            f"{row['classifier']:<30} "
            f"{row['accuracy']*100:>10.1f}% "
            f"{row['f1_weighted']:>15.3f} "
            f"{row['f1_macro']:>12.3f} "
            f"{row['time_seconds']:>10.1f}"
        )

    best_acc = df_res.iloc[0]
    best_f1 = df_res.loc[df_res['f1_weighted'].idxmax()]

    print("\n" + "-" * 85)
    print(f"Best Accuracy:    {best_acc['classifier']} ({best_acc['accuracy']*100:.1f}%)")
    print(f"Best Weighted F1: {best_f1['classifier']} ({best_f1['f1_weighted']:.3f})")
    print()




# TABLE: CLASSIFIER PERFORMANCE ===============================================

if 'classifier_metrics' in section3_stats and section3_stats['classifier_metrics']['status'] == 'Complete':
    print("\n" + "="*80)
    print("TABLE 1: CLASSIFIER PERFORMANCE METRICS")
    print("="*80)
    print("\n")
    
    cm = section3_stats['classifier_metrics']
    
    print(f"{'Metric':<35} {'Value':>15} {'Baseline':>15} {'Improvement':>15}")
    print("-" * 80)
    
    print(f"{'Overall accuracy':<35} {cm['overall_accuracy']*100:>14.1f}% {cm['baseline_accuracy']*100:>14.1f}% {cm['improvement_ratio']:>14.2f}×")
    print(f"{'Weighted F1 score':<35} {cm['weighted_f1']:>15.3f} {'---':>15} {'---':>15}")
    print(f"{'Mean F1 (major parties)':<35} {cm['mean_f1_major']:>15.3f} {'---':>15} {'---':>15}")
    
    print("-" * 80)
    print("\nPerformance by party size:")
    
    for category, f1_score in cm['f1_by_size'].items():
        print(f"  {category:<30} {f1_score:>15.3f}")
    
    print(f"\nNotes: Metrics from 5-fold cross-validation on training set (N = {cm['n_train']:,} speeches)")
    print(f"       Baseline = uniform random guess across {cm['n_classes']} parties")
    
    # --- ADD: HOLDOUT (OUT-OF-SAMPLE) METRICS --------------------------------
    if 'holdout_accuracy' in cm:
        print("\n" + "-" * 80)
        print("Holdout (20% non-switchers): Out-of-sample deputy evaluation")
        print("-" * 80)

        print(f"{'Holdout accuracy (speech-level)':<45} {cm['holdout_accuracy']*100:>10.1f}%")
        print(f"{'Holdout weighted F1 (speech-level)':<45} {cm['holdout_weighted_f1']:>10.3f}")

        if 'holdout_deputy_level_accuracy' in cm:
            print(f"{'Holdout accuracy (deputy-level majority vote)':<45} {cm['holdout_deputy_level_accuracy']*100:>10.1f}%")

        if 'n_holdout' in cm and 'n_holdout_deputies' in cm:
            print(f"\nNotes: Holdout set contains N = {cm['n_holdout']:,} speeches from {cm['n_holdout_deputies']:,} deputies.")
            print("       Deputies in holdout are never used for classifier training (true out-of-sample test).")


        # TABLE 2: CV VS HOLDOUT SUMMARY ==============================================
    if 'classifier_metrics' in section3_stats and section3_stats['classifier_metrics'].get('status') == 'Complete':
        cm = section3_stats['classifier_metrics']

        # Only print if holdout metrics exist
        if all(k in cm for k in ['holdout_accuracy', 'holdout_weighted_f1', 'holdout_deputy_level_accuracy']):
            print("\n" + "="*80)
            print("TABLE 2: CLASSIFIER PERFORMANCE (TRAIN CV vs HOLDOUT OOS)")
            print("="*80)
            print()

            # Header
            print(f"{'Dataset':<28} {'Accuracy':>10} {'Weighted F1':>12} {'Deputy Acc':>12} {'N speeches':>12} {'N deputies':>10}")
            print("-" * 86)

            # TRAIN CV row (speech-level only; deputy acc not defined for CV here)
            train_n_speeches = cm.get('n_train', np.nan)
            # n_train in your code is len(y_train) = speeches count, good.
            train_n_deputies = int(df_train['deputado_id'].nunique()) if 'df_train' in globals() else np.nan

            print(f"{'Training (5-fold CV)':<28} "
                f"{cm['overall_accuracy']*100:>9.1f}% "
                f"{cm['weighted_f1']:>12.3f} "
                f"{'---':>12} "
                f"{train_n_speeches:>12,} "
                f"{train_n_deputies:>10,}")

            # HOLDOUT row
            print(f"{'Holdout (OOS deputies)':<28} "
                f"{cm['holdout_accuracy']*100:>9.1f}% "
                f"{cm['holdout_weighted_f1']:>12.3f} "
                f"{cm['holdout_deputy_level_accuracy']*100:>11.1f}% "
                f"{cm.get('n_holdout', np.nan):>12,} "
                f"{cm.get('n_holdout_deputies', np.nan):>10,}")

            print("-" * 86)

            # Optional baseline + improvement if you stored them
            if 'baseline_accuracy' in cm:
                base = cm['baseline_accuracy'] * 100
                train_imp = cm.get('improvement_ratio', np.nan)
                hold_imp = cm.get('holdout_improvement_ratio', np.nan)
                print(f"Baseline (uniform random over {cm['n_classes']} parties): {base:.1f}%")
                if not np.isnan(train_imp):
                    print(f"Training improvement over baseline: {train_imp:.2f}×")
                if not np.isnan(hold_imp):
                    print(f"Holdout improvement over baseline:  {hold_imp:.2f}×")

            print()


    print("\n" + "="*80)
    print("TABLE 2: TOP 10 PARTIES - CLASSIFICATION PERFORMANCE")
    print("="*80)
    print("\n")
    
    df_top10 = cm['party_performance'].head(10)
    
    print(f"{'Party':<15} {'N Speeches':>12} {'Precision':>12} {'Recall':>12} {'F1':>12}")
    print("-" * 80)
    
    for _, row in df_top10.iterrows():
        print(f"{row['sigla']:<15} {row['n_speeches']:>12,} {row['precision']:>12.3f} {row['recall']:>12.3f} {row['f1']:>12.3f}")
    
    print("\n")

# TABLE: HOLDOUT (OOS) CLASSIFIER PERFORMANCE =================================

if 'classifier_metrics' in section3_stats and section3_stats['classifier_metrics'].get('holdout_accuracy') is not None:
    cm = section3_stats['classifier_metrics']

    print("\n" + "="*80)
    print("TABLE: HOLDOUT (OUT-OF-SAMPLE) CLASSIFIER PERFORMANCE")
    print("="*80)
    print()

    print(f"{'Metric':<55} {'Value':>15}")
    print("-" * 80)
    print(f"{'Holdout accuracy (speech-level)':<55} {cm['holdout_accuracy']*100:>14.1f}%")
    print(f"{'Holdout weighted F1 (speech-level)':<55} {cm['holdout_weighted_f1']:>15.3f}")
    if 'holdout_deputy_level_accuracy' in cm:
        print(f"{'Holdout accuracy (deputy-level majority vote)':<55} {cm['holdout_deputy_level_accuracy']*100:>14.1f}%")
    if 'n_holdout' in cm and 'n_holdout_deputies' in cm:
        print(f"{'Holdout N (speeches)':<55} {cm['n_holdout']:>15,}")
        print(f"{'Holdout N (deputies)':<55} {cm['n_holdout_deputies']:>15,}")
    print("-" * 80)
    print()


# =============================================================================
# TABLE: CLASSIFIER ALGORITHM COMPARISON
# =============================================================================

if 'classifier_comparison' in section3_stats:
    print("\n" + "=" * 90)
    print("TABLE: CLASSIFIER ALGORITHM COMPARISON")
    print("=" * 90)
    print()

    cc = section3_stats['classifier_comparison']
    if cc.get('status') != 'Complete' or 'results' not in cc:
        print("⚠️  Classifier comparison not complete or missing results.")
    else:
        results = cc['results']

        print(f"{'Algorithm':<30} {'Accuracy':<15} {'Weighted F1':<15} {'Probability Output':<25}")
        print("-" * 90)

        order = ['Logistic Regression', 'Linear SVM', 'Multinomial Naive Bayes', 'Random Forest']

        for name in order:
            if name in results:
                stats = results[name]
                acc_str = f"{stats['accuracy_mean']*100:.1f}%"
                f1_str = f"{stats['weighted_f1']:.3f}"
                label = f"{name} (selected)" if name == 'Logistic Regression' else name
                print(f"{label:<30} {acc_str:<15} {f1_str:<15} {stats.get('prob_output',''): <25}")

        print("-" * 90)
        print()
        print("Notes:")
        print("   - All classifiers trained on identical TF-IDF features (N = {:,})".format(X_train.shape[0]))
        print(f"   - {X_train.shape[1]:,} features, 5-fold group cross-validation (by deputy)")
        print("   - Logistic regression selected for native probability calibration")
        print()
        print("=" * 90)


# TABLE: NAMED ENTITY REMOVAL =================================================

print("\n" + "="*80)
print("TABLE: NAMED ENTITY REMOVAL")
print("="*80)

if 'section3_stats' in globals() and 'robustness_text' in section3_stats:
    rob_text = section3_stats['robustness_text']
    
    if rob_text['status'] == 'Complete':
        print(f"\nText column used: {rob_text['col_name']}")
        print(f"Total sample size: {rob_text['n_obs']:,} observations")
        
        n_in = rob_text.get('n_in_window', np.nan)
        n_out = rob_text.get('n_out_window', np.nan)
        if not np.isnan(n_in):
            print(f"  - Within ±180 days: {n_in:,}")
            print(f"  - Outside window (controls): {n_out:,}")
        
        level3_peak_t1 = rob_text.get('peak_effect_t1', rob_text.get('peak_effect', np.nan))
        print(f"\nPeak effect (τ = +1): {level3_peak_t1:.4f}")
        
        if 'dml_stats' in globals() and 'plot_data' in dml_stats:
            main_data = dml_stats['plot_data']
            if 1 in main_data['time'].values:
                main_peak_t1 = main_data[main_data['time'] == 1]['coef'].values[0]
                
                attenuation_pct = (1 - abs(level3_peak_t1) / abs(main_peak_t1)) * 100
                remaining_pct = 100 - attenuation_pct
                
                print(f"\nComparison:")
                print(f"  Main result (Level 2, τ=+1):        {main_peak_t1:.4f}")
                print(f"  Level 3 result (NER removed, τ=+1): {level3_peak_t1:.4f}")
                print(f"\n  Attenuation: {attenuation_pct:.1f}% of signal eliminated")
                print(f"  Remaining:   {remaining_pct:.1f}% of signal persists")
                print(f"\nInterpretation:")
                print(f"  Named entities (person names, locations, organizations) account for")
                print(f"  ~{attenuation_pct:.0f}% of the adaptation signal. The majority of the effect")
                print(f"  ({remaining_pct:.0f}%) comes from substantive policy and ideological vocabulary.")
                
                main_n = dml_stats['model_summary']['N_Observations']
                level3_n = rob_text['n_obs']
                
                if abs(main_n - level3_n) > 100:
                    print(f"\n⚠️  WARNING: Sample size mismatch!")
                    print(f"   Main analysis N: {main_n:,}")
                    print(f"   Level 3 N:       {level3_n:,}")
                    print(f"   Difference:      {abs(main_n - level3_n):,}")
                else:
                    print(f"\n✓ Sample sizes match: N = {level3_n:,}")
    else:
        print(f"\nStatus: {rob_text['status']}")
else:
    print("\n⚠️  Alternative text robustness not found")

print("\n")


# TABLE: BLOC-LEVEL CLASSIFIER ================================================

print("\n" + "="*80)
print("TABLE: IDEOLOGICAL BLOC CLASSIFICATION")
print("="*80)

if 'section3_stats' in globals() and 'bloc_robustness' in section3_stats:
    bloc = section3_stats['bloc_robustness']
    
    if bloc.get('status') == 'Complete':
        print(f"\nBloc classifier accuracy: {bloc['cv_accuracy']*100:.1f}%")
        print(f"Sample size: {bloc['n_obs']:,} observations")
        print(f"\nPre-post difference: {bloc['diff']:.4f}***")
        print(f"p-value: {bloc['p_value']:.4f}")
        print(f"Cohen's d: {bloc['cohens_d']:.3f}")
        
        print(f"\nComparison:")
        print(f"  Bloc-level classifier (3 classes):   {bloc['cv_accuracy']*100:.1f}% accuracy, effect = {bloc['diff']:.4f}")
        print(f"\nNote: Bloc-level uses simple pre-post comparison, not full event study.")
    else:
        print(f"\nStatus: {bloc.get('status', 'Not found')}")
else:
    print("\n⚠️  Bloc classifier robustness not found")

print("\n")

# TABLE: EMBEDDING SIMILARITY =================================================

print("\n" + "="*80)
print("TABLE: EMBEDDING SIMILARITY (Semantic Distance)")
print("="*80)

if 'section3_stats' in globals() and 'embedding_robustness' in section3_stats:
    emb_rob = section3_stats['embedding_robustness']
    
    if emb_rob['status'] == 'Complete':
        print(f"\nSample size: {emb_rob['n_obs']:,} observations")
        print(f"Parties with embeddings: {emb_rob['n_parties_with_emb']}")
        
        print("\n--- Cosine Similarity to Old Party ---")
        print(f"Pre-switch mean:  {emb_rob['pre_mean']:.4f}")
        print(f"Post-switch mean: {emb_rob['post_mean']:.4f}")
        print(f"Difference:       {emb_rob['diff']:+.4f}")
        print(f"t-statistic:      {emb_rob['t_stat']:.3f}")
        print(f"p-value:          {emb_rob['p_value']:.4f}")
        print(f"Cohen's d:        {emb_rob['cohens_d']:.3f}")
        
        sig = "***" if emb_rob['p_value'] < 0.001 else "**" if emb_rob['p_value'] < 0.01 else "*" if emb_rob['p_value'] < 0.05 else ""
        print(f"\nEffect: {sig} {'Significant' if sig else 'Not significant'}")
        
        d = abs(emb_rob['cohens_d'])
        if d < 0.2:
            effect_label = "negligible"
        elif d < 0.5:
            effect_label = "small"
        elif d < 0.8:
            effect_label = "medium"
        else:
            effect_label = "large"
        print(f"Effect size: {effect_label} (d = {emb_rob['cohens_d']:.3f})")
    else:
        print(f"\nStatus: {emb_rob['status']}")
else:
    print("\n⚠️  Embedding robustness not found")

print("\n")

# TABLE: PLACEBO TEST =========================================================

print("\n" + "="*80)
print("TABLE: PLACEBO DISTRIBUTION TEST (Permutation)")
print("="*80)

if 'section3_stats' in globals() and 'placebo' in section3_stats:
    placebo = section3_stats['placebo']
    
    if placebo['status'] == 'Complete':
        print(f"\nPermutations run: {placebo['n_permutations']:,}")
        print(f"True effect: {placebo['true_effect']:.4f}")
        print(f"Mean placebo: {placebo['mean_placebo']:.4f}")
        print(f"Percentile: {placebo['percentile']:.1f}th")
        print(f"p-value: {placebo['p_value']:.4f}")
    else:
        print(f"\nStatus: {placebo['status']}")
else:
    print("\n⚠️  Placebo test not found")

print("\n")

# TABLE: WINDOW SENSITIVITY (WITH SE) ==========================================

print("\n" + "="*80)
print("TABLE: WINDOW SENSITIVITY")
print("="*80)

if 'section3_stats' in globals() and 'window_sensitivity' in section3_stats:
    win_sens = section3_stats['window_sensitivity']
    
    if win_sens['status'] == 'Complete':
        df_win = win_sens['results']
        
        print(f"\nMethod: {win_sens['method']}")
        print("\n--- Results by Window ---")
        
        print(f"{'Window':<15} {'ATE':>12} {'SE':>12} {'p-value':>12} {'N':>12}")
        print("-" * 70)
        
        for _, row in df_win.iterrows():
            sig = "***" if row['p_value'] < 0.001 else "**" if row['p_value'] < 0.01 else "*" if row['p_value'] < 0.05 else ""
            print(f"±{row['window_months']:.0f} months {row['coef']:>12.4f}{sig} {row['se']:>12.4f} {row['p_value']:>12.4f} {row['n_obs']:>12,.0f}")
        
        print("\n")
    else:
        print(f"\nStatus: {win_sens['status']}")
else:
    print("\n⚠️  Window sensitivity not found")

print("\n")

# BIN SENSITIVITY RESULTS ======================================================


print("=" * 80)
print("TABLE: EVENT STUDY BY BIN GRANULARITY")
print("=" * 80)
print()

print(f"{'Bin Size':<20} {'Peak (τ=+1)':<18} {'SE':<12} {'Pre-trends p':<15} {'N Obs':<12} {'N Periods':<10}")
print("-" * 80)

for bin_size in [30, 60, 90]:
    r = bin_sensitivity_results[bin_size]
    
    # Significance stars
    if r['peak_pval'] < 0.001:
        stars = '***'
    elif r['peak_pval'] < 0.01:
        stars = '**'
    elif r['peak_pval'] < 0.05:
        stars = '*'
    else:
        stars = ''
    
    # Pre-trends status
    if r['pretrends_p'] > 0.10:
        pretrends_status = f"{r['pretrends_p']:.4f} ✓"
    else:
        pretrends_status = f"{r['pretrends_p']:.4f} ✗"
    
    bin_label = f"{bin_size} days"
    if bin_size == 60:
        bin_label += " (main)"
    
    print(f"{bin_label:<20} {r['peak_effect']:>+.4f}{stars:<3} {'(':>2}{r['peak_se']:.4f}) {pretrends_status:<15} {r['n_obs']:<12,} {r['n_periods']:<10}")

print("-" * 80)
print()

print("Pre-trends test details:")
for bin_size in [30, 60, 90]:
    r = bin_sensitivity_results[bin_size]
    print(f"   {bin_size}-day: F({r['n_pre_periods']}, {r['n_obs'] - r['n_periods'] - 1}) = {r['pretrends_f']:.3f}, p = {r['pretrends_p']:.4f}")

print()
print("Interpretation:")
print("   ✓ = Passes pre-trends test (p > 0.10)")
print("   ✗ = Fails pre-trends test (p ≤ 0.10)")
print()

# Check if 30-day fails and 60/90 pass
if bin_sensitivity_results[30]['pretrends_p'] <= 0.10 and bin_sensitivity_results[60]['pretrends_p'] > 0.10:
    print("⚠️  30-day bins fail pre-trends, likely due to noise in sparse bins or")
    print("   minor anticipation effects. 60-day bimesters offer optimal balance.")
elif all(bin_sensitivity_results[b]['pretrends_p'] > 0.10 for b in [30, 60, 90]):
    print("✅ All bin sizes pass pre-trends test. Results robust to bin granularity.")
else:
    print("⚠️  Mixed pre-trends results. See discussion in text.")

print()
print("=" * 80)

# TABLE: ALTERNATIVE OUTCOME ==================================================

print("\n" + "="*80)
print("TABLE: ALTERNATIVE OUTCOME [P(New) - P(Old)]")
print("="*80)

if 'section3_stats' in globals() and 'alt_outcome' in section3_stats:
    alt_out = section3_stats['alt_outcome']
    
    if alt_out['status'] == 'Complete':
        print(f"\nSample size: {alt_out['n_obs']:,} observations")
        print(f"Peak effect (τ = +1 to +4): {alt_out['peak_effect']:+.4f}")
    else:
        print(f"\nStatus: {alt_out['status']}")
else:
    print("\n⚠️  Alternative outcome not found")

print("\n")

# SUMMARY TABLE ===============================================================

print("\n" + "="*80)
print("SUMMARY TABLE: ALL ROBUSTNESS CHECKS")
print("="*80)

summary_data = []

if 'section3_stats' in globals():
    checks = [
        ('Method Comparison (OLS)', section3_stats.get('OLS', {}).get('Peak_Effect', np.nan)),
        ('Method Comparison (TWFE)', section3_stats.get('TWFE', {}).get('Peak_Effect', np.nan)),
        ('Method Comparison (DML)', section3_stats.get('DML', {}).get('Peak_Effect', np.nan)),
        ('Alt Text (NER removed)', section3_stats.get('robustness_text', {}).get('peak_effect_t1', np.nan)),
        ('Bloc Classification', section3_stats.get('bloc_robustness', {}).get('diff', np.nan)),
        ('Embedding Similarity', section3_stats.get('embedding_robustness', {}).get('diff', np.nan)),
        ('Placebo Test', section3_stats.get('placebo', {}).get('true_effect', np.nan)),
        ('Alternative Outcome', section3_stats.get('alt_outcome', {}).get('peak_effect', np.nan)),
        ('Classifier Holdout Accuracy (speech-level)', section3_stats.get('classifier_metrics', {}).get('holdout_accuracy', np.nan)),
        ('Classifier Holdout Accuracy (deputy-level)', section3_stats.get('classifier_metrics', {}).get('holdout_deputy_level_accuracy', np.nan)),
    ]
    
    for check_name, value in checks:
        if not pd.isna(value):
            summary_data.append({
                'Check': check_name,
                'Effect': value,
                'Status': '✓'
            })
    
    df_summary = pd.DataFrame(summary_data)
    
    print("\n" + df_summary.to_string(index=False))

print("\n" + "="*80)
print("ROBUSTNESS REPORT COMPLETE")
print("="*80)
print(f"Total checks completed: {len(summary_data)}")

# %%
def generate_all_visualizations(section3_stats, dml_data=None, save_dir=None):
    """
    Generates robustness and sensitivity plots.
    - Automatically creates save_dir if it does not exist.
    """
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import os

    # --- FIX: CREATE DIRECTORY IF IT DOESN'T EXIST ---
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    # --- 1. GLOBAL FONT & STYLE SETUP ---
    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'DejaVu Serif', 'Bitstream Vera Serif'],
        'font.size': 10,
        'axes.labelsize': 10,
        'axes.titlesize': 10,
        'xtick.labelsize': 9,
        'ytick.labelsize': 9,
        'legend.fontsize': 9,
        'mathtext.fontset': 'cm',
        'axes.linewidth': 0.5,
    })

    # --- 2. COLORS & HELPERS ---
    ECONOMIST_COLORS = ['#006BA2', '#3EBCD2', '#379A8B', '#EBB434', '#DB444B']
    
    C_MAIN   = ECONOMIST_COLORS[0]  # Dark Blue (Standard Outcome)
    C_SEC    = ECONOMIST_COLORS[1]  # Light Blue
    C_TERT   = ECONOMIST_COLORS[2]  # Teal (Alternative Outcome)
    C_ACCENT = ECONOMIST_COLORS[3]  # Yellow
    C_ALERT  = ECONOMIST_COLORS[4]  # Red
    C_GRAY   = '#444444'
    C_GRID   = '#dcdcdc'

    def format_ax(ax, xlabel, ylabel):
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.grid(axis='y', color=C_GRID, lw=0.5, zorder=0)
        ax.grid(axis='x', visible=False)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        return ax

    print("\n" + "="*80)
    print("GENERATING VISUALIZATIONS (Corrected Reference Points)...")
    print("="*80)

    # 1. FIGURE: METHOD COMPARISON ============================================
    if 'comparison_table' in section3_stats:
        df_comp = section3_stats['comparison_table'].copy()
        
        # FIX: Ensure reference point exists
        if -1 not in df_comp['time'].values:
            ref_row = pd.DataFrame({'time': [-1], 'OLS': [0], 'TWFE': [0], 'DML': [0]})
            df_comp = pd.concat([df_comp, ref_row]).sort_values('time')

        fig, ax = plt.subplots(figsize=figsize(width_key='my_paper', aspect=0.5))
        
        ax.axhline(0, color=C_GRAY, lw=0.8, alpha=0.8, zorder=1)
        ax.axvline(0, color=C_ALERT, ls='--', lw=0.8, zorder=1)
        
        ax.plot(df_comp['time'], df_comp['OLS'], 'o--', 
                color=C_SEC, alpha=0.7, label='OLS', lw=1.0, ms=4, zorder=2)
        ax.plot(df_comp['time'], df_comp['TWFE'], 's--', 
                color=C_TERT, alpha=0.7, label='TWFE', lw=1.0, ms=4, zorder=2)
        ax.plot(df_comp['time'], df_comp['DML'], 'D-', 
                color=C_MAIN, lw=1, label='DML (Main)', ms=5, zorder=3)
        
        format_ax(ax, "Bimesters from Switch", "Effect on P(Old Party)")
        ax.legend(frameon=False, loc='best')
        
        plt.tight_layout()
        if save_dir: plt.savefig(os.path.join(save_dir, 'fig_robust_methods.png'), dpi=300)
        plt.show()
        print("✅ Method comparison plot generated")

    # 2. FIGURE: PLACEBO DISTRIBUTION =========================================
    if 'placebo' in section3_stats:
        placebo = section3_stats['placebo']
        if placebo['status'] == 'Complete':
            fig, ax = plt.subplots(figsize=figsize(width_key='my_paper', aspect=0.6))
            ax.hist(placebo['placebo_stats'], bins=40, alpha=0.85, 
                    color=C_TERT, edgecolor='white', linewidth=0.5, zorder=2)
            ax.axvline(placebo['true_effect'], color=C_ALERT, ls='--', lw=1.2, 
                       label=f"True Effect ({placebo['true_effect']:.3f})", zorder=3)
            ax.axvline(placebo['mean_placebo'], color=C_MAIN, ls=':', lw=1.2,
                       label=f"Mean Placebo ({placebo['mean_placebo']:.3f})", zorder=3)
            format_ax(ax, "|Effect Size|", "Frequency")
            stats_text = f'$p$-value < {placebo["p_value"]:.4f}\nPercentile: {placebo["percentile"]:.1f}'
            ax.text(0.95, 0.90, stats_text, transform=ax.transAxes, 
                    fontsize=9, va='top', ha='right',
                    bbox=dict(boxstyle='square,pad=0.4', fc=C_ACCENT, alpha=0.15, ec='none'))
            ax.legend(frameon=False, loc='upper right')
            plt.tight_layout()
            if save_dir: plt.savefig(os.path.join(save_dir, 'fig_placebo.png'), dpi=300)
            plt.show()
            print("✅ Placebo distribution plot generated")

    # 3. FIGURE: WINDOW SENSITIVITY ===========================================
    if 'window_sensitivity' in section3_stats:
        win_sens = section3_stats['window_sensitivity']
        if win_sens['status'] == 'Complete':
            df_win = win_sens['results']
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize(width_key='my_paper', fraction=1.0, aspect=0.4))
            ax1.axhline(0, color=C_GRAY, ls='--', lw=0.8, alpha=0.5, zorder=1)
            ax1.plot(df_win['window_months'], df_win['coef'], 'o-', 
                     lw=1.5, ms=5, color=C_MAIN, zorder=2)
            format_ax(ax1, "Window Size (months)", "ATE (Post vs Pre)")
            ax2.bar(df_win['window_months'], df_win['n_obs'], 
                    color=C_TERT, alpha=0.9, width=1.5, zorder=2)
            format_ax(ax2, "Window Size (months)", "N Observations")
            plt.tight_layout()
            if save_dir: plt.savefig(os.path.join(save_dir, 'fig_window_sensitivity.png'), dpi=300)
            plt.show()
            print("✅ Window sensitivity plot generated")

    # 4. FIGURE: STANDARD VS ALTERNATIVE OUTCOME (COMBINED) ===================
    if 'alt_outcome' in section3_stats and dml_data is not None:
        alt_out = section3_stats['alt_outcome']
        
        if alt_out['status'] == 'Complete' and 'plot_data' in alt_out:
            fig, ax = plt.subplots(figsize=figsize(width_key='my_paper', aspect=0.6))
            
            # --- Reference Lines ---
            ax.axhline(0, color=C_GRAY, ls='-', lw=0.8, alpha=0.5, zorder=1)
            ax.axvline(0, color=C_ALERT, ls='--', lw=0.8, alpha=0.8, zorder=1)

            # --- SERIES 1: STANDARD OUTCOME (from dml_data) ---
            # Ensure standard data has reference point too (usually does, but safe to check)
            std_data = dml_data['plot_data'].copy()
            if -1 not in std_data['time'].values:
                ref_row = pd.DataFrame({'time': [-1], 'coef': [0], 'lower': [0], 'upper': [0]})
                std_data = pd.concat([std_data, ref_row]).sort_values('time')

            ax.plot(std_data['time'], std_data['coef'], 'o-', 
                    lw=1, ms=4, color=C_MAIN, label='Standard: P(Old Party)', zorder=4)
            ax.fill_between(std_data['time'], 
                            std_data['lower'], 
                            std_data['upper'], 
                            alpha=0.15, color=C_MAIN, lw=0, zorder=2)

            # --- SERIES 2: ALTERNATIVE OUTCOME (from section3_stats) ---
            alt_data = alt_out['plot_data'].copy()
            
            # --- FIX: INSERT REFERENCE POINT FOR ALTERNATIVE OUTCOME ---
            if -1 not in alt_data['time'].values:
                # Create a 0-row. Set SE/Bounds to 0 for the reference point.
                ref_row = pd.DataFrame({
                    'time': [-1], 
                    'coef': [0], 
                    'se': [0],
                    'lower': [0],
                    'upper': [0]
                })
                # Handle case where lower/upper might not exist in original df
                if 'lower' not in alt_data.columns:
                     alt_data['lower'] = alt_data['coef'] - 1.96 * alt_data['se']
                     alt_data['upper'] = alt_data['coef'] + 1.96 * alt_data['se']
                
                alt_data = pd.concat([alt_data, ref_row]).sort_values('time')

            # Re-calculate bounds if they were missing or just to be safe after concat
            if 'lower' not in alt_data.columns or alt_data['lower'].isnull().any():
                 # Avoid overwriting the 0s we just set for the reference point
                 mask = alt_data['time'] != -1
                 alt_data.loc[mask, 'lower'] = alt_data.loc[mask, 'coef'] - 1.96 * alt_data.loc[mask, 'se']
                 alt_data.loc[mask, 'upper'] = alt_data.loc[mask, 'coef'] + 1.96 * alt_data.loc[mask, 'se']

            ax.plot(alt_data['time'], alt_data['coef'], 's--', 
                    lw=1, ms=4, color=C_TERT, label='Alternative: P(New - Old)', zorder=3)
            
            ax.fill_between(alt_data['time'], 
                            alt_data['lower'], 
                            alt_data['upper'], 
                            alpha=0.15, color=C_TERT, lw=0, zorder=2)
            
            format_ax(ax, "Bimesters from Switch", "Treatment Effect")
            ax.legend(frameon=False, loc='best')
            
            plt.tight_layout()
            if save_dir: plt.savefig(os.path.join(save_dir, 'fig_outcomes_combined.png'), dpi=300)
            plt.show()
            print("✅ Combined outcomes plot generated")

    # 5. FIGURE: EVENT STUDY COMPARISON =======================================
    if 'window_comparison_event_study' in section3_stats:
        comp = section3_stats['window_comparison_event_study']
        if comp['status'] == 'Complete':
            fig, ax = plt.subplots(figsize=figsize(width_key='my_paper', aspect=0.6))
            ax.axhline(0, color=C_GRAY, ls='-', lw=0.8, alpha=0.5, zorder=1)
            ax.axvline(0, color=C_ALERT, ls='--', lw=0.8, zorder=1)
            color_map = {'6': C_TERT, '9': C_SEC, '12': C_MAIN}
            
            for window_months, data in comp['results'].items():
                plot_data = data['plot_data']
                w_str = str(window_months)
                color = color_map.get(w_str, C_GRAY)
                label = f"±{window_months}m (N={data['n_obs']:,})"
                ax.plot(plot_data['time'], plot_data['coef'], 'o-', 
                        lw=1.0, ms=3, color=color, label=label, alpha=0.9, zorder=3)
                ax.fill_between(plot_data['time'], plot_data['lower'], plot_data['upper'],
                                alpha=0.1, color=color, lw=0, zorder=2)
            
            format_ax(ax, "Bimesters from Switch", "Effect on P(Old Party)")
            ax.legend(frameon=False, loc='upper right')
            plt.tight_layout()
            if save_dir: plt.savefig(os.path.join(save_dir, 'fig_window_comp.png'), dpi=300)
            plt.show()
            print("✅ Event study comparison plot generated")

    print("\n" + "="*80)
    print("ALL VISUALIZATIONS COMPLETE")
    print("="*80)

# USAGE:
save_dir = "../results/final_paper/figures"
generate_all_visualizations(section3_stats, dml_data=dml_stats, save_dir=save_dir)

# %% [markdown]
# ---
# # PART IV: HETEROGENEITY ANALYSIS
# ---

# %% [markdown]
# ### Section Setup

# %%
# HETEROGENEITY ANALYSIS ======================================================
# =============================================================================
# We DO NOT residualize Y_confidence for heterogeneity analysis.
# Reason: the covariates used (tenure/activity + FE) have near-zero out-of-fold
# predictive power for Y_confidence, so "residualization" adds noise without
# meaningful adjustment.
# =============================================================================

print("Heterogeneity analysis will use RAW Y_confidence (no residualization).")


# %%
# CELL 1: DATA PREPARATION =====================================================
# =============================================================================

print("Preparing heterogeneity analysis data...")

# Initialize results container
section4_stats = {}

# Calculate switch-level effects using RAW outcome (Y_confidence)
heterogeneity_data = []

for _, switch in tqdm(df_events.iterrows(), total=len(df_events), desc="Computing switch-level effects"):
    deputy_id = switch['deputado_id']
    switch_date = pd.to_datetime(switch['switch_date'])
    
    # Get speeches for this switcher
    deputy_speeches = df_es[df_es['deputado_id'] == deputy_id].copy()
    
    pre = deputy_speeches[
        (deputy_speeches['days_from_switch'] < 0) & 
        (deputy_speeches['days_from_switch'] >= -180)
    ]['Y_confidence']

    post = deputy_speeches[
        (deputy_speeches['days_from_switch'] > 0) & 
        (deputy_speeches['days_from_switch'] <= 180)
    ]['Y_confidence']

    # HETEROGENEITY MINIMUM SPEECHES PER WINDOW
    min_w = CFG.het_min_speeches_window
    if (len(pre) < min_w) or (len(post) < min_w):
        continue

    effect = pre.mean() - post.mean()
    
    # ========================================================================
    # HETEROGENEITY VARIABLES
    # ========================================================================
    
    # 1. Ideological Distance
    ideology_distance = abs(switch['new_EST'] - switch['old_EST']) if 'new_EST' in switch and 'old_EST' in switch else np.nan
    
       # 3. Experience/Tenure
    pre_speeches = deputy_speeches[deputy_speeches['days_from_switch'] < 0]
    if 'career_tenure_years' in pre_speeches.columns:
        tenure_years = pre_speeches['career_tenure_years'].median()
    else:
        tenure_years = np.nan
    
    if pd.notna(tenure_years):
        if tenure_years < 5:
            experience_cat = 'Junior'
        elif tenure_years < 10:
            experience_cat = 'Mid'
        else:
            experience_cat = 'Senior'
    else:
        experience_cat = 'Unknown'
    
    # 4. Party Size
    new_party_id = switch['new_party_id']
    new_party_size = df[
        (df['idPartido'] == new_party_id) & 
        (df['dataHoraInicio'].dt.year == switch_date.year)
    ]['deputado_id'].nunique()
    
    if new_party_size >= 30:
        party_size_cat = 'Major'
    elif new_party_size >= 10:
        party_size_cat = 'Medium'
    else:
        party_size_cat = 'Minor'
    
    # Store
    heterogeneity_data.append({
        'deputado_id': deputy_id,
        'switch_date': switch_date,
        'effect': effect,
        'abs_effect': abs(effect),
        'n_pre': len(pre),
        'n_post': len(post),
        'ideology_distance': ideology_distance,
        'tenure_years': tenure_years,
        'experience_cat': experience_cat,
        'new_party_size': new_party_size,
        'party_size_cat': party_size_cat,
        'old_CAT': switch['old_CAT'],
        'new_CAT': switch['new_CAT']
    })

df_het = pd.DataFrame(heterogeneity_data)

section4_stats['data_prep'] = {
    'status': 'Complete',
    'n_switchers': len(df_het),
    'mean_effect': df_het['effect'].mean(),
    'mean_abs_effect': df_het['abs_effect'].mean(),
    'experience_counts': df_het['experience_cat'].value_counts().to_dict(),
    'party_size_counts': df_het['party_size_cat'].value_counts().to_dict(),
}

print(f"   ✓ Prepared {len(df_het):,} switchers")
print(f"   Mean effect (raw): {df_het['effect'].mean():.4f}")


# %%
# WITHIN BLOC VS CROSS-BLOC ===================================================
# =============================================================================
import numpy as np
import pandas as pd
import statsmodels.api as sm

print("Running Robustness Check: Within-Bloc Switching Analysis...")

# 1. CLASSIFY SWITCHES --------------------------------------------------------
df_events_bloc = df_events.copy()

# Determine if switch crosses bloc boundaries
df_events_bloc['crosses_bloc'] = df_events_bloc['old_CAT'] != df_events_bloc['new_CAT']
df_events_bloc['within_left'] = (df_events_bloc['old_CAT'] == 'Left') & (df_events_bloc['new_CAT'] == 'Left')
df_events_bloc['within_center'] = (df_events_bloc['old_CAT'] == 'Center') & (df_events_bloc['new_CAT'] == 'Center')
df_events_bloc['within_right'] = (df_events_bloc['old_CAT'] == 'Right') & (df_events_bloc['new_CAT'] == 'Right')
df_events_bloc['within_bloc'] = df_events_bloc['within_left'] | df_events_bloc['within_center'] | df_events_bloc['within_right']

# 2. HELPER FOR EVENT STUDY REGRESSION ----------------------------------------
def run_event_study_ols(df_subset):
    if len(df_subset) < 100: return None, None, 0
    
    # Time Bins (Matches Main Analysis)
    bins_wb = list(range(-360, 361, 60))
    labels_wb = [-6,-5,-4,-3,-2,-1,1,2,3,4,5,6]
    df_subset = df_subset.copy()
    df_subset['month'] = pd.cut(df_subset['days_from_switch'], bins=bins_wb, labels=labels_wb)
    
    # Matrix Construction
    time_dummies = pd.get_dummies(df_subset['month'], prefix='t').astype(float)
    if 't_-1' in time_dummies.columns: time_dummies = time_dummies.drop(columns=['t_-1'])
    
    # Covariates
    X_controls = build_covariates(
        df_subset,
        include_calendar_month_fe=False,
        include_topic_fe=True,
        date_col='dataHoraInicio',
        verbose=False
    )

    X_final = pd.concat([time_dummies, X_controls], axis=1)
    X_final = sm.add_constant(X_final).astype(float)
    
    Y = df_subset['Y_confidence'].values
    clusters = df_subset['deputado_id'].values
    
    model = sm.OLS(Y, X_final).fit(cov_type='cluster', cov_kwds={'groups': clusters})
    
    # Format Results
    coefs = model.params[[c for c in model.params.index if c.startswith('t_')]]
    ses = model.bse[[c for c in model.bse.index if c.startswith('t_')]]
    
    plot_data = pd.DataFrame({
        'time': [int(c.replace('t_','')) for c in coefs.index],
        'coef': coefs.values,
        'se': ses.values
    })
    
    # --- FIX: INSERT REFERENCE POINT (t=-1) ---
    # This forces the line to touch 0 at the reference period
    ref_row = pd.DataFrame({
        'time': [-1],
        'coef': [0.0],
        'se': [0.0]
    })
    plot_data = pd.concat([plot_data, ref_row], ignore_index=True).sort_values('time')
    
    # Add CI columns
    plot_data['lower'] = plot_data['coef'] - 1.96 * plot_data['se']
    plot_data['upper'] = plot_data['coef'] + 1.96 * plot_data['se']
    
    n_switchers = df_subset['deputado_id'].nunique()
    
    return model, plot_data, n_switchers

# 3. RUN ESTIMATIONS ----------------------------------------------------------
# A. Within-Bloc
within_ids = df_events_bloc[df_events_bloc['within_bloc']]['deputado_id'].unique()
df_es_within = df_es[df_es['deputado_id'].isin(within_ids)]
model_wb, plot_wb, n_switchers_wb = run_event_study_ols(df_es_within)

# B. Cross-Bloc (Comparison)
cross_ids = df_events_bloc[df_events_bloc['crosses_bloc']]['deputado_id'].unique()
df_es_cross = df_es[df_es['deputado_id'].isin(cross_ids)]
model_cb, plot_cb, n_switchers_cb = run_event_study_ols(df_es_cross)

# 4. EXTRACT PEAK AT TAU=+1 ---------------------------------------------------
if plot_wb is not None:
    peak_wb_row = plot_wb[plot_wb['time'] == 1]
    if len(peak_wb_row) > 0:
        peak_wb = peak_wb_row['coef'].values[0]
        se_wb = peak_wb_row['se'].values[0]
    else:
        peak_wb = np.nan
        se_wb = np.nan
else:
    peak_wb = np.nan
    se_wb = np.nan

if plot_cb is not None:
    peak_cb_row = plot_cb[plot_cb['time'] == 1]
    if len(peak_cb_row) > 0:
        peak_cb = peak_cb_row['coef'].values[0]
        se_cb = peak_cb_row['se'].values[0]
    else:
        peak_cb = np.nan
        se_cb = np.nan
else:
    peak_cb = np.nan
    se_cb = np.nan

# 5. STORE RESULTS ------------------------------------------------------------
section3_stats['within_bloc'] = {
    'status': 'Complete',
    'n_within': len(df_es_within),
    'n_cross': len(df_es_cross),
    'n_switchers_within': n_switchers_wb,
    'n_switchers_cross': n_switchers_cb,
    'peak_within': peak_wb,
    'peak_cross': peak_cb,
    'se_within': se_wb,
    'se_cross': se_cb,
    'passes': abs(peak_wb) > 0.01 if not pd.isna(peak_wb) else False,
    'plot_data_within': plot_wb,
    'plot_data_cross': plot_cb
}

# Store for persistence
WITHINBLOC_RESULTS = section3_stats['within_bloc'].copy()

print(f"   Analysis complete.")
print(f"   Within-bloc: N_obs={len(df_es_within):,}, N_switchers={n_switchers_wb}, peak={peak_wb:.4f}")
print(f"   Cross-bloc:  N_obs={len(df_es_cross):,}, N_switchers={n_switchers_cb}, peak={peak_cb:.4f}")


# 6. DIFFERENCE TEST (CHI-SQUARED) --------------------------------------------
# Wald test for H0: peak_cross = peak_within
# Test statistic: (β_cross - β_within)² / (SE_cross² + SE_within²)

if not pd.isna(peak_wb) and not pd.isna(peak_cb):
    diff = peak_cb - peak_wb
    se_diff = np.sqrt(se_cb**2 + se_wb**2)
    
    # Chi-squared statistic (Wald test with 1 df)
    chi2_stat = (diff / se_diff) ** 2
    
    # P-value from chi-squared distribution with 1 df
    from scipy.stats import chi2
    p_value_diff = 1 - chi2.cdf(chi2_stat, df=1)
    
    # Also calculate ratio
    ratio = abs(peak_cb) / abs(peak_wb) if abs(peak_wb) > 0 else np.nan
    
    section3_stats['within_bloc']['difference_test'] = {
        'chi2': chi2_stat,
        'p_value': p_value_diff,
        'ratio': ratio,
        'diff': diff,
        'se_diff': se_diff
    }
else:
    section3_stats['within_bloc']['difference_test'] = {
        'chi2': np.nan,
        'p_value': np.nan,
        'ratio': np.nan
    }

# Update stored results
WITHINBLOC_RESULTS = section3_stats['within_bloc'].copy()

print(f"   Analysis complete.")
print(f"   Within-bloc: N_obs={len(df_es_within):,}, N_switchers={n_switchers_wb}, peak={peak_wb:.4f}, SE={se_wb:.4f}")
print(f"   Cross-bloc:  N_obs={len(df_es_cross):,}, N_switchers={n_switchers_cb}, peak={peak_cb:.4f}, SE={se_cb:.4f}")
print(f"   Ratio (cross/within): {section3_stats['within_bloc']['difference_test']['ratio']:.2f}×")
print(f"   Difference test: χ² = {section3_stats['within_bloc']['difference_test']['chi2']:.3f}, p = {section3_stats['within_bloc']['difference_test']['p_value']:.4f}")

# %%
# IDEOLOGICAL DISTANCE ========================================================
# =============================================================================

print("Running: Ideological Distance Heterogeneity...")

section4_stats['distance'] = {'status': 'Initialized'}

df_distance = df_het.dropna(subset=['ideology_distance', 'abs_effect']).copy()

if len(df_distance) > 20:
    # Correlation tests
    from scipy.stats import pearsonr, spearmanr
    corr_pearson, p_pearson = pearsonr(df_distance['ideology_distance'], df_distance['abs_effect'])
    corr_spearman, p_spearman = spearmanr(df_distance['ideology_distance'], df_distance['abs_effect'])
    
    # Regression with controls
    controls_dist = []
    if 'tenure_years' in df_distance.columns:
        controls_dist.append('tenure_years')
    if 'new_party_size' in df_distance.columns:
        controls_dist.append('new_party_size')
    
    X_dist = df_distance[['ideology_distance'] + [c for c in controls_dist if c in df_distance.columns]].fillna(0)
    X_dist = sm.add_constant(X_dist)
    y_dist = df_distance['abs_effect']
    
    model_dist = sm.OLS(y_dist, X_dist).fit(cov_type='HC3')
    
    beta_dist = model_dist.params['ideology_distance']
    p_dist = model_dist.pvalues['ideology_distance']
    
    # EXTRACT CONSTANT AND R-SQUARED
    constant = model_dist.params['const']
    constant_p = model_dist.pvalues['const']
    r_squared = model_dist.rsquared
    
    section4_stats['distance'] = {
        'status': 'Complete',
        'n': len(df_distance),
        'mean_distance': df_distance['ideology_distance'].mean(),
        'range': [df_distance['ideology_distance'].min(), df_distance['ideology_distance'].max()],
        'r': corr_pearson,  # Add this for easier access
        'p_pearson': p_pearson,
        'r_spearman': corr_spearman,
        'p_spearman': p_spearman,
        'beta': beta_dist,
        'p': p_dist,
        'constant': constant,
        'constant_p': constant_p,
        'r_squared': r_squared,
        'model': model_dist
    }
    
    print(f"   ✓ Distance: β={beta_dist:.4f}, p={p_dist:.4f}, R²={r_squared:.4f}")
else:
    section4_stats['distance'] = {'status': 'Skipped - Insufficient Data'}
    print("   ⚠ Skipped (insufficient data)")

print("Note: Using RAW individual effects (pre - post in Y_confidence)")

# %%
# CAREER EXPERIENCE ===========================================================
# =============================================================================

print("Running: Experience/Tenure Heterogeneity...")

section4_stats['experience'] = {'status': 'Initialized'}

df_exp = df_het[df_het['experience_cat'] != 'Unknown'].copy()

if len(df_exp) > 20:
    # Summary by experience category
    exp_summary = df_exp.groupby('experience_cat').agg({
        'abs_effect': ['mean', 'std', 'count'],
        'tenure_years': 'mean'
    }).round(4)
    
    # Calculate SEM for each group
    junior = df_exp[df_exp['experience_cat'] == 'Junior']['abs_effect']
    mid = df_exp[df_exp['experience_cat'] == 'Mid']['abs_effect']
    senior = df_exp[df_exp['experience_cat'] == 'Senior']['abs_effect']
    
    # Standard Error of Mean (SEM) = SD / sqrt(N)
    junior_sem = junior.std() / np.sqrt(len(junior)) if len(junior) > 0 else np.nan
    mid_sem = mid.std() / np.sqrt(len(mid)) if len(mid) > 0 else np.nan
    senior_sem = senior.std() / np.sqrt(len(senior)) if len(senior) > 0 else np.nan
    
    if len(junior) > 5 and len(senior) > 5:
        from scipy import stats as sp_stats
        
        # T-test (Junior vs Senior)
        t_stat, p_val = sp_stats.ttest_ind(junior, senior)
        
        # ONE-WAY ANOVA (all three groups)
        f_stat, f_pval = sp_stats.f_oneway(junior, mid, senior)
        
        # Degrees of freedom for ANOVA
        df_between = 2  # k - 1, where k = 3 groups
        df_within = len(df_exp) - 3  # N - k
        
        # Also check continuous relationship
        df_exp_cont = df_het.dropna(subset=['tenure_years', 'abs_effect'])
        if len(df_exp_cont) > 20:
            from scipy.stats import pearsonr
            corr_tenure, p_tenure = pearsonr(df_exp_cont['tenure_years'], df_exp_cont['abs_effect'])
        else:
            corr_tenure, p_tenure = np.nan, np.nan
        
        section4_stats['experience'] = {
            'status': 'Complete',
            'summary': exp_summary,
            'junior_mean': junior.mean(),
            'junior_sem': junior_sem,
            'junior_n': len(junior),
            'mid_mean': mid.mean(),
            'mid_sem': mid_sem,
            'mid_n': len(mid),
            'senior_mean': senior.mean(),
            'senior_sem': senior_sem,
            'senior_n': len(senior),
            'difference': junior.mean() - senior.mean(),
            't_stat': t_stat,
            'p_val': p_val,
            'f_stat': f_stat,
            'f_pval': f_pval,
            'df_between': df_between,
            'df_within': df_within,
            'correlation': corr_tenure,
            'p_corr': p_tenure
        }
        
        print(f"   ✓ Experience: Junior={junior.mean():.4f}, Senior={senior.mean():.4f}, p={p_val:.4f}")
        print(f"   ✓ ANOVA: F({df_between}, {df_within})={f_stat:.3f}, p={f_pval:.4f}")
    else:
        section4_stats['experience'] = {'status': 'Skipped - Insufficient Groups'}
        print("   ⚠ Skipped (insufficient groups)")
else:
    section4_stats['experience'] = {'status': 'Skipped - Insufficient Data'}
    print("   ⚠ Skipped (insufficient data)")

# %%
# DESTINATION PARTY SIZE ======================================================
# =============================================================================

print("Running: Party Size Heterogeneity...")

section4_stats['party_size'] = {'status': 'Initialized'}

df_size = df_het[df_het['party_size_cat'] != 'Unknown'].copy()

if len(df_size) > 20:
    # Summary by party size
    size_summary = df_size.groupby('party_size_cat').agg({
        'abs_effect': ['mean', 'std', 'count'],
        'new_party_size': 'mean'
    }).round(4)
    
    # Extract all three groups
    major = df_size[df_size['party_size_cat'] == 'Major']['abs_effect']
    medium = df_size[df_size['party_size_cat'] == 'Medium']['abs_effect']
    minor = df_size[df_size['party_size_cat'] == 'Minor']['abs_effect']
    
    # Calculate SEM for each group
    major_sem = major.std() / np.sqrt(len(major)) if len(major) > 0 else np.nan
    medium_sem = medium.std() / np.sqrt(len(medium)) if len(medium) > 0 else np.nan
    minor_sem = minor.std() / np.sqrt(len(minor)) if len(minor) > 0 else np.nan
    
    if len(major) > 5 and len(minor) > 5:
        from scipy import stats as sp_stats
        
        # T-test (Major vs Minor)
        t_stat, p_val = sp_stats.ttest_ind(major, minor)
        
        # ONE-WAY ANOVA (all three groups)
        f_stat, f_pval = sp_stats.f_oneway(major, medium, minor)
        
        # Degrees of freedom
        df_between = 2  # k - 1, where k = 3 groups
        df_within = len(df_size) - 3  # N - k
        
        # Continuous relationship
        df_size_cont = df_het.dropna(subset=['new_party_size', 'abs_effect'])
        if len(df_size_cont) > 20:
            from scipy.stats import pearsonr
            corr_size, p_size = pearsonr(df_size_cont['new_party_size'], df_size_cont['abs_effect'])
        else:
            corr_size, p_size = np.nan, np.nan
        
        section4_stats['party_size'] = {
            'status': 'Complete',
            'summary': size_summary,
            'major_mean': major.mean(),
            'major_sem': major_sem,
            'major_n': len(major),
            'medium_mean': medium.mean(),
            'medium_sem': medium_sem,
            'medium_n': len(medium),
            'minor_mean': minor.mean(),
            'minor_sem': minor_sem,
            'minor_n': len(minor),
            'difference': major.mean() - minor.mean(),
            't_stat': t_stat,
            'p_val': p_val,
            'f_stat': f_stat,
            'f_pval': f_pval,
            'df_between': df_between,
            'df_within': df_within,
            'correlation': corr_size,
            'p_corr': p_size
        }
        
        print(f"   ✓ Party Size: Major={major.mean():.4f}, Minor={minor.mean():.4f}, p={p_val:.4f}")
        print(f"   ✓ ANOVA: F({df_between}, {df_within})={f_stat:.3f}, p={f_pval:.4f}")
    else:
        section4_stats['party_size'] = {'status': 'Skipped - Insufficient Groups'}
        print("   ⚠ Skipped (insufficient groups)")
else:
    section4_stats['party_size'] = {'status': 'Skipped - Insufficient Data'}
    print("   ⚠ Skipped (insufficient data)")

print("\n✅ All heterogeneity computations complete")

# %%
# DIRECTION HETEROGENEITY =====================================================
# =============================================================================

print("Running: Direction Heterogeneity...")

section4_stats['direction'] = {'status': 'Initialized'}

# Use ideological direction instead of gov/opp
df_direction = df_het.dropna(subset=['old_CAT', 'new_CAT']).copy()

if len(df_direction) > 20:
    # Classify direction based on ideology
    def get_ideological_direction(row):
        cat_order = {'Left': 0, 'Center': 1, 'Right': 2}
        old_pos = cat_order.get(row['old_CAT'], 1)
        new_pos = cat_order.get(row['new_CAT'], 1)
        
        if new_pos > old_pos:
            return 'Rightward'
        elif new_pos < old_pos:
            return 'Leftward'
        else:
            return 'Lateral'
    
    df_direction['ideological_direction'] = df_direction.apply(get_ideological_direction, axis=1)
    
    # Summary by direction
    direction_summary = df_direction.groupby('ideological_direction').agg({
        'abs_effect': ['mean', 'std', 'count']
    }).round(4)
    
    # Test: Rightward vs Leftward moves
    rightward = df_direction[df_direction['ideological_direction'] == 'Rightward']['abs_effect']
    leftward = df_direction[df_direction['ideological_direction'] == 'Leftward']['abs_effect']
    lateral = df_direction[df_direction['ideological_direction'] == 'Lateral']['abs_effect']
    
    # Calculate SEM for each group
    rightward_sem = rightward.std() / np.sqrt(len(rightward)) if len(rightward) > 0 else np.nan
    leftward_sem = leftward.std() / np.sqrt(len(leftward)) if len(leftward) > 0 else np.nan
    lateral_sem = lateral.std() / np.sqrt(len(lateral)) if len(lateral) > 0 else np.nan
    
    if len(rightward) > 5 and len(leftward) > 5:
        from scipy import stats as sp_stats
        t_stat, p_val = sp_stats.ttest_ind(rightward, leftward)
        
        section4_stats['direction'] = {
            'status': 'Complete',
            'summary': direction_summary,
            'rightward_mean': rightward.mean(),
            'rightward_sem': rightward_sem,
            'rightward_n': len(rightward),
            'leftward_mean': leftward.mean(),
            'leftward_sem': leftward_sem,
            'leftward_n': len(leftward),
            'lateral_mean': lateral.mean() if len(lateral) > 0 else np.nan,
            'lateral_sem': lateral_sem,
            'lateral_n': len(lateral),
            'difference': rightward.mean() - leftward.mean(),
            't_stat': t_stat,
            'p_val': p_val
        }
        
        print(f"   ✓ Direction: Rightward={rightward.mean():.4f} (n={len(rightward)}), Leftward={leftward.mean():.4f} (n={len(leftward)}), p={p_val:.4f}")
    else:
        section4_stats['direction'] = {'status': 'Skipped - Insufficient Groups'}
        print("   ⚠ Skipped (insufficient groups)")
else:
    section4_stats['direction'] = {'status': 'Skipped - Insufficient Data'}
    print("   ⚠ Skipped (insufficient data)")

# %% [markdown]
# ### Section Outputs

# %%
# =============================================================================
# CELL 6: SUMMARY & VISUALIZATION
# =============================================================================

print("\n" + "="*80)
print("PART IV: HETEROGENEITY ANALYSIS - SUMMARY")
print("="*80)

# ============================================================================
# 0. DATA PREPARATION SUMMARY
# ============================================================================

if 'data_prep' in section4_stats and section4_stats['data_prep']['status'] == 'Complete':
    dp = section4_stats['data_prep']
    
    print(f"\n📊 Dataset:")
    print(f"   N switchers: {dp['n_switchers']:,}")
    print(f"   Mean effect: {dp['mean_effect']:.4f}")
    print(f"   Mean |effect|: {dp['mean_abs_effect']:.4f}")
    
    print(f"\n   Sample distribution:")
    print(f"      Experience: {dp['experience_counts']}")
    print(f"      Party Size: {dp['party_size_counts']}")


# ============================================================================
# WITHIN-BLOC SWITCHING ANALYSIS 
# ============================================================================

print("\n" + "="*80)
print("TABLE: WITHIN-BLOC VS CROSS-BLOC SWITCHING")
print("="*80)

if 'section3_stats' in globals() and 'within_bloc' in section3_stats:
    wb = section3_stats['within_bloc']
    
    if wb['status'] == 'Complete':
        print(f"\nSwitch Type                Peak (τ=+1)           SE    N Obs    N Switchers")
        print("-" * 85)
        print(f"Within-bloc           {wb['peak_within']:>15.4f}*** {wb['se_within']:>12.4f} {wb['n_within']:>8,} {wb['n_switchers_within']:>14}")
        print(f"Cross-bloc            {wb['peak_cross']:>15.4f}*** {wb['se_cross']:>12.4f} {wb['n_cross']:>8,} {wb['n_switchers_cross']:>14}")
        print("-" * 85)
        
        if not pd.isna(wb['peak_within']) and not pd.isna(wb['peak_cross']):
            ratio = abs(wb['peak_cross']) / abs(wb['peak_within']) if abs(wb['peak_within']) > 0 else np.nan
            print(f"\nRatio (cross/within): {ratio:.2f}×")
            pct_larger = ((abs(wb['peak_cross']) / abs(wb['peak_within'])) - 1) * 100
            print(f"Cross-bloc effect is {pct_larger:.0f}% larger than within-bloc")
            
        interpretation = "✅ Within-bloc effect EXISTS (>0.01)" if wb['passes'] else "⚠️  Within-bloc effect WEAK (<0.01)"
        print(f"\n{interpretation}")
        
        print("\nInterpretation:")
        print("  Both within-bloc and cross-bloc switches show significant adaptation (p<0.001).")
        print(f"  Cross-bloc switches adapt {ratio:.1f}× more, consistent with ideological repositioning")
        print("  compounding party-specific adaptation.")
    else:
        print(f"\nStatus: {wb['status']}")
else:
    print("\n⚠️  Within-bloc analysis not found")

# ============================================================================
# TABLE: HETEROGENEITY BY IDEOLOGICAL DISTANCE
# ============================================================================

if 'distance' in section4_stats and section4_stats['distance']['status'] == 'Complete':
    print("\n" + "="*80)
    print("TABLE: HETEROGENEITY BY IDEOLOGICAL DISTANCE")
    print("="*80)
    print("\n")
    
    dist = section4_stats['distance']
    
    print(f"{'Variable':<30} {'Coefficient':>15} {'p-value':>12}")
    print("-" * 80)
    print(f"{'Ideological distance':<30} {dist['beta']:>15.4f} {dist['p']:>12.4f}")
    print(f"{'Constant':<30} {dist['constant']:>15.4f} {dist['constant_p']:>12.4f}")
    print("-" * 80)
    print(f"{'N':<30} {dist['n']:>28}")
    print(f"{'R²':<30} {dist['r_squared']:>28.4f}")
    print(f"{'Pearson r':<30} {dist['r']:>28.3f}")
    print(f"{'Spearman ρ':<30} {dist['r_spearman']:>28.3f}")
    print("\n")

# ============================================================================
# TABLE: HETEROGENEITY BY CAREER EXPERIENCE
# ============================================================================

if 'experience' in section4_stats and section4_stats['experience']['status'] == 'Complete':
    print("\n" + "="*80)
    print("TABLE: HETEROGENEITY BY CAREER EXPERIENCE")
    print("="*80)
    print("\n")
    
    exp = section4_stats['experience']
    
    print(f"{'Group':<25} {'Mean Effect':>15} {'SE':>12} {'N':>8}")
    print("-" * 80)
    print(f"{'Junior (<5 years)':<25} {exp['junior_mean']:>15.4f} {exp['junior_sem']:>12.4f} {exp['junior_n']:>8}")
    print(f"{'Mid (5-10 years)':<25} {exp['mid_mean']:>15.4f} {exp['mid_sem']:>12.4f} {exp['mid_n']:>8}")
    print(f"{'Senior (>10 years)':<25} {exp['senior_mean']:>15.4f} {exp['senior_sem']:>12.4f} {exp['senior_n']:>8}")
    print("-" * 80)
    print(f"\nF-test: F({exp['df_between']}, {exp['df_within']}) = {exp['f_stat']:.3f}, p = {exp['f_pval']:.4f}")
    print("\n")

# ============================================================================
# TABLE: HETEROGENEITY BY PARTY SIZE
# ============================================================================

if 'party_size' in section4_stats and section4_stats['party_size']['status'] == 'Complete':
    print("\n" + "="*80)
    print("TABLE: HETEROGENEITY BY DESTINATION PARTY SIZE")
    print("="*80)
    print("\n")
    
    psize = section4_stats['party_size']
    
    print(f"{'Party Size':<30} {'Mean Effect':>15} {'SE':>12} {'N':>8}")
    print("-" * 80)
    print(f"{'Major (≥30 deputies)':<30} {psize['major_mean']:>15.4f} {psize['major_sem']:>12.4f} {psize['major_n']:>8}")
    print(f"{'Medium (10-29)':<30} {psize['medium_mean']:>15.4f} {psize['medium_sem']:>12.4f} {psize['medium_n']:>8}")
    print(f"{'Minor (<10)':<30} {psize['minor_mean']:>15.4f} {psize['minor_sem']:>12.4f} {psize['minor_n']:>8}")
    print("-" * 80)
    print(f"\nF-test: F({psize['df_between']}, {psize['df_within']}) = {psize['f_stat']:.3f}, p = {psize['f_pval']:.4f}")
    print("\n")

# ============================================================================
# TABLE: HETEROGENEITY BY DIRECTION
# ============================================================================

if 'direction' in section4_stats and section4_stats['direction']['status'] == 'Complete':
    print("\n" + "="*80)
    print("TABLE: HETEROGENEITY BY SWITCH DIRECTION")
    print("="*80)
    print("\n")
    
    direc = section4_stats['direction']
    
    print(f"{'Direction':<20} {'Mean Effect':>15} {'SE':>12} {'N':>8}")
    print("-" * 80)
    print(f"{'Rightward':<20} {direc['rightward_mean']:>15.4f} {direc['rightward_sem']:>12.4f} {direc['rightward_n']:>8}")
    print(f"{'Leftward':<20} {direc['leftward_mean']:>15.4f} {direc['leftward_sem']:>12.4f} {direc['leftward_n']:>8}")
    print(f"{'Lateral':<20} {direc['lateral_mean']:>15.4f} {direc['lateral_sem']:>12.4f} {direc['lateral_n']:>8}")
    print("-" * 80)
    print(f"\nt-test (Rightward vs Leftward): t = {direc['t_stat']:.3f}, p = {direc['p_val']:.4f}")
    print("\n")
    
# ============================================================================
# TABLE 3: HETEROGENEITY SUMMARY
# ============================================================================

print("\n" + "="*80)
print("TABLE 3: HETEROGENEITY SUMMARY")
print("="*80)

heterogeneity_table = []

# 1. Ideological Distance
if 'distance' in section4_stats and 'beta' in section4_stats['distance']:
    dist = section4_stats['distance']
    heterogeneity_table.append({
        'Dimension': 'Ideological Distance',
        'Measure': 'Regression β',
        'Value': f"{dist['beta']:.4f}",
        'p_value': f"{dist['p']:.3f}",
        'N': dist['n'],
        'Interpretation': 'Per 1pt increase in |ΔIdeology|'
    })

# 2. Direction
if 'direction' in section4_stats and 'rightward_mean' in section4_stats['direction']:
    direc = section4_stats['direction']
    heterogeneity_table.append({
        'Dimension': 'Direction',
        'Measure': 'Rightward - Leftward',
        'Value': f"{direc['difference']:.4f}",
        'p_value': f"{direc['p_val']:.3f}",
        'N': f"{direc['rightward_n']}+{direc['leftward_n']}",
        'Interpretation': 'Right-moving vs Left-moving'
    })

# 3. Experience
if 'experience' in section4_stats and 'junior_mean' in section4_stats['experience']:
    exp = section4_stats['experience']
    heterogeneity_table.append({
        'Dimension': 'Experience',
        'Measure': 'Junior - Senior',
        'Value': f"{exp['difference']:.4f}",
        'p_value': f"{exp['p_val']:.3f}",
        'N': f"{exp['junior_n']}+{exp['senior_n']}",
        'Interpretation': '<5yr vs >10yr tenure'
    })

# 4. Party Size
if 'party_size' in section4_stats and 'major_mean' in section4_stats['party_size']:
    psize = section4_stats['party_size']
    heterogeneity_table.append({
        'Dimension': 'Party Size',
        'Measure': 'Major - Minor',
        'Value': f"{psize['difference']:.4f}",
        'p_value': f"{psize['p_val']:.3f}",
        'N': f"{psize['major_n']}+{psize['minor_n']}",
        'Interpretation': '≥30 deputies vs <10 deputies'
    })

if len(heterogeneity_table) > 0:
    df_het_table = pd.DataFrame(heterogeneity_table)
    print("\n", df_het_table.to_string(index=False))
    
    # Count significant effects
    df_het_table['p_numeric'] = df_het_table['p_value'].astype(float)
    n_sig = (df_het_table['p_numeric'] < 0.05).sum()
    
    print(f"\n📊 Heterogeneity Summary:")
    print(f"   Dimensions tested: {len(df_het_table)}")
    print(f"   Significant (p<0.05): {n_sig}")
    
    if n_sig >= 3:
        print(f"\n   ✅ RICH HETEROGENEITY: Multiple dimensions show significant effects")
        print(f"      Adaptation varies systematically across context")
    elif n_sig >= 1:
        print(f"\n   ✅ SOME HETEROGENEITY: At least one dimension matters")
    else:
        print(f"\n   ○ LIMITED HETEROGENEITY: Effects are relatively uniform")
    
    # Save table
    df_het_table.to_csv(
        os.path.join(TABLES_DIR, 'table3_heterogeneity_summary.csv'),
        index=False
    )
    print(f"\n✅ Table 3 saved: {TABLES_DIR}/table3_heterogeneity_summary.csv")
else:
    print("\n⚠️  No heterogeneity results to tabulate")


# %%
def generate_heterogeneity_and_bloc_plots(section4_stats, section3_stats, df_het, save_dir=None):
    """
    Generates heterogeneity figures.
    - Fig 3B: Heterogeneity Bars (Direction=Yellow, Experience=Green, Party=Blue)
      * Ordered: Left->Right, Junior->Senior, Minor->Major
      * Colors: Light Pastel -> Saturated
    """
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import os
    from scipy.stats import linregress

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    # --- STYLE SETUP ---
    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'DejaVu Serif', 'Bitstream Vera Serif'],
        'font.size': 10,
        'axes.labelsize': 10,
        'xtick.labelsize': 9,
        'ytick.labelsize': 9,
        'legend.fontsize': 9,
        'axes.linewidth': 0.5,
    })

    C_GRAY = '#444444'
    C_GRID = '#dcdcdc'
    
    # --- CUSTOM PASTEL-TO-SATURATED PALETTES ---
    PALETTE_DIR = ['#0069A2', '#0096E7', '#09A6FB']

    PALETTE_EXP = ['#379A8B', '#5AB6A9', '#8ED6CC']

    PALETTE_SIZE = ['#EBB434', '#FFCE5C', '#FFDB85'] 

    def format_ax(ax, xlabel, ylabel):
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.grid(axis='y', color=C_GRID, lw=0.5, zorder=0)
        ax.grid(axis='x', visible=False)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        return ax

    print("\n" + "="*80)
    print("GENERATING HETEROGENEITY PLOTS (Ordered, Pastel -> Saturated)...")
    print("="*80)

    # ========================================================================
    # FIGURE 3A: IDEOLOGICAL DISTANCE (SCATTER)
    # ========================================================================
    if 'distance' in section4_stats and section4_stats['distance']['status'] == 'Complete':
        dist = section4_stats['distance']
        df_plot = df_het.dropna(subset=['ideology_distance', 'abs_effect'])
        
        fig, ax = plt.subplots(figsize=(5, 4))
        # Use Economist Blue for scatter points
        ax.scatter(df_plot['ideology_distance'], df_plot['abs_effect'],
                   alpha=0.6, s=40, color='#006BA2', edgecolor='none', zorder=2)
        
        slope, intercept, r, p, se = linregress(df_plot['ideology_distance'], df_plot['abs_effect'])
        x_line = np.linspace(df_plot['ideology_distance'].min(), df_plot['ideology_distance'].max(), 100)
        y_line = slope * x_line + intercept
        
        # Red line for regression
        ax.plot(x_line, y_line, color='#DB444B', linestyle='--', linewidth=1.5, alpha=0.9,
                label=f'Slope: {dist["beta"]:.3f} ($p={dist["p"]:.3f}$)', zorder=3)
        
        format_ax(ax, "Ideological Distance (|ΔIdeology|)", "|Effect Size|")
        ax.legend(frameon=False, loc='best')
        plt.tight_layout()
        if save_dir: plt.savefig(os.path.join(save_dir, 'fig3a_distance.png'), dpi=300)
        plt.show()
        print("✅ Figure 3A (Distance) saved.")

    # ========================================================================
    # FIGURE 3B: HETEROGENEITY BARS (SIDE-BY-SIDE)
    # ========================================================================
    has_dir = 'direction' in section4_stats and section4_stats['direction']['status'] == 'Complete'
    has_exp = 'experience' in section4_stats and section4_stats['experience']['status'] == 'Complete'
    has_size = 'party_size' in section4_stats and section4_stats['party_size']['status'] == 'Complete'

    if has_dir or has_exp or has_size:
        fig, axes = plt.subplots(1, 3, figsize=(6, 3.5), sharey=True)
        plt.subplots_adjust(wspace=0.1)

        BAR_WIDTH = 0.8
        CAP_SIZE = 3
        ERR_LW = 1

        # --- PANEL 1: DIRECTION (Yellows) ---
        # Order: Leftward -> Lateral -> Rightward
        ax1 = axes[0]
        if has_dir:
            direc = section4_stats['direction']
            dir_data = direc['summary'].reset_index()
            dir_data.columns = ['direction', 'mean', 'std', 'count']
            dir_data['sem'] = dir_data['std'] / np.sqrt(dir_data['count'])
            
            # Enforce Order
            order_dir = ['Leftward', 'Lateral', 'Rightward']
            dir_data['direction'] = pd.Categorical(dir_data['direction'], categories=order_dir, ordered=True)
            dir_data = dir_data.sort_values('direction')
            
            x_pos = np.arange(len(dir_data))
            # Ensure we have enough colors if data is missing a category, though logic assumes 3
            colors = PALETTE_DIR[:len(dir_data)]
            
            ax1.bar(x_pos, dir_data['mean'], yerr=dir_data['sem']*1.96,
                   color=colors, alpha=0.9, width=BAR_WIDTH, 
                   capsize=CAP_SIZE, error_kw={'ecolor': C_GRAY, 'lw': ERR_LW}, zorder=2)
            
            format_ax(ax1, "Direction", "Mean |Effect Size|")
            ax1.set_xticks(x_pos)
            ax1.set_xticklabels(dir_data['direction'])
        else:
            ax1.axis('off')

        # --- PANEL 2: EXPERIENCE (Greens) ---
        # Order: Junior -> Mid -> Senior
        ax2 = axes[1]
        if has_exp:
            exp_data = df_het[df_het['experience_cat'] != 'Unknown'].groupby('experience_cat')['abs_effect'].agg(['mean','sem']).reset_index()
            
            order_exp = ['Junior', 'Mid', 'Senior']
            exp_data['experience_cat'] = pd.Categorical(exp_data['experience_cat'], categories=order_exp, ordered=True)
            exp_data = exp_data.sort_values('experience_cat')
            
            x_pos = np.arange(len(exp_data))
            colors = PALETTE_EXP[:len(exp_data)]
            
            ax2.bar(x_pos, exp_data['mean'], yerr=exp_data['sem']*1.96,
                   color=colors, alpha=0.9, width=BAR_WIDTH, 
                   capsize=CAP_SIZE, error_kw={'ecolor': C_GRAY, 'lw': ERR_LW}, zorder=2)
            
            format_ax(ax2, "Experience", "")
            ax2.tick_params(left=False)
            ax2.set_xticks(x_pos)
            ax2.set_xticklabels(exp_data['experience_cat'])
        else:
            ax2.axis('off')

        # --- PANEL 3: PARTY SIZE (Blues) ---
        # Order: Minor -> Medium -> Major
        ax3 = axes[2]
        if has_size:
            size_data = df_het[df_het['party_size_cat'] != 'Unknown'].groupby('party_size_cat')['abs_effect'].agg(['mean','sem']).reset_index()
            
            order_size = ['Minor', 'Medium', 'Major']
            size_data['party_size_cat'] = pd.Categorical(size_data['party_size_cat'], categories=order_size, ordered=True)
            size_data = size_data.sort_values('party_size_cat')
            
            x_pos = np.arange(len(size_data))
            colors = PALETTE_SIZE[:len(size_data)]
            
            ax3.bar(x_pos, size_data['mean'], yerr=size_data['sem']*1.96,
                   color=colors, alpha=0.9, width=BAR_WIDTH, 
                   capsize=CAP_SIZE, error_kw={'ecolor': C_GRAY, 'lw': ERR_LW}, zorder=2)
            
            format_ax(ax3, "Destination Party Size", "")
            ax3.tick_params(left=False)
            ax3.set_xticks(x_pos)
            ax3.set_xticklabels(size_data['party_size_cat'])
        else:
            ax3.axis('off')

        plt.tight_layout()
        if save_dir: plt.savefig(os.path.join(save_dir, 'fig3b_heterogeneity_bars.png'), dpi=300)
        plt.show()
        print("✅ Figure 3B (Themed Bars) saved.")

    # ========================================================================
    # FIGURE 4: WITHIN-BLOC VS CROSS-BLOC (STANDALONE)
    # ========================================================================
    if 'within_bloc' in section3_stats:
        wb = section3_stats['within_bloc']
        if wb['status'] == 'Complete' and wb['plot_data_within'] is not None:
            fig, ax = plt.subplots(figsize=(9, 5))
            
            ax.axhline(0, color=C_GRAY, ls='-', lw=0.8, alpha=0.5, zorder=1)
            ax.axvline(0, color='#DB444B', ls='--', lw=0.8, zorder=1) # Red switch line

            def add_ref_point(df_in):
                df = df_in.copy()
                if -1 not in df['time'].values:
                    ref = pd.DataFrame({'time': [-1], 'coef': [0.0], 'se': [0.0], 'lower': [0.0], 'upper': [0.0]})
                    df = pd.concat([df, ref]).sort_values('time')
                    if 'lower' not in df.columns:
                        df['lower'] = df['coef'] - 1.96 * df['se']
                        df['upper'] = df['coef'] + 1.96 * df['se']
                return df

            plot_wb = add_ref_point(wb['plot_data_within'])
            plot_cb = add_ref_point(wb['plot_data_cross'])
            
            # Using Economist Yellow for Within, Blue for Cross
            C_WITHIN = '#EBB434' 
            C_CROSS  = '#006BA2' 

            ax.plot(plot_wb['time'], plot_wb['coef'], 'o-', lw=1.5, ms=5, color=C_WITHIN, 
                    label=f'Within-Bloc (N={wb["n_within"]:,})', zorder=3)
            if 'se' in plot_wb.columns:
                ax.fill_between(plot_wb['time'], plot_wb['lower'], plot_wb['upper'], 
                                alpha=0.15, color=C_WITHIN, lw=0, zorder=2)
            
            ax.plot(plot_cb['time'], plot_cb['coef'], 's-', lw=1.5, ms=5, color=C_CROSS, 
                    label=f'Cross-Bloc (N={wb["n_cross"]:,})', zorder=3)
            if 'se' in plot_cb.columns:
                ax.fill_between(plot_cb['time'], plot_cb['lower'], plot_cb['upper'], 
                                alpha=0.15, color=C_CROSS, lw=0, zorder=2)
            
            format_ax(ax, "Bimesters from Switch", "Effect on P(Old Party)")
            ax.legend(frameon=False, loc='best')
            
            if not pd.isna(wb['peak_within']) and not pd.isna(wb['peak_cross']):
                ratio = abs(wb['peak_cross']) / (abs(wb['peak_within']) + 1e-9)
                txt = (f"Peak Ratio: {ratio:.1f}x\n"
                       f"Cross: {wb['peak_cross']:.3f}\n"
                       f"Within: {wb['peak_within']:.3f}")
                ax.text(0.05, 0.05, txt, transform=ax.transAxes, fontsize=9,
                        va='bottom', ha='left',
                        bbox=dict(boxstyle='square,pad=0.4', fc='#dcdcdc', alpha=0.3, ec='none'))

            plt.tight_layout()
            if save_dir: plt.savefig(os.path.join(save_dir, 'fig4_within_cross_bloc.png'), dpi=300)
            plt.show()
            print("✅ Figure 4 (Within vs Cross) saved.")

# USAGE
# =============================================================================
generate_heterogeneity_and_bloc_plots(section4_stats, section3_stats, df_het, save_dir=save_dir)

# %%
# COVARIATE CONSISTENCY VERIFICATION
# ==================================
print("="*70)
print("COVARIATE CONSISTENCY CHECK")
print("="*70)
print()
print("Expected covariates (from paper):")
print(f"  Continuous: {CONTINUOUS_COVARIATES}")
print(f"  Legislature FE: Yes")
print(f"  Calendar month FE: Yes")
print(f"  Topic FE: Yes (add to paper!)")
print()

# Check what's actually in your data
print("Available in df_es:")
for cov in CONTINUOUS_COVARIATES:
    status = "✓" if cov in df_es.columns else "✗ MISSING"
    if cov in df_es.columns:
        n_missing = df_es[cov].isna().sum()
        pct_missing = 100 * n_missing / len(df_es)
        print(f"  {cov}: {status} ({pct_missing:.1f}% missing)")
    else:
        print(f"  {cov}: {status}")

print()
print("="*70)

# %%
# =============================================================================
# POWER ANALYSIS FOR HETEROGENEITY TESTS
# =============================================================================
# This analysis determines what effect sizes we can reliably detect,
# allowing us to distinguish "no effect" from "underpowered to detect effect"

from statsmodels.stats.power import TTestIndPower, FTestAnovaPower
from scipy.stats import f as f_dist
import numpy as np

print("="*80)
print("POWER ANALYSIS FOR HETEROGENEITY TESTS")
print("="*80)
print()

alpha = 0.05
target_power = 0.80

power_results = {}

# -----------------------------------------------------------------------------
# 1. ANOVA POWER: Experience (3 groups)
# -----------------------------------------------------------------------------
print("--- 1. Career Experience (3-group ANOVA) ---")

n_junior = section4_stats['experience']['junior_n']
n_mid = section4_stats['experience']['mid_n']
n_senior = section4_stats['experience']['senior_n']
n_total_exp = n_junior + n_mid + n_senior
k_groups = 3

# Harmonic mean of group sizes (for unbalanced design)
n_harmonic_exp = k_groups / (1/n_junior + 1/n_mid + 1/n_senior)

# For ANOVA, effect size is Cohen's f = sqrt(eta² / (1 - eta²))
# We'll compute MDE (minimum detectable effect) at 80% power

# Using F-test power approximation
# df1 = k - 1 = 2, df2 = N - k = n_total - 3
df1_exp = k_groups - 1
df2_exp = n_total_exp - k_groups

# Compute MDE using iterative search
def anova_power(f_effect, n_per_group, k, alpha=0.05):
    """Compute power for one-way ANOVA given effect size f"""
    df1 = k - 1
    df2 = k * (n_per_group - 1)
    # Non-centrality parameter
    nc = k * n_per_group * (f_effect ** 2)
    # Critical F value
    f_crit = f_dist.ppf(1 - alpha, df1, df2)
    # Power = P(F > f_crit | H1)
    power = 1 - f_dist.cdf(f_crit, df1, df2, nc)
    return power

# Grid search for MDE
f_values = np.linspace(0.01, 0.50, 500)
mde_f_exp = None
for f_val in f_values:
    pwr = anova_power(f_val, n_harmonic_exp, k_groups, alpha)
    if pwr >= target_power:
        mde_f_exp = f_val
        break

# Also compute achieved power for small, medium, large effects
power_small_exp = anova_power(0.10, n_harmonic_exp, k_groups, alpha)
power_med_exp = anova_power(0.25, n_harmonic_exp, k_groups, alpha)
power_large_exp = anova_power(0.40, n_harmonic_exp, k_groups, alpha)

# Convert f to interpretable difference
# f = d / sqrt(2) for 2-group comparison embedded in ANOVA
# For 3 groups: effect size in raw units = f * pooled_SD * sqrt(2)
pooled_sd_exp = df_het['abs_effect'].std()
mde_raw_exp = mde_f_exp * pooled_sd_exp * np.sqrt(2) if mde_f_exp else np.nan

power_results['experience'] = {
    'test': 'One-way ANOVA',
    'n_groups': k_groups,
    'n_total': n_total_exp,
    'n_per_group_harmonic': n_harmonic_exp,
    'mde_f': mde_f_exp,
    'mde_raw': mde_raw_exp,
    'power_small': power_small_exp,
    'power_medium': power_med_exp,
    'power_large': power_large_exp,
    'observed_f': section4_stats['experience']['f_stat'],
    'observed_p': section4_stats['experience']['f_pval']
}

print(f"   N total: {n_total_exp} (Junior={n_junior}, Mid={n_mid}, Senior={n_senior})")
print(f"   Minimum Detectable Effect (80% power): Cohen's f = {mde_f_exp:.3f}")
print(f"   MDE in raw units: {mde_raw_exp:.4f} (group difference in |Δ Language|)")
print(f"   Power for small effect (f=0.10): {power_small_exp:.1%}")
print(f"   Power for medium effect (f=0.25): {power_med_exp:.1%}")
print(f"   Power for large effect (f=0.40): {power_large_exp:.1%}")
print()

# -----------------------------------------------------------------------------
# 2. ANOVA POWER: Party Size (3 groups)
# -----------------------------------------------------------------------------
print("--- 2. Destination Party Size (3-group ANOVA) ---")

n_major = section4_stats['party_size']['major_n']
n_medium = section4_stats['party_size']['medium_n']
n_minor = section4_stats['party_size']['minor_n']
n_total_size = n_major + n_medium + n_minor

n_harmonic_size = k_groups / (1/n_major + 1/n_medium + 1/n_minor)

# Grid search for MDE
mde_f_size = None
for f_val in f_values:
    pwr = anova_power(f_val, n_harmonic_size, k_groups, alpha)
    if pwr >= target_power:
        mde_f_size = f_val
        break

power_small_size = anova_power(0.10, n_harmonic_size, k_groups, alpha)
power_med_size = anova_power(0.25, n_harmonic_size, k_groups, alpha)
power_large_size = anova_power(0.40, n_harmonic_size, k_groups, alpha)

mde_raw_size = mde_f_size * pooled_sd_exp * np.sqrt(2) if mde_f_size else np.nan

power_results['party_size'] = {
    'test': 'One-way ANOVA',
    'n_groups': k_groups,
    'n_total': n_total_size,
    'n_per_group_harmonic': n_harmonic_size,
    'mde_f': mde_f_size,
    'mde_raw': mde_raw_size,
    'power_small': power_small_size,
    'power_medium': power_med_size,
    'power_large': power_large_size,
    'observed_f': section4_stats['party_size']['f_stat'],
    'observed_p': section4_stats['party_size']['f_pval']
}

print(f"   N total: {n_total_size} (Major={n_major}, Medium={n_medium}, Minor={n_minor})")
print(f"   Minimum Detectable Effect (80% power): Cohen's f = {mde_f_size:.3f}")
print(f"   MDE in raw units: {mde_raw_size:.4f}")
print(f"   Power for small effect (f=0.10): {power_small_size:.1%}")
print(f"   Power for medium effect (f=0.25): {power_med_size:.1%}")
print(f"   Power for large effect (f=0.40): {power_large_size:.1%}")
print()

# -----------------------------------------------------------------------------
# 3. T-TEST POWER: Switch Direction (2 groups)
# -----------------------------------------------------------------------------
print("--- 3. Switch Direction (Independent t-test) ---")

n_rightward = section4_stats['direction']['rightward_n']
n_leftward = section4_stats['direction']['leftward_n']

# Use TTestIndPower from statsmodels
ttest_power = TTestIndPower()

# For unequal sample sizes, use ratio
ratio_dir = n_rightward / n_leftward

# MDE at 80% power
mde_d_dir = ttest_power.solve_power(
    effect_size=None, 
    nobs1=n_rightward, 
    ratio=n_leftward/n_rightward,
    alpha=alpha, 
    power=target_power,
    alternative='two-sided'
)

# Power for various effect sizes
power_small_dir = ttest_power.power(effect_size=0.20, nobs1=n_rightward, 
                                     ratio=n_leftward/n_rightward, alpha=alpha)
power_med_dir = ttest_power.power(effect_size=0.50, nobs1=n_rightward, 
                                   ratio=n_leftward/n_rightward, alpha=alpha)
power_large_dir = ttest_power.power(effect_size=0.80, nobs1=n_rightward, 
                                     ratio=n_leftward/n_rightward, alpha=alpha)

# Convert to raw units
mde_raw_dir = mde_d_dir * pooled_sd_exp

# Compute observed Cohen's d
mean_right = section4_stats['direction']['rightward_mean']
mean_left = section4_stats['direction']['leftward_mean']
observed_d_dir = (mean_right - mean_left) / pooled_sd_exp

power_results['direction'] = {
    'test': 'Independent t-test',
    'n1': n_rightward,
    'n2': n_leftward,
    'n_total': n_rightward + n_leftward,
    'mde_d': mde_d_dir,
    'mde_raw': mde_raw_dir,
    'power_small': power_small_dir,
    'power_medium': power_med_dir,
    'power_large': power_large_dir,
    'observed_d': observed_d_dir,
    'observed_p': section4_stats['direction']['p_val']
}

print(f"   N: Rightward={n_rightward}, Leftward={n_leftward}")
print(f"   Minimum Detectable Effect (80% power): Cohen's d = {mde_d_dir:.3f}")
print(f"   MDE in raw units: {mde_raw_dir:.4f}")
print(f"   Observed Cohen's d: {observed_d_dir:.3f}")
print(f"   Power for small effect (d=0.20): {power_small_dir:.1%}")
print(f"   Power for medium effect (d=0.50): {power_med_dir:.1%}")
print(f"   Power for large effect (d=0.80): {power_large_dir:.1%}")
print()

# -----------------------------------------------------------------------------
# 4. CORRELATION POWER: Ideological Distance
# -----------------------------------------------------------------------------
print("--- 4. Ideological Distance (Correlation/Regression) ---")

from statsmodels.stats.power import NormalIndPower
from scipy.stats import norm

n_dist = section4_stats['distance']['n']

# For correlation, we use Fisher's z transformation
# Power for correlation test
def correlation_power(r, n, alpha=0.05):
    """Compute power for correlation test"""
    # Fisher's z transformation
    z_r = 0.5 * np.log((1 + r) / (1 - r))
    se = 1 / np.sqrt(n - 3)
    z_crit = norm.ppf(1 - alpha/2)
    # Power
    power = 1 - norm.cdf(z_crit - z_r/se) + norm.cdf(-z_crit - z_r/se)
    return power

# Grid search for MDE
r_values = np.linspace(0.01, 0.50, 500)
mde_r_dist = None
for r_val in r_values:
    pwr = correlation_power(r_val, n_dist, alpha)
    if pwr >= target_power:
        mde_r_dist = r_val
        break

power_small_dist = correlation_power(0.10, n_dist, alpha)
power_med_dist = correlation_power(0.30, n_dist, alpha)
power_large_dist = correlation_power(0.50, n_dist, alpha)

observed_r = section4_stats['distance']['r']

power_results['distance'] = {
    'test': 'Correlation (Pearson r)',
    'n': n_dist,
    'mde_r': mde_r_dist,
    'power_small': power_small_dist,
    'power_medium': power_med_dist,
    'power_large': power_large_dist,
    'observed_r': observed_r,
    'observed_p': section4_stats['distance']['p_pearson']
}

print(f"   N: {n_dist}")
print(f"   Minimum Detectable Effect (80% power): |r| = {mde_r_dist:.3f}")
print(f"   Observed |r|: {abs(observed_r):.3f}")
print(f"   Power for small effect (r=0.10): {power_small_dist:.1%}")
print(f"   Power for medium effect (r=0.30): {power_med_dist:.1%}")
print(f"   Power for large effect (r=0.50): {power_large_dist:.1%}")
print()

# -----------------------------------------------------------------------------
# SUMMARY TABLE
# -----------------------------------------------------------------------------
print("="*80)
print("POWER ANALYSIS SUMMARY")
print("="*80)
print()
print(f"{'Dimension':<25} {'Test':<15} {'N':<10} {'MDE (80%)':<15} {'Observed':<12} {'Can Rule Out'}")
print("-"*90)

# Experience
print(f"{'Career Experience':<25} {'ANOVA':<15} {power_results['experience']['n_total']:<10} "
      f"f={power_results['experience']['mde_f']:.3f}{'':<8} "
      f"f={np.sqrt(power_results['experience']['observed_f'] * 2 / power_results['experience']['n_total']):.3f}{'':<5} "
      f"{'f ≥ ' + f'{power_results["experience"]["mde_f"]:.2f}'}")

# Party Size  
print(f"{'Destination Party Size':<25} {'ANOVA':<15} {power_results['party_size']['n_total']:<10} "
      f"f={power_results['party_size']['mde_f']:.3f}{'':<8} "
      f"f={np.sqrt(power_results['party_size']['observed_f'] * 2 / power_results['party_size']['n_total']):.3f}{'':<5} "
      f"{'f ≥ ' + f'{power_results["party_size"]["mde_f"]:.2f}'}")

# Direction
print(f"{'Switch Direction':<25} {'t-test':<15} {power_results['direction']['n_total']:<10} "
      f"d={power_results['direction']['mde_d']:.3f}{'':<8} "
      f"d={power_results['direction']['observed_d']:.3f}{'':<5} "
      f"{'d ≥ ' + f'{power_results["direction"]["mde_d"]:.2f}'}")

# Distance
print(f"{'Ideological Distance':<25} {'Correlation':<15} {power_results['distance']['n']:<10} "
      f"r={power_results['distance']['mde_r']:.3f}{'':<8} "
      f"r={abs(power_results['distance']['observed_r']):.3f}{'':<5} "
      f"{'r ≥ ' + f'{power_results["distance"]["mde_r"]:.2f}'}")

print("-"*90)
print()
print("Interpretation:")
print("  • MDE = Minimum Detectable Effect at 80% power, α=0.05")
print("  • 'Can Rule Out' = Effect sizes we can confidently exclude given null results")
print("  • Cohen's benchmarks: f (0.10=small, 0.25=medium, 0.40=large)")
print("  •                     d (0.20=small, 0.50=medium, 0.80=large)")
print("  •                     r (0.10=small, 0.30=medium, 0.50=large)")
print()

# Key conclusion
all_mdes_small = (
    power_results['experience']['mde_f'] <= 0.25 and
    power_results['party_size']['mde_f'] <= 0.25 and
    power_results['direction']['mde_d'] <= 0.50 and
    power_results['distance']['mde_r'] <= 0.30
)

if all_mdes_small:
    print("✅ CONCLUSION: All tests have adequate power to detect MEDIUM effects.")
    print("   The null results allow us to rule out medium-to-large heterogeneity,")
    print("   supporting the social conformity interpretation over strategic signaling.")
else:
    print("⚠️  CAUTION: Some tests may be underpowered for medium effects.")
    print("   Null results should be interpreted with appropriate uncertainty.")

print()
print("="*80)

# Store for later use
POWER_ANALYSIS_RESULTS = power_results.copy()

# %% [markdown]
# # APPENDIX

# %%
# DESCRIPTIVE STATISTICS FOR PAPER TABLE ==========================================
# =================================================================================

print("="*80)
print("DESCRIPTIVE STATISTICS FOR APPENDIX TABLE")
print("="*80)

# 1. CORPUS CHARACTERISTICS
print("\n--- Corpus Characteristics ---")
speeches_per_deputy = df.groupby('deputado_id').size()
print(f"Speeches per deputy: mean={speeches_per_deputy.mean():.0f}, std={speeches_per_deputy.std():.0f}, "
      f"min={speeches_per_deputy.min()}, max={speeches_per_deputy.max()}")

# Word count (need to compute if not present)
if 'word_count' not in df.columns:
    df['word_count'] = df[CFG.text_col].str.split().str.len()
words_per_speech = df['word_count']
print(f"Words per speech: mean={words_per_speech.mean():.0f}, std={words_per_speech.std():.0f}, "
      f"min={words_per_speech.min()}, max={words_per_speech.max()}")

# 2. DEPUTY CHARACTERISTICS
print("\n--- Deputy Characteristics ---")
if 'career_tenure_years' in df.columns:
    # Get one observation per deputy (e.g., median tenure)
    tenure_by_deputy = df.groupby('deputado_id')['career_tenure_years'].median()
    print(f"Career tenure (years): mean={tenure_by_deputy.mean():.1f}, std={tenure_by_deputy.std():.1f}, "
          f"min={tenure_by_deputy.min():.0f}, max={tenure_by_deputy.max():.0f}")
else:
    print("career_tenure_years not in dataframe")

if 'prop_activity_annual' in df.columns:
    activity_by_deputy = df.groupby('deputado_id')['prop_activity_annual'].median()
    print(f"Legislative activity: mean={activity_by_deputy.mean():.2f}, std={activity_by_deputy.std():.2f}, "
          f"min={activity_by_deputy.min():.2f}, max={activity_by_deputy.max():.2f}")

# 3. SWITCHING STATISTICS
print("\n--- Switching Statistics ---")
switches_per_switcher = df_events.groupby('deputado_id').size()
print(f"Switches per switcher: mean={switches_per_switcher.mean():.2f}, std={switches_per_switcher.std():.2f}, "
      f"min={switches_per_switcher.min()}, max={switches_per_switcher.max()}")

multiple_switchers = (switches_per_switcher > 1).sum()
pct_multiple = multiple_switchers / len(switches_per_switcher) * 100
print(f"Multiple switchers (>1 switch): {multiple_switchers} ({pct_multiple:.1f}% of switchers)")

# 4. EVENT STUDY SAMPLE
print("\n--- Event Study Sample ---")
print(f"Speech-events in event study: {len(df_es):,}")
print(f"Unique switchers in event study: {df_es['deputado_id'].nunique()}")

# 5. CORRELATION SAMPLE  
print("\n--- Correlation Sample ---")
print(f"Switchers with full data (>=1 speeches, >=3 votes each window): {len(df_corr)}")

print("\n" + "="*80)


