import pandas as pd
import numpy as np
import os
import json
import joblib
import base64
from io import BytesIO
from tqdm import tqdm

from rdkit import Chem
from rdkit.Chem import Descriptors
import shap

import optuna
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from xgboost import XGBRegressor

import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import MaxNLocator

# ==========================================
# CONFIGURATION & SETUP
# ==========================================
INPUT_FILE = 'cleaned_solubility.csv'
OUTPUT_TRAIN = 'train_data_final.csv'
OUTPUT_TEST = 'test_data_final.csv'
SCALER_PATH = 'scaler_final.joblib'
MODEL_PATH = 'trained_model.joblib'
SUMMARY_FILE = 'model_summary.txt'
HTML_OUT = 'rdkit_custom_report.html'
PLOTS_DIR = 'plots'

REMOVED_FEATS_FILE = 'removed_features.csv'
REMOVED_MOLS_FILE = 'removed_molecules.csv'

if not os.path.exists(PLOTS_DIR):
    os.makedirs(PLOTS_DIR)

TARGET_ATOMS = ['C', 'H', 'N', 'O', 'S', 'Cl', 'F', 'P', 'I', 'Br', 'Fe', 'Co', 
                'Pt', 'Na', 'Ca', 'Mg', 'B', 'K', 'Al', 'As', 'Au', 'Li', 'Ga', 'Se', 'Si']

RDKIT_TARGETS = ['MolWt', 'HeavyAtomCount', 'NumHeteroatoms', 'NOCount', 
                 'NumAromaticRings', 'MolLogP', 'NumHAcceptors', 'NumHDonors']

N_TRIALS = 150
BOOTSTRAP_N = 1000
optuna.logging.set_verbosity(optuna.logging.WARNING)

sns.set_theme(style="whitegrid")

# ==========================================
# 1. FEATURISATION FUNCTIONS
# ==========================================
def calculate_atom_counts(mol):
    if mol is None: return None
    counts = {el: 0 for el in TARGET_ATOMS}
    counts['C_Aliphatic'] = 0
    counts['C_Aromatic'] = 0
    counts['O_Aromatic'] = 0
    
    mol_h = Chem.AddHs(mol)
    for atom in mol_h.GetAtoms():
        sym = atom.GetSymbol()
        if sym in counts: counts[sym] += 1
        if sym == 'C':
            if atom.GetIsAromatic(): counts['C_Aromatic'] += 1
            else: counts['C_Aliphatic'] += 1
        elif sym == 'O' and atom.GetIsAromatic(): 
            counts['O_Aromatic'] += 1
    return counts

def calculate_bond_counts(mol):
    if mol is None: return None
    counts = {
        'single_bonds': 0,
        'double_bonds': 0,
        'triple_bonds': 0,
        'aromatic_bonds': 0 
    }
    for bond in mol.GetBonds():
        bond_type = bond.GetBondType()
        if bond_type == Chem.rdchem.BondType.SINGLE: counts['single_bonds'] += 1
        elif bond_type == Chem.rdchem.BondType.DOUBLE: counts['double_bonds'] += 1
        elif bond_type == Chem.rdchem.BondType.TRIPLE: counts['triple_bonds'] += 1
        elif bond_type == Chem.rdchem.BondType.AROMATIC: counts['aromatic_bonds'] += 1
    return counts

def calculate_ring_sizes(mol):
    if mol is None: return None
    counts = {f'ring_size_{i}': 0 for i in range(3, 10)}
    ri = mol.GetRingInfo()
    for ring in ri.AtomRings():
        sz = len(ring)
        if 3 <= sz <= 9: counts[f'ring_size_{sz}'] += 1
    return counts

# --- Added Polar Bond Function ---
def calculate_polar_bond_counts(mol):
    if mol is None: return None
    mol_h = Chem.AddHs(mol)
    counts = {'OH_bonds': 0, 'NH_bonds': 0, 'SH_bonds': 0, 'CO_bonds': 0, 'CN_bonds': 0, 'C_Halogen_bonds': 0}
    halogens = {9, 17, 35, 53}  
    for bond in mol_h.GetBonds():
        a1, a2 = bond.GetBeginAtom().GetAtomicNum(), bond.GetEndAtom().GetAtomicNum()
        nums = tuple(sorted([a1, a2]))
        if nums == (1, 8): counts['OH_bonds'] += 1
        elif nums == (1, 7): counts['NH_bonds'] += 1
        elif nums == (1, 16): counts['SH_bonds'] += 1
        elif nums == (6, 8): counts['CO_bonds'] += 1
        elif nums == (6, 7): counts['CN_bonds'] += 1
        elif nums[0] == 6 and nums[1] in halogens: counts['C_Halogen_bonds'] += 1
    return counts

# ==========================================
# 2. UTILITY FUNCTIONS
# ==========================================
def save_and_get_b64(fig, filename):
    filepath = os.path.join(PLOTS_DIR, filename)
    fig.savefig(filepath, format='png', bbox_inches='tight', dpi=150)
    buf = BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', dpi=150)
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode('utf-8')

def calc_metrics(y_true, y_pred):
    m = {'R2': [], 'RMSE': [], 'MAE': [], 'Pct_07': [], 'Pct_10': []}
    n = len(y_true)
    for _ in tqdm(range(BOOTSTRAP_N), desc="Bootstrapping Metrics"):
        idx = np.random.randint(0, n, n)
        yt, yp = y_true[idx], y_pred[idx]
        m['R2'].append(r2_score(yt, yp))
        m['RMSE'].append(np.sqrt(mean_squared_error(yt, yp)))
        m['MAE'].append(mean_absolute_error(yt, yp))
        m['Pct_07'].append(np.mean(np.abs(yt - yp) <= 0.7) * 100)
        m['Pct_10'].append(np.mean(np.abs(yt - yp) <= 1.0) * 100)
    return {k: {'mean': np.mean(v), 'lo': np.percentile(v, 2.5), 'hi': np.percentile(v, 97.5)} for k, v in m.items()}

# ==========================================
# 3. MAIN WORKFLOW
# ==========================================
def main():
    if not os.path.exists(INPUT_FILE):
        print(f"Error: {INPUT_FILE} not found.")
        return

    print("--- Starting Featurisation ---")
    df_raw = pd.read_csv(INPUT_FILE).dropna(subset=['SMILES', 'LogS_Median']).drop_duplicates('SMILES')
    
    results, valid_smiles, failed_mols = [], [], []
    rdkit_names = [d[0] for d in Descriptors.descList]
    
    for sm in tqdm(df_raw['SMILES'], desc="Calculating Descriptors"):
        mol = Chem.MolFromSmiles(sm)
        if mol:
            try:
                rdkit_vals = [desc_func(mol) for _, desc_func in Descriptors.descList]
                atom_counts = calculate_atom_counts(mol)
                bond_counts = calculate_bond_counts(mol)
                ring_counts = calculate_ring_sizes(mol)
                
                # --- Extract polar bond counts ---
                polar_bond_counts = calculate_polar_bond_counts(mol)
                
                if atom_counts and bond_counts and ring_counts and polar_bond_counts:
                    row = rdkit_vals + list(atom_counts.values()) + list(bond_counts.values()) + list(ring_counts.values()) + list(polar_bond_counts.values())
                    results.append(row)
                    valid_smiles.append(sm)
            except Exception as e:
                failed_mols.append({'SMILES': sm, 'Reason': f"Error: {str(e)}"})
        else:
            failed_mols.append({'SMILES': sm, 'Reason': "Invalid SMILES"})

    dummy_mol = Chem.MolFromSmiles('C1CCCCC1') 
    atom_names = list(calculate_atom_counts(dummy_mol).keys())
    bond_names = list(calculate_bond_counts(dummy_mol).keys())
    ring_names = list(calculate_ring_sizes(dummy_mol).keys())
    
    # --- Register polar bond names ---
    polar_bond_names = list(calculate_polar_bond_counts(dummy_mol).keys())
    
    custom_names = atom_names + bond_names + ring_names + polar_bond_names
    all_col_names = rdkit_names + custom_names
    
    desc_df = pd.DataFrame(results, columns=all_col_names)
    desc_df['SMILES'] = valid_smiles
    df_merged = pd.merge(df_raw[['SMILES', 'LogS_Median']], desc_df, on='SMILES')
    
    removed_features_log = []
    removed_mols_log = pd.DataFrame(failed_mols)

    # ---------------------------------------------------------
    # 1. MANUAL DROPS
    # ---------------------------------------------------------
    if 'Ipc' in df_merged.columns: 
        df_merged.drop(columns=['Ipc'], inplace=True)
        removed_features_log.append({'Feature': 'Ipc', 'Reason': 'Manual Removal'})

    # ---------------------------------------------------------
    # 2. NaN HANDLING
    # ---------------------------------------------------------
    nan_threshold = 0.10 * len(df_merged)
    nan_counts = df_merged.isna().sum()
    nan_cols = nan_counts[nan_counts > nan_threshold].index.tolist()
    
    nan_cols = [c for c in nan_cols if c not in ['SMILES', 'LogS_Median']]
    for col in nan_cols:
        removed_features_log.append({'Feature': col, 'Reason': f'NaN threshold ({nan_counts[col]} NaNs)'})
    
    df_merged.drop(columns=nan_cols, inplace=True)
    
    nan_rows_mask = df_merged.isna().any(axis=1)
    nan_mols = df_merged[nan_rows_mask][['SMILES']].copy()
    nan_mols['Reason'] = 'Contained NaN in final set'
    if not nan_mols.empty:
        removed_mols_log = pd.concat([removed_mols_log, nan_mols], ignore_index=True)
    
    df_merged = df_merged[~nan_rows_mask]

    # ---------------------------------------------------------
    # 3. ZERO VARIANCE REMOVAL
    # ---------------------------------------------------------
    feature_cols = [c for c in df_merged.columns if c not in ['SMILES', 'LogS_Median']]
    variance = df_merged[feature_cols].var()
    zero_var_cols = variance[variance == 0].index.tolist()
    
    for col in zero_var_cols:
        removed_features_log.append({'Feature': col, 'Reason': 'Zero Variance'})
        
    df_merged.drop(columns=zero_var_cols, inplace=True)

    # ---------------------------------------------------------
    # 4. LOW FREQUENCY REMOVAL
    # ---------------------------------------------------------
    feature_cols = [c for c in df_merged.columns if c not in ['SMILES', 'LogS_Median']]
    low_freq_cols = []
    
    for col in feature_cols:
        non_zero_count = (df_merged[col] != 0).sum()
        if non_zero_count <= 5:
            low_freq_cols.append(col)
            removed_features_log.append({'Feature': col, 'Reason': f'Low Frequency ({non_zero_count} non-zero)'})
            
    df_merged.drop(columns=low_freq_cols, inplace=True)

    dropped_custom = [c for c in zero_var_cols + low_freq_cols if c in custom_names]

    pd.DataFrame(removed_features_log).to_csv(REMOVED_FEATS_FILE, index=False)
    removed_mols_log.to_csv(REMOVED_MOLS_FILE, index=False)
    
    # Split
    train_df, test_df = train_test_split(df_merged, test_size=0.2, random_state=42)
    train_df.to_csv(OUTPUT_TRAIN, index=False)
    test_df.to_csv(OUTPUT_TEST, index=False)
    
    features = [c for c in train_df.columns if c not in ['SMILES', 'LogS_Median']]
    X_train_raw, Y_train = train_df[features].values, train_df['LogS_Median'].values
    X_test_raw, Y_test = test_df[features].values, test_df['LogS_Median'].values

    # --- Track surviving polar bonds ---
    surviving_atoms = [f for f in features if f in atom_names]
    surviving_bonds = [f for f in features if f in bond_names]
    surviving_rings = [f for f in features if f in ring_names]
    surviving_polar_bonds = [f for f in features if f in polar_bond_names]
    surviving_custom = surviving_atoms + surviving_bonds + surviving_rings + surviving_polar_bonds

    # --- B. Optuna CV Training ---
    print("\n--- Starting Optuna Optimization ---")
    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 1500),
            'max_depth': trial.suggest_int('max_depth', 3, 15),
            'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.3, log=True),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.1, 1.0),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 20),
            'gamma': trial.suggest_float('gamma', 0.0, 5.0),
            'n_jobs': -1, 
            'random_state': 42
        }
        
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        scores = []
        for t_idx, v_idx in kf.split(X_train_raw):
            X_fold_train, X_fold_valid = X_train_raw[t_idx], X_train_raw[v_idx]
            y_fold_train, y_fold_valid = Y_train[t_idx], Y_train[v_idx]
            
            scaler = StandardScaler()
            X_fold_train_sc = scaler.fit_transform(X_fold_train)
            X_fold_valid_sc = scaler.transform(X_fold_valid)
            
            m = XGBRegressor(**params)
            m.fit(X_fold_train_sc, y_fold_train)
            preds = m.predict(X_fold_valid_sc)
            scores.append(np.mean(np.abs(y_fold_valid - preds) <= 0.7) * 100)
            
        return np.mean(scores)

    study = optuna.create_study(direction='maximize')
    with tqdm(total=N_TRIALS, desc="Optuna Trials") as pbar:
        def callback(study, trial): pbar.update(1)
        study.optimize(objective, n_trials=N_TRIALS, callbacks=[callback])
    
    # --- C. Final Model Fit ---
    print("\n--- Training Final Model ---")
    final_scaler = StandardScaler()
    X_train_sc = final_scaler.fit_transform(X_train_raw)
    X_test_sc = final_scaler.transform(X_test_raw)
    
    model = XGBRegressor(**study.best_params, n_jobs=-1, random_state=42)
    model.fit(X_train_sc, Y_train)
    preds = model.predict(X_test_sc)
    
    joblib.dump(final_scaler, SCALER_PATH)
    joblib.dump(model, MODEL_PATH)
    stats = calc_metrics(Y_test, preds)

    with open(SUMMARY_FILE, 'w') as f:
        f.write("=== Model Performance Summary ===\n")
        f.write(f"Dropped Custom Features (Zero Variance / Low Freq): {', '.join(dropped_custom) if dropped_custom else 'None'}\n\n")
        f.write("Best Hyperparameters:\n" + json.dumps(study.best_params, indent=4) + "\n\n")
        for metric, vals in stats.items():
            f.write(f"{metric}: {vals['mean']:.3f} (95% CI: {vals['lo']:.3f} - {vals['hi']:.3f})\n")

    # ==========================================
    # 4. PLOTTING
    # ==========================================
    print("\n--- Generating Plots ---")
    
    explainer = shap.TreeExplainer(model)
    shap_vals = explainer.shap_values(X_test_sc)
    
    def save_shap_subset(target_features, filename):
        if not target_features: return ""
        indices = [features.index(f) for f in target_features]
        fig = plt.figure(figsize=(10, 6))
        shap.summary_plot(shap_vals[:, indices], X_test_sc[:, indices], feature_names=target_features, show=False)
        plt.title("")
        return save_and_get_b64(fig, filename)

    b64_shap_all = save_shap_subset(surviving_custom, "shap_all_custom.png")
    b64_shap_bonds = save_shap_subset(surviving_bonds, "shap_bonds_only.png")
    b64_shap_rings = save_shap_subset(surviving_rings, "shap_rings_only.png")
    
    # --- Specific SHAP plot for Polar Bonds ---
    b64_shap_polar_bonds = save_shap_subset(surviving_polar_bonds, "shap_polar_bonds_only.png")

    # Standard Histograms (Bonds & Rings)
    hist_features = surviving_bonds + surviving_rings
    num_plots = len(hist_features)
    cols = 5
    rows = (num_plots // cols) + (1 if num_plots % cols != 0 else 0)
    
    if num_plots > 0:
        fig_hist = plt.figure(figsize=(15, 3 * rows)) 
        for i, feat in enumerate(hist_features):
            ax = fig_hist.add_subplot(rows, cols, i+1)
            data = train_df[feat]
            bins = np.arange(data.min() - 0.5, data.max() + 1.5, 1) 
            ax.hist(data, bins=bins, color="skyblue", edgecolor='black')
            ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=6))
            ax.set_xlabel(feat)
            ax.set_title("")
        plt.tight_layout()
        b64_hist = save_and_get_b64(fig_hist, "bond_ring_histograms.png")
    else:
        b64_hist = ""

    # --- Specific Histograms for Polar Bonds ---
    num_plots_polar = len(surviving_polar_bonds)
    cols_polar = 3 # Smaller grid for polar bonds
    rows_polar = (num_plots_polar // cols_polar) + (1 if num_plots_polar % cols_polar != 0 else 0)
    
    if num_plots_polar > 0:
        fig_hist_polar = plt.figure(figsize=(12, 3 * rows_polar)) 
        for i, feat in enumerate(surviving_polar_bonds):
            ax = fig_hist_polar.add_subplot(rows_polar, cols_polar, i+1)
            data = train_df[feat]
            bins = np.arange(data.min() - 0.5, data.max() + 1.5, 1) 
            ax.hist(data, bins=bins, color="lightcoral", edgecolor='black') # Different color for distinction
            ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=6))
            ax.set_xlabel(feat)
            ax.set_title("")
        plt.tight_layout()
        b64_hist_polar = save_and_get_b64(fig_hist_polar, "polar_bond_histograms.png")
    else:
        b64_hist_polar = ""

    # Correlation to LogS
    def plot_logs_corr(feats, filename):
        if not feats: return ""
        corr = train_df[feats].corrwith(train_df['LogS_Median']).sort_values()
        fig = plt.figure(figsize=(8, max(4, len(feats)*0.3)))
        sns.barplot(x=corr.values, y=corr.index, palette='coolwarm')
        plt.xlabel("Correlation with LogS")
        plt.title("")
        return save_and_get_b64(fig, filename)

    b64_corr_no_atoms = plot_logs_corr(surviving_bonds + surviving_rings, "corr_logs_no_atoms.png")
    b64_corr_with_atoms = plot_logs_corr(surviving_custom, "corr_logs_with_atoms.png")

    # Heatmap Matrix
    valid_rdkit = [c for c in RDKIT_TARGETS if c in train_df.columns]
    corr_matrix = train_df[surviving_custom + valid_rdkit].corr().loc[surviving_custom, valid_rdkit]
    
    fig_corr = plt.figure(figsize=(12, max(8, len(surviving_custom)*0.4)))
    sns.heatmap(corr_matrix, cmap='coolwarm', center=0, annot=True, fmt=".2f",
                square=False, linewidths=.5, cbar_kws={"shrink": .5}, annot_kws={"size": 10})
    plt.title("")
    b64_corr_mat = save_and_get_b64(fig_corr, "correlation_heatmap.png")

    # Performance Graph
    fig_perf = plt.figure(figsize=(7, 6))
    sns.scatterplot(x=Y_test, y=preds, alpha=0.6, color='blue')
    plt.plot([Y_test.min(), Y_test.max()], [Y_test.min(), Y_test.max()], 'r--')
    plt.xlabel("Actual LogS")
    plt.ylabel("Predicted LogS")
    plt.title("")
    b64_perf = save_and_get_b64(fig_perf, "performance_scatter.png")

    # --- D. HTML Report ---
    # Updated HTML string to include Polar Bond SHAP and Histograms
    print("\n--- Generating HTML Report ---")
    html = f"""
    <!DOCTYPE html>
    <html><head>
    <meta charset="utf-8">
    <style>
        body{{font-family: Arial, sans-serif; margin: 40px; color: #333;}} 
        table{{border-collapse: collapse; margin-bottom: 20px; width: 600px;}} 
        th, td{{padding: 10px; border: 1px solid #ddd; text-align: left;}} 
        th{{background: #f2f2f2;}} 
        img{{max-width: 100%; height: auto; margin-bottom: 30px; display: block;}}
        .section-title {{border-bottom: 2px solid #333; padding-bottom: 5px;}}
    </style>
    </head><body>
    <h1>Model Performance & Feature Report</h1>
    
    <h2 class="section-title">1. Test Set Performance Metrics</h2>
    <table>
        <tr><th>Metric</th><th>Mean [95% CI]</th></tr>
        <tr><td>R&sup2; Score</td><td>{stats['R2']['mean']:.3f} [{stats['R2']['lo']:.3f} - {stats['R2']['hi']:.3f}]</td></tr>
        <tr><td>RMSE</td><td>{stats['RMSE']['mean']:.3f} [{stats['RMSE']['lo']:.3f} - {stats['RMSE']['hi']:.3f}]</td></tr>
        <tr><td>MAE</td><td>{stats['MAE']['mean']:.3f} [{stats['MAE']['lo']:.3f} - {stats['MAE']['hi']:.3f}]</td></tr>
        <tr><td>% within 0.7 log units</td><td>{stats['Pct_07']['mean']:.1f}% [{stats['Pct_07']['lo']:.1f} - {stats['Pct_07']['hi']:.1f}]</td></tr>
        <tr><td>% within 1.0 log units</td><td>{stats['Pct_10']['mean']:.1f}% [{stats['Pct_10']['lo']:.1f} - {stats['Pct_10']['hi']:.1f}]</td></tr>
    </table>

    <h2 class="section-title">2. Model Performance (Actual vs Predicted)</h2>
    <img src="data:image/png;base64,{b64_perf}">

    <h2 class="section-title">3. SHAP Analysis</h2>
    <h3>All Custom Features (Includes Polar Bonds)</h3>
    <img src="data:image/png;base64,{b64_shap_all}">
    <h3>Bonds Only (Structural)</h3>
    <img src="data:image/png;base64,{b64_shap_bonds}">
    <h3>Specific Polar Bonds Only</h3>
    <img src="data:image/png;base64,{b64_shap_polar_bonds}">
    <h3>Rings Only</h3>
    <img src="data:image/png;base64,{b64_shap_rings}">

    <h2 class="section-title">4. Bond & Ring Distributions</h2>
    <p><strong>Dropped Custom Features:</strong> {', '.join(dropped_custom) if dropped_custom else 'None'}</p>
    
    <h3>Structural Bonds & Rings</h3>
    <img src="data:image/png;base64,{b64_hist}">
    
    <h3>Polar Bonds Distributions</h3>
    <img src="data:image/png;base64,{b64_hist_polar}">

    <h2 class="section-title">5. LogS Correlation</h2>
    <h3>Bonds & Rings (No Atoms)</h3>
    <img src="data:image/png;base64,{b64_corr_no_atoms}">
    <h3>All Custom Features</h3>
    <img src="data:image/png;base64,{b64_corr_with_atoms}">

    <br>
    <h2 class="section-title">6. Correlation Heatmap</h2>
    <img src="data:image/png;base64,{b64_corr_mat}">

    </body></html>
    """
    with open(HTML_OUT, "w", encoding="utf-8") as f: 
        f.write(html)
        
    print(f"Done! Report saved to {HTML_OUT}")
    print(f"All standalone images saved in the '{PLOTS_DIR}/' folder.")

if __name__ == "__main__":
    main()