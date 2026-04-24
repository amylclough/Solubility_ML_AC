import pandas as pd
import numpy as np
import os
import joblib
import time
import optuna
import matplotlib.pyplot as plt
import base64
from io import BytesIO
from tqdm import tqdm
from rdkit import Chem
from rdkit.Chem import Descriptors, rdFingerprintGenerator
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# --- Configuration ---
INPUT_FILE = 'cleaned_solubility.csv'
REMOVED_FEATS_FILE = 'removed_features.csv'
REMOVED_MOLS_FILE = 'removed_molecules.csv'
N_TRIALS = 150
N_BITS = 2048
BOOTSTRAP_N = 1000

optuna.logging.set_verbosity(optuna.logging.WARNING)

# --- Featurization Functions ---
def calculate_descriptors(smiles_list):
    """Calculates all RDKit 2D descriptors."""
    desc_names = [d[0] for d in Descriptors.descList]
    results, valid_smiles, failed_mols = [], [], []
    
    print(f"Generating RDKit 2D descriptors for {len(smiles_list)} molecules...")
    for sm in tqdm(smiles_list):
        mol = Chem.MolFromSmiles(sm)
        if mol:
            try:
                row = [desc_func(mol) for _, desc_func in Descriptors.descList]
                results.append(row)
                valid_smiles.append(sm)
            except Exception as e:
                failed_mols.append({'SMILES': sm, 'Reason': f"RDKit Error: {str(e)}"})
        else:
            failed_mols.append({'SMILES': sm, 'Reason': "Invalid SMILES"})
            
    return pd.DataFrame(results, columns=desc_names), valid_smiles, failed_mols

def generate_mf(smiles_list, radius):
    """Generates Morgan Fingerprints."""
    mf_gen = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=N_BITS)
    fps = []
    for sm in smiles_list:
        mol = Chem.MolFromSmiles(sm)
        fps.append(np.array(mf_gen.GetFingerprintAsNumPy(mol)))
    return np.array(fps)

# --- Training & Evaluation Functions ---
def calculate_metrics_with_ci(y_true, y_pred):
    """Calculates metrics with 95% CI via bootstrapping."""
    metrics = {'R2': [], 'RMSE': [], 'MAE': [], 'Pct_07': [], 'Pct_10': []}
    n = len(y_true)
    for _ in range(BOOTSTRAP_N):
        idx = np.random.randint(0, n, n)
        yt, yp = y_true[idx], y_pred[idx]
        metrics['R2'].append(r2_score(yt, yp))
        metrics['RMSE'].append(np.sqrt(mean_squared_error(yt, yp)))
        metrics['MAE'].append(mean_absolute_error(yt, yp))
        metrics['Pct_07'].append(np.mean(np.abs(yt - yp) <= 0.7) * 100)
        metrics['Pct_10'].append(np.mean(np.abs(yt - yp) <= 1.0) * 100)
    
    return {k: {'mean': np.mean(v), 'lower': np.percentile(v, 2.5), 'upper': np.percentile(v, 97.5)} 
            for k, v in metrics.items()}

def create_corr_plot_base64(y_true, y_pred, show_ci=False):
    """Generates a correlation plot (no title, no grid) and returns base64 string."""
    plt.figure(figsize=(5, 5))
    plt.scatter(y_true, y_pred, alpha=0.4, s=20, color='#1f77b4')
    
    # Unity Line
    min_val = min(min(y_true), min(y_pred))
    max_val = max(max(y_true), max(y_pred))
    plt.plot([min_val, max_val], [min_val, max_val], 'k--', lw=1)
    
    # Error thresholds
    if show_ci:
        plt.fill_between([min_val, max_val], [min_val-0.7, max_val-0.7], [min_val+0.7, max_val+0.7], 
                         color='red', alpha=0.1, label='+/- 0.7')
    
    plt.xlabel('Experimental LogS_Median')
    plt.ylabel('Predicted LogS_Median')
    plt.grid(False) # Ensure no gridlines
    # No title requested
    
    buf = BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    plt.close()
    return base64.b64encode(buf.getvalue()).decode('utf-8')

def objective(trial, X_train, Y_train):
    """Optuna objective function using the unified, broader hyperparameter ranges."""
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 1500),
        'max_depth': trial.suggest_int('max_depth', 3, 15),
        'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.3, log=True),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.1, 1.0),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 20),
        'gamma': trial.suggest_float('gamma', 0.0, 5.0),
        'n_jobs': -1, 'random_state': 42
    }
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    scores = []
    
    # K-Fold Cross Validation without early stopping
    for t_idx, v_idx in kf.split(X_train):
        m = XGBRegressor(**params)
        m.fit(X_train[t_idx], Y_train[t_idx])
        preds = m.predict(X_train[v_idx])
        scores.append(np.mean(np.abs(Y_train[v_idx] - preds) <= 0.7) * 100)
    return np.mean(scores)

# --- Main Pipeline ---
def main():
    # ==========================================
    # PART 1: DATA LOADING & FEATURISATION
    # ==========================================
    if not os.path.exists(INPUT_FILE):
        print(f"Error: {INPUT_FILE} not found.")
        return

    df_raw = pd.read_csv(INPUT_FILE).dropna(subset=['SMILES', 'LogS_Median']).drop_duplicates('SMILES')
    desc_df, valid_smiles, failed_mols = calculate_descriptors(df_raw['SMILES'])
    desc_df['SMILES'] = valid_smiles
    df_merged = pd.merge(df_raw[['SMILES', 'LogS_Median']], desc_df, on='SMILES')
    
    removed_features_log, removed_mols_log = [], pd.DataFrame(failed_mols)

    # Clean Columns
    if 'Ipc' in df_merged.columns:
        df_merged.drop(columns=['Ipc'], inplace=True)
        removed_features_log.append({'Feature': 'Ipc', 'Reason': 'Manual Removal (Overflow Risk)'})
    
    nan_threshold = 0.10 * len(df_merged)
    nan_counts = df_merged.isna().sum()
    nan_cols = nan_counts[nan_counts > nan_threshold].index.tolist()
    
    for col in nan_cols:
        if col not in ['SMILES', 'LogS_Median']:
            removed_features_log.append({'Feature': col, 'Reason': f'Exceeded limit ({nan_counts[col]} NaNs)'})
    df_merged.drop(columns=nan_cols, inplace=True)
    
    # Clean Rows
    nan_rows_mask = df_merged.isna().any(axis=1)
    nan_mols = df_merged[nan_rows_mask][['SMILES']].copy()
    nan_mols['Reason'] = 'Contained NaN in final descriptor set'
    removed_mols_log = pd.concat([removed_mols_log, nan_mols], ignore_index=True)
    df_merged = df_merged[~nan_rows_mask]
    
    pd.DataFrame(removed_features_log).to_csv(REMOVED_FEATS_FILE, index=False)
    removed_mols_log.to_csv(REMOVED_MOLS_FILE, index=False)

    # Train/Test Split & Scaling
    train_df, test_df = train_test_split(df_merged, test_size=0.2, random_state=42)
    feat_cols = [c for c in train_df.columns if c not in ['SMILES', 'LogS_Median']]
    
    scaler = StandardScaler()
    X_train_rdkit = scaler.fit_transform(train_df[feat_cols])
    X_test_rdkit = scaler.transform(test_df[feat_cols])
    
    Y_train, Y_test = train_df['LogS_Median'].values, test_df['LogS_Median'].values
    print(f"\nFeaturization Complete: {len(train_df)} Train, {len(test_df)} Test molecules.")

    # ==========================================
    # PART 2: TRAINING & COMBINATIONS
    # ==========================================
    master_predictions = test_df[['SMILES', 'LogS_Median']].copy()
    master_params, results_for_html = [], []
    
    experiments = [
        ('Just RDKit', 0, True, False),
        ('Just MF (R2)', 2, False, True),
        ('RDKit + MF (R1)', 1, True, True),
        ('RDKit + MF (R2)', 2, True, True),
        ('RDKit + MF (R3)', 3, True, True)
    ]

    for name, radius, use_rdkit, use_mf in experiments:
        print(f"\nRunning Experiment: {name}")
        start_time = time.time()
        
        X_tr_list, X_te_list = [], []
        if use_rdkit:
            X_tr_list.append(X_train_rdkit)
            X_te_list.append(X_test_rdkit)
        if use_mf:
            print(f"Generating Morgan Fingerprints (Radius {radius})...")
            X_tr_list.append(generate_mf(train_df['SMILES'], radius))
            X_te_list.append(generate_mf(test_df['SMILES'], radius))
            
        X_tr, X_te = np.concatenate(X_tr_list, axis=1), np.concatenate(X_te_list, axis=1)
        
        # Optimize & Train
        study = optuna.create_study(direction='maximize')
        study.optimize(lambda t: objective(t, X_tr, Y_train), n_trials=N_TRIALS)
        
        best_m = XGBRegressor(**study.best_params, n_jobs=-1, random_state=42)
        best_m.fit(X_tr, Y_train)
        
        duration = (time.time() - start_time) / 60
        preds = best_m.predict(X_te)
        
        # Data Logging
        master_predictions[f'Pred_{name}'] = preds
        master_params.append({'Experiment': name, **study.best_params})
        
        # Two graphs per experiment (With and Without CI bounds)
        results_for_html.append({
            'name': name, 
            'time': duration, 
            'metrics': calculate_metrics_with_ci(Y_test, preds),
            'img_ci': create_corr_plot_base64(Y_test, preds, show_ci=True),
            'img_no_ci': create_corr_plot_base64(Y_test, preds, show_ci=False)
        })

    # Save outputs
    master_predictions.to_csv('test_set_predictions.csv', index=False)
    pd.DataFrame(master_params).to_csv('best_params_comparison.csv', index=False)
    
    # ==========================================
    # PART 3: HTML REPORT GENERATION
    # ==========================================
    html = "<html><head><style>table{border-collapse: collapse;} th, td{padding: 10px; border: 1px solid black; text-align: center;}</style></head><body>"
    html += "<h1>Solubility Model Feature Comparison</h1>"
    html += "<table><tr><th>Experiment</th><th>Time (min)</th><th>Metrics (95% CI)</th><th>Plot (With CI bounds)</th><th>Plot (No bounds)</th></tr>"
    
    for r in results_for_html:
        m = r['metrics']
        metric_text = (f"R2: {m['R2']['mean']:.3f} [{m['R2']['lower']:.2f}-{m['R2']['upper']:.2f}]<br>"
                       f"RMSE: {m['RMSE']['mean']:.3f}<br>MAE: {m['MAE']['mean']:.3f}<br>"
                       f"% < 0.7: {m['Pct_07']['mean']:.1f}%<br>% < 1.0: {m['Pct_10']['mean']:.1f}%")
        
        html += f"<tr><td><b>{r['name']}</b></td><td>{r['time']:.2f}</td><td>{metric_text}</td>"
        html += f"<td><img src='data:image/png;base64,{r['img_ci']}' width='250'></td>"
        html += f"<td><img src='data:image/png;base64,{r['img_no_ci']}' width='250'></td></tr>"
    
    html += "</table></body></html>"
    with open("model_comparison_report.html", "w") as f: 
        f.write(html)
        
    print("\nProcess Complete! Reports generated: model_comparison_report.html, test_set_predictions.csv, best_params_comparison.csv")

if __name__ == "__main__":
    main()