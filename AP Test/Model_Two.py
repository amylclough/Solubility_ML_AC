import pandas as pd
import numpy as np
import os
import joblib
import time
import optuna
import matplotlib.pyplot as plt
import seaborn as sns
import base64
import shap
from io import BytesIO
from tqdm import tqdm
from rdkit import Chem
from rdkit.Chem import Descriptors
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from xgboost import XGBRegressor

# --- Configuration ---
INPUT_FILE = 'cleaned_solubility.csv'
OUTPUT_TRAIN = 'train_data_raw_with_ap.csv'
OUTPUT_TEST = 'test_data_raw_with_ap.csv'
SCALER_PATH = 'scaler_with_ap.joblib'
REMOVED_FEATS_FILE = 'removed_features.csv'
REMOVED_MOLS_FILE = 'removed_molecules.csv'
HTML_OUT = 'rdkit_ap_report.html'

# Added output files for predictions and best parameters
PREDICTIONS_OUT = 'rdkit_ap_predictions.csv'
PARAMS_OUT = 'rdkit_ap_best_params.csv'

N_TRIALS = 150
BOOTSTRAP_N = 1000

optuna.logging.set_verbosity(optuna.logging.WARNING)

# --- Featurisation Functions ---
def calculate_ap(mol):
    """Calculates Aromatic Proportion (AP) as used in ESOL."""
    if mol is None: return np.nan
    aromatic_atoms = sum(1 for at in mol.GetAtoms() if at.GetIsAromatic())
    heavy_atoms = mol.GetNumHeavyAtoms()
    return aromatic_atoms / heavy_atoms if heavy_atoms > 0 else 0

def calculate_descriptors(smiles_list):
    """Calculates all RDKit 2D descriptors + Aromatic Proportion."""
    desc_names = [d[0] for d in Descriptors.descList]
    all_col_names = desc_names + ['AromaticProportion']
    
    results = []
    valid_smiles = []
    failed_mols = []
    
    print(f"Generating descriptors for {len(smiles_list)} molecules...")
    for sm in tqdm(smiles_list, desc="Calculating Descriptors"):
        mol = Chem.MolFromSmiles(sm)
        if mol:
            try:
                row = [desc_func(mol) for _, desc_func in Descriptors.descList]
                row.append(calculate_ap(mol))
                results.append(row)
                valid_smiles.append(sm)
            except Exception as e:
                failed_mols.append({'SMILES': sm, 'Reason': f"RDKit Error: {str(e)}"})
        else:
            failed_mols.append({'SMILES': sm, 'Reason': "Invalid SMILES"})
            
    return pd.DataFrame(results, columns=all_col_names), valid_smiles, failed_mols

# --- Modeling Functions ---
def calculate_metrics_with_ci(y_true, y_pred):
    metrics = {'R2': [], 'RMSE': [], 'MAE': [], 'Pct_07': [], 'Pct_10': []}
    n = len(y_true)
    for _ in tqdm(range(BOOTSTRAP_N), desc="Bootstrapping Metrics"):
        idx = np.random.randint(0, n, n)
        yt, yp = y_true[idx], y_pred[idx]
        metrics['R2'].append(r2_score(yt, yp))
        metrics['RMSE'].append(np.sqrt(mean_squared_error(yt, yp)))
        metrics['MAE'].append(mean_absolute_error(yt, yp))
        metrics['Pct_07'].append(np.mean(np.abs(yt - yp) <= 0.7) * 100)
        metrics['Pct_10'].append(np.mean(np.abs(yt - yp) <= 1.0) * 100)
    
    return {k: {'mean': np.mean(v), 'lower': np.percentile(v, 2.5), 'upper': np.percentile(v, 97.5)} 
            for k, v in metrics.items()}

def get_base64_img():
    buf = BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    plt.close()
    return base64.b64encode(buf.getvalue()).decode('utf-8')

def objective(trial, X_train, Y_train):
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
    
    for t_idx, v_idx in kf.split(X_train):
        m = XGBRegressor(**params)
        m.fit(X_train[t_idx], Y_train[t_idx])
        preds = m.predict(X_train[v_idx])
        scores.append(np.mean(np.abs(Y_train[v_idx] - preds) <= 0.7) * 100)
    return np.mean(scores)

def main():
    # ==========================================
    # PHASE 1: FEATURISATION & PREPROCESSING
    # ==========================================
    print("\n--- Starting Featurisation Phase ---")
    if not os.path.exists(INPUT_FILE):
        raise FileNotFoundError(f"Error: {INPUT_FILE} not found. Please ensure it is in the directory.")

    df_raw = pd.read_csv(INPUT_FILE).dropna(subset=['SMILES', 'LogS_Median']).drop_duplicates('SMILES')
    
    desc_df, valid_smiles, failed_mols = calculate_descriptors(df_raw['SMILES'])
    desc_df['SMILES'] = valid_smiles
    df_merged = pd.merge(df_raw[['SMILES', 'LogS_Median']], desc_df, on='SMILES')
    
    removed_features_log = []
    removed_mols_log = pd.DataFrame(failed_mols)

    if 'Ipc' in df_merged.columns:
        df_merged.drop(columns=['Ipc'], inplace=True)
        removed_features_log.append({'Feature': 'Ipc', 'Reason': 'Manual Removal (Overflow Risk)'})
    
    nan_threshold = 0.10 * len(df_merged)
    nan_counts = df_merged.isna().sum()
    nan_cols = nan_counts[nan_counts > nan_threshold].index.tolist()
    
    for col in nan_cols:
        if col not in ['SMILES', 'LogS_Median']:
            removed_features_log.append({'Feature': col, 'Reason': f'NaN threshold ({nan_counts[col]})'})
            
    df_merged.drop(columns=nan_cols, inplace=True)
    
    nan_rows_mask = df_merged.isna().any(axis=1)
    nan_mols = df_merged[nan_rows_mask][['SMILES']].copy()
    nan_mols['Reason'] = 'Contained NaN in final set'
    removed_mols_log = pd.concat([removed_mols_log, nan_mols], ignore_index=True)
    df_merged = df_merged[~nan_rows_mask]
    
    pd.DataFrame(removed_features_log).to_csv(REMOVED_FEATS_FILE, index=False)
    removed_mols_log.to_csv(REMOVED_MOLS_FILE, index=False)

    train_df, test_df = train_test_split(df_merged, test_size=0.2, random_state=42)
    
    # Check scaling happens AFTER AP is added
    feat_cols = [c for c in train_df.columns if c not in ['SMILES', 'LogS_Median']]
    assert 'AromaticProportion' in feat_cols, "AromaticProportion is missing before scaling!"
    
    scaler = StandardScaler()
    scaler.fit(train_df[feat_cols])
    joblib.dump(scaler, SCALER_PATH)
    train_df.to_csv(OUTPUT_TRAIN, index=False)
    test_df.to_csv(OUTPUT_TEST, index=False)

    # ==========================================
    # PHASE 2: MODEL TRAINING & EVALUATION
    # ==========================================
    print("\n--- Starting Training Phase ---")
    X_train_scaled = scaler.transform(train_df[feat_cols])
    X_test_scaled = scaler.transform(test_df[feat_cols])
    Y_train = train_df['LogS_Median'].values
    Y_test = test_df['LogS_Median'].values
    
    print(f"Running {N_TRIALS} Optuna trials with 5-fold CV...")
    start_time = time.time()
    study = optuna.create_study(direction='maximize')
    
    with tqdm(total=N_TRIALS, desc="Hyperparameter Tuning") as pbar:
        def tqdm_callback(study, trial):
            pbar.update(1)
        study.optimize(lambda t: objective(t, X_train_scaled, Y_train), n_trials=N_TRIALS, callbacks=[tqdm_callback])
    
    print("Training final model on full train set...")
    best_m = XGBRegressor(**study.best_params, n_jobs=-1, random_state=42)
    best_m.fit(X_train_scaled, Y_train) 
    duration = (time.time() - start_time) / 60
    preds = best_m.predict(X_test_scaled)

    # --- Save Outputs (Predictions & Best Params) ---
    print("\n--- Saving Output Files ---")
    print(f"Saving predictions to {PREDICTIONS_OUT}...")
    predictions_df = pd.DataFrame({
        'Actual_LogS': Y_test,
        'Predicted_LogS': preds
    })
    predictions_df.to_csv(PREDICTIONS_OUT, index=False)
    
    print(f"Saving best parameters to {PARAMS_OUT}...")
    params_df = pd.DataFrame([study.best_params])
    params_df.to_csv(PARAMS_OUT, index=False)

    # ==========================================
    # PHASE 3: ANALYSIS & PLOTTING
    # ==========================================
    print("\n--- Generating Metrics and Plots ---")
    stats = calculate_metrics_with_ci(Y_test, preds)

    # 1. Performance Plot (No Title)
    plt.figure(figsize=(6,6))
    plt.scatter(Y_test, preds, alpha=0.5, s=20)
    plt.plot([Y_test.min(), Y_test.max()], [Y_test.min(), Y_test.max()], 'k--')
    plt.grid(True)
    plt.xlabel("True LogS")
    plt.ylabel("Predicted LogS")
    perf_img = get_base64_img()

    # 2. Correlation calculations and plots
    r2_ap_logs = np.corrcoef(train_df['AromaticProportion'], train_df['LogS_Median'])[0, 1]**2
    r2_numaro_logs = np.corrcoef(train_df['NumAromaticRings'], train_df['LogS_Median'])[0, 1]**2
    r2_ap_numaro = np.corrcoef(train_df['AromaticProportion'], train_df['NumAromaticRings'])[0, 1]**2

    # Graph: AP vs LogS (No Title)
    plt.figure(figsize=(6, 5))
    plt.scatter(train_df['AromaticProportion'], train_df['LogS_Median'], alpha=0.3, color='steelblue')
    plt.grid(True)
    plt.xlabel("Aromatic Proportion")
    plt.ylabel("LogS Median")
    ap_logs_img = get_base64_img()

    # Graph: NumAromaticRings vs LogS (No Title)
    plt.figure(figsize=(6, 5))
    plt.scatter(train_df['NumAromaticRings'], train_df['LogS_Median'], alpha=0.3, color='orange')
    plt.grid(True)
    plt.xlabel("NumAromaticRings")
    plt.ylabel("LogS Median")
    numaro_logs_img = get_base64_img()

    # Graph: AP vs NumAromaticRings (No Title)
    plt.figure(figsize=(6, 5))
    plt.scatter(train_df['AromaticProportion'], train_df['NumAromaticRings'], alpha=0.3, color='green')
    plt.grid(True)
    plt.xlabel("Aromatic Proportion")
    plt.ylabel("NumAromaticRings")
    ap_numaro_img = get_base64_img()

    # 3. SHAP Analysis
    print("Generating SHAP values...")
    explainer = shap.TreeExplainer(best_m)
    shap_values = explainer.shap_values(X_test_scaled)
    shap_importance = np.abs(shap_values).mean(0)
    
    feat_imp_df = pd.DataFrame({'Feature': feat_cols, 'Importance': shap_importance})
    feat_imp_df = feat_imp_df.sort_values('Importance', ascending=False).reset_index(drop=True)
    
    # Find Exact Ranks 
    ap_rank = feat_imp_df[feat_imp_df['Feature'] == 'AromaticProportion'].index[0] + 1
    numaro_rank = feat_imp_df[feat_imp_df['Feature'] == 'NumAromaticRings'].index[0] + 1

    top_n = feat_imp_df.head(20).copy()
    if 'AromaticProportion' not in top_n['Feature'].values:
        ap_row = feat_imp_df[feat_imp_df['Feature'] == 'AromaticProportion']
        top_n = pd.concat([top_n, ap_row])
    
    # Graph: SHAP (No Title)
    plt.figure(figsize=(10, 8))
    sns.barplot(x='Importance', y='Feature', data=top_n, palette='viridis')
    plt.grid(True, axis='x')
    shap_img = get_base64_img()

    # ==========================================
    # PHASE 4: HTML REPORT GENERATION
    # ==========================================
    html = f"""
    <html><head><style>
        body{{font-family: sans-serif; margin: 40px;}}
        table{{border-collapse: collapse; width: 700px; margin-bottom: 30px;}}
        th, td{{padding: 10px; border: 1px solid #ddd; text-align: left;}}
        th{{background: #f2f2f2;}}
        .img-container{{margin-bottom: 40px; display: block;}}
        h4 {{margin-bottom: 5px;}}
    </style></head><body>
    <h1>RDKit + Aromatic Proportion Full Pipeline Report</h1>
    <p><b>Hyperparameter Tuning Time:</b> {duration:.2f} mins (150 Trials)</p>
    
    <h2>1. Model Performance (Test Set)</h2>
    <table>
        <tr><th>Metric</th><th>Mean [95% CI]</th></tr>
        <tr><td>R&sup2;</td><td>{stats['R2']['mean']:.3f} [{stats['R2']['lower']:.3f}-{stats['R2']['upper']:.3f}]</td></tr>
        <tr><td>RMSE</td><td>{stats['RMSE']['mean']:.3f} [{stats['RMSE']['lower']:.3f}-{stats['RMSE']['upper']:.3f}]</td></tr>
        <tr><td>MAE</td><td>{stats['MAE']['mean']:.3f} [{stats['MAE']['lower']:.3f}-{stats['MAE']['upper']:.3f}]</td></tr>
        <tr><td>% Within 0.7</td><td>{stats['Pct_07']['mean']:.1f}% [{stats['Pct_07']['lower']:.1f}-{stats['Pct_07']['upper']:.1f}]</td></tr>
        <tr><td>% Within 1.0</td><td>{stats['Pct_10']['mean']:.1f}% [{stats['Pct_10']['lower']:.1f}-{stats['Pct_10']['upper']:.1f}]</td></tr>
    </table>
    
    <div class="img-container">
        <h4>Test Set Prediction vs True LogS (R&sup2;: {stats['R2']['mean']:.3f})</h4>
        <img src="data:image/png;base64,{perf_img}">
    </div>

    <h2>2. Aromaticity Analysis & R&sup2; Correlations (Train Set)</h2>
    <ul>
        <li><b>Aromatic Proportion vs LogS R&sup2;:</b> {r2_ap_logs:.4f}</li>
        <li><b>NumAromaticRings vs LogS R&sup2;:</b> {r2_numaro_logs:.4f}</li>
        <li><b>Aromatic Proportion vs NumAromaticRings R&sup2;:</b> {r2_ap_numaro:.4f}</li>
    </ul>
    
    <h3>Individual Plots</h3>
    <div class="img-container">
        <h4>Aromatic Proportion vs LogS (R&sup2; = {r2_ap_logs:.4f})</h4>
        <img src="data:image/png;base64,{ap_logs_img}"><br><br>
        
        <h4>NumAromaticRings vs LogS (R&sup2; = {r2_numaro_logs:.4f})</h4>
        <img src="data:image/png;base64,{numaro_logs_img}"><br><br>
        
        <h4>Aromatic Proportion vs NumAromaticRings (R&sup2; = {r2_ap_numaro:.4f})</h4>
        <img src="data:image/png;base64,{ap_numaro_img}">
    </div>

    <h2>3. Feature Importance (SHAP Ranking)</h2>
    <ul>
        <li><b>Aromatic Proportion</b> ranked <b>#{ap_rank}</b> in importance out of {len(feat_cols)} features.</li>
        <li><b>NumAromaticRings</b> ranked <b>#{numaro_rank}</b> in importance out of {len(feat_cols)} features.</li>
    </ul>
    <div class="img-container">
        <h4>Top SHAP Features (Forced Aromatic Proportion)</h4>
        <img src="data:image/png;base64,{shap_img}">
    </div>
    </body></html>
    """
    
    with open(HTML_OUT, "w") as f: 
        f.write(html)
    print(f"\nPipeline Complete. Full report saved to {HTML_OUT}")

if __name__ == "__main__":
    main()